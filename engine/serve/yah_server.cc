// yah_server: the OpenAI Responses API (POST /v1/responses) over one TextGenerator, one generation at a time.
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "httplib/httplib.h"
#include "model/engine.hpp"
#include "model/generator.hpp"
#include "serve/chat_template.hpp"
#include "serve/fake_generator.hpp"
#include "serve/responses.hpp"

namespace {

using yah::core::TokenId;
using yah::model::GenerateParams;
using yah::model::GenerateResult;
using yah::model::TextGenerator;
using namespace yah::serve;

int Usage() {
  std::fprintf(stderr,
               "usage: yah_server --model <gguf> --prefill <hal set> --decode <hal set> [--npu] [--host 127.0.0.1] "
               "[--port 8080]\n"
               "       --npu: the NPU computes part of the prefill's GEMMs (a set emitted with YAH_NPU_SPLIT)\n"
               "       yah_server --model <gguf> --fake [--host ...] [--port ...]   canned replies, no GPU\n");
  return 2;
}

class Server {
 public:
  explicit Server(TextGenerator& generator) : generator_(generator) {
    for (const char* stop : {"<|im_end|>", "<|endoftext|>"}) {
      if (auto id = generator_.tokenizer().FindSpecial(stop)) stop_ids_.push_back(*id);
    }
    if (stop_ids_.empty()) throw std::runtime_error("the tokenizer has no <|im_end|>");
    created_ = std::time(nullptr);
  }

  void Responses(const httplib::Request& req, httplib::Response& res) {
    try {
      Json body;
      try {
        body = Json::parse(req.body);
      } catch (const Json::parse_error& error) {
        throw ApiError(400, std::string("Invalid JSON body: ") + error.what());
      }
      const ResponseRequest request = ParseResponseRequest(body);
      std::string text;
      try {
        text = RenderChat(request.messages, request.chat);
      } catch (const std::invalid_argument& error) {
        throw ApiError(400, error.what(), "input");
      }
      const std::vector<TokenId> prompt = generator_.tokenizer().Encode(text);

      const std::uint32_t context = generator_.context();
      const auto prompt_tokens = static_cast<std::uint32_t>(prompt.size());
      GenerateParams params;
      params.stop_ids = stop_ids_;
      params.sampling.temperature = static_cast<float>(request.temperature);
      params.sampling.top_p = static_cast<float>(request.top_p);
      if (prompt_tokens >= context) {
        throw ApiError(400,
                       "The input has " + std::to_string(prompt_tokens) + " tokens; the context is " +
                           std::to_string(context) + ".",
                       "input", "context_length_exceeded");
      }
      params.max_tokens = request.max_output_tokens.value_or(context - prompt_tokens);
      if (params.max_tokens > context - prompt_tokens) {
        throw ApiError(400,
                       "input (" + std::to_string(prompt_tokens) + " tokens) + max_output_tokens (" +
                           std::to_string(params.max_tokens) + ") exceeds the context of " + std::to_string(context) +
                           " tokens.",
                       "max_output_tokens", "context_length_exceeded");
      }

      if (!request.stream) {
        ResponseStream stream(request, nullptr);
        Run(stream, prompt, params);
        res.set_content(Dump(stream.Response()), "application/json");
        return;
      }
      res.set_header("Cache-Control", "no-cache");
      auto provider = [this, request, prompt, params](std::size_t, httplib::DataSink& sink) {
        Stream(request, prompt, params, sink);
        return true;
      };
      res.set_chunked_content_provider("text/event-stream", provider);
    } catch (const ApiError& error) {
      res.status = error.status;
      res.set_content(Dump(ErrorBody(error)), "application/json");
    }
  }

  void Models(const httplib::Request&, httplib::Response& res) {
    const Json model = {
        {"id", generator_.model_name()}, {"object", "model"}, {"created", created_}, {"owned_by", "yah"}};
    res.set_content(Dump({{"object", "list"}, {"data", Json::array({model})}}), "application/json");
  }

 private:
  void Stream(const ResponseRequest& request, const std::vector<TokenId>& prompt, const GenerateParams& params,
              httplib::DataSink& sink) {
    ResponseStream stream(request, [&sink](const std::string& event) {
      return sink.is_writable() && sink.write(event.data(), event.size());
    });
    try {
      Run(stream, prompt, params);
    } catch (const std::exception& error) {
      stream.Fail(error.what());
    }
    sink.done();
  }

  void Run(ResponseStream& stream, const std::vector<TokenId>& prompt, const GenerateParams& params) {
    std::lock_guard lock(mutex_);
    GenerateResult result;
    result.prompt_tokens = static_cast<std::uint32_t>(prompt.size());
    result.finish_reason = "stop";
    try {
      // A client that left while waiting for the lock costs no generation.
      if (stream.Start()) {
        result = generator_.Generate(
            prompt, params, [&](TokenId id) { return stream.OnToken(generator_.tokenizer().DecodeToken(id)); });
      }
    } catch (const std::exception& error) {
      std::fprintf(stderr, "yah_server: %s failed: %s\n", stream.id().c_str(), error.what());
      throw;
    }
    stream.Finish(result);
    const double tok_s = result.decode_ms > 0.0 ? result.generated_tokens * 1000.0 / result.decode_ms : 0.0;
    std::fprintf(stderr, "yah_server: %s prompt=%u output=%u prefill_ms=%.1f decode_tok_s=%.1f finish=%s\n",
                 stream.id().c_str(), result.prompt_tokens, result.generated_tokens, result.prefill_ms, tok_s,
                 stream.client_alive() ? result.finish_reason.c_str() : "cancelled");
  }

  TextGenerator& generator_;
  std::mutex mutex_;
  std::vector<TokenId> stop_ids_;
  std::int64_t created_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
  std::string model, prefill, decode, host = "127.0.0.1";
  int port = 8080;
  bool fake = false, npu = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const bool has_value = i + 1 < argc;
    if (arg == "--fake") {
      fake = true;
    } else if (arg == "--npu") {
      npu = true;
    } else if (arg == "--model" && has_value) {
      model = argv[++i];
    } else if (arg == "--prefill" && has_value) {
      prefill = argv[++i];
    } else if (arg == "--decode" && has_value) {
      decode = argv[++i];
    } else if (arg == "--host" && has_value) {
      host = argv[++i];
    } else if (arg == "--port" && has_value) {
      port = std::atoi(argv[++i]);
    } else {
      return Usage();
    }
  }
  if (model.empty() || (!fake && (prefill.empty() || decode.empty()))) return Usage();

  try {
    std::unique_ptr<TextGenerator> generator;
    if (fake) {
      generator = std::make_unique<FakeGenerator>(model);
    } else {
      yah::model::Engine::Options options;
      options.model = model;
      options.prefill_hal = prefill;
      options.decode_hal = decode;
      options.npu = npu;
      generator = std::make_unique<yah::model::Engine>(options);
    }
    Server server(*generator);
    httplib::Server http;
    http.Post("/v1/responses",
              [&](const httplib::Request& req, httplib::Response& res) { server.Responses(req, res); });
    http.Get("/v1/models", [&](const httplib::Request& req, httplib::Response& res) { server.Models(req, res); });
    http.Get("/health", [](const httplib::Request&, httplib::Response& res) {
      res.set_content(R"({"status":"ok"})", "application/json");
    });
    http.set_error_handler([](const httplib::Request&, httplib::Response& res) {
      if (!res.body.empty()) return httplib::Server::HandlerResponse::Unhandled;
      const ApiError error(res.status, res.status == 404 ? "Not found." : "HTTP error " + std::to_string(res.status));
      res.set_content(Dump(ErrorBody(error)), "application/json");
      return httplib::Server::HandlerResponse::Handled;
    });
    http.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
      std::string message = "internal error";
      try {
        std::rethrow_exception(ep);
      } catch (const std::exception& error) {
        message = error.what();
      } catch (...) {
      }
      res.status = 500;
      res.set_content(Dump(ErrorBody(ApiError(500, message, "", "server_error", "server_error"))), "application/json");
    });
    std::fprintf(stderr, "yah_server: %s%s on http://%s:%d\n", generator->model_name().c_str(), fake ? " (fake)" : "",
                 host.c_str(), port);
    if (!http.listen(host, port)) {
      std::fprintf(stderr, "yah_server: cannot listen on %s:%d\n", host.c_str(), port);
      return 1;
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "yah_server: %s\n", error.what());
    return 1;
  }
}
