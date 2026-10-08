// LoomDecoder: the GEMV-based single-token decode step on Loom through HRX.
//
// HAL set: tools/emit_decode.py. decode.txt "ctx T": the paged KV pools hold T rows.
// decode.txt "kv q KB VB": quantized KV in the prefill's kv8a16 / kv4a16 formats (gen_kvq.py).
// Per layer:
//   rmsnorm -> full attention: attn_q / attn_k / attn_v GEMV, unpack q|gate, QK norm + RoPE,
//                              append K / V to the paged pools, split-K attention + reduce (gate),
//                              attn_output GEMV += hidden
//           -> recurrent:      attn_qkv / attn_gate / ssm_alpha / ssm_beta GEMV,
//                              conv + DeltaNet in one kernel (gated norm inside), ssm_out GEMV += hidden
//   rmsnorm -> ffn_gate|ffn_up SwiGLU GEMV -> ffn_down GEMV += hidden
// Head: rmsnorm -> output GEMV -> argmax into a device token stream.
// The next step's embedding kernel (IQ4_XS token_embd) reads that stream: steps queue back to back, no host round trip.
//
// The recurrent state and the KV pools are external (LoomDecoderState) and use the prefill's layouts.
// Use OwnState() for a prompt fed through decode, or bind the prefill's state (loom_forward_pp with YAH_GEN).
// Quantized KV: each layer's open 16-key V tile stays f16 until its 16th key arrives.
// So a bound state must start on a 16-key tile boundary (prefill runs are whole chunks).
#ifndef YAH_MODEL_LOOM_DECODER_HPP_
#define YAH_MODEL_LOOM_DECODER_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "core/config.hpp"
#include "core/gguf.hpp"
#include "model/loom_runtime.hpp"

namespace yah::model {

struct LoomDecoderState {
  std::vector<hrx_buffer_ref_t> kpool, vtpool;  // per full-attention layer, T * 1024 f16 each
  // Quantized KV (decode.txt "kv q KB VB"), per full-attention layer; the f16 pools are then unused.
  // K codes, K scales, K channel means subtracted before quantizing (zeros in OwnState), V^T codes, V^T stats.
  std::vector<hrx_buffer_ref_t> kq, ks, km, vq, vs;
  hrx_buffer_ref_t ptab{};           // T / 256 i32 (logical page -> physical page)
  hrx_buffer_t convstate = nullptr;  // per recurrent layer si: si * 10240 * 4 f32
  hrx_buffer_t dstate = nullptr;     // per recurrent layer si: si * 48 * 128 * 128 f32
};

class LoomDecoder {
 public:
  static constexpr std::uint32_t kHidden = 5120, kFfn = 17408, kAttn = 6144, kQProj = 12288, kKv = 1024;
  static constexpr std::uint32_t kInner = 6144, kQkv = 10240, kHeadsV = 48, kTs = 48, kState = 128;
  static constexpr std::uint32_t kHeads = 24, kKvHeads = 4, kVocab = 248320;
  static constexpr std::uint32_t kConvState = kQkv * 4, kStateElems = kHeadsV * kState * kState;
  static constexpr std::uint32_t kR = 2, kW = 4;  // gen_gemv defaults: rows per wave, waves per workgroup

  LoomDecoder(LoomDevice& gpu, const core::Gguf& gguf, const core::Qwen35Config& cfg, std::string dir,
              hrx_buffer_t weights, std::size_t delta)
      : gpu_(gpu), gguf_(gguf), cfg_(cfg), dir_(std::move(dir)), weights_(weights), delta_(delta) {
    {
      const auto txt = ReadAll(dir_ + "/decode.txt");
      std::istringstream in(std::string(txt.begin(), txt.end()));
      std::string k;
      in >> k >> T_;
      if (k != "ctx" || T_ == 0 || T_ % 256) throw LoomError("bad decode.txt in " + dir_);
      // "rw <kind> R W": the GEMV launch geometry of each kind in this set.
      std::string kind;
      std::uint32_t r = 0, w = 0;
      while (in >> k >> kind >> r >> w) {
        if (k == "rw") rw_[kind] = {r, w};
        if (k == "grid") grid_[kind] = r;  // the exact grid each GEMV kernel was compiled for
        if (k == "kv") {
          kbits_ = r;
          vbits_ = w;
        }  // "kv q KB VB"
      }
      const auto ok = [](std::uint32_t b) { return b == 4 || b == 8; };
      if (quant() && !(ok(kbits_) && ok(vbits_))) throw LoomError("decode.txt: kv formats must both be 8 or 4");
    }
    const std::uint32_t npg = T_ / 256;
    if (static_cast<std::uint32_t>(Find("token_embd.weight")->type) != 23)
      throw LoomError("token_embd: only IQ4_XS is wired");
    const char* tnames[5] = {"grid_iq3s.bin", "grid_iq3xxs.bin", "grid_iq2xxs.bin", "grid_iq2xs.bin",
                             "ksigns_iq2xs.bin"};
    for (const char* tn : tnames) {
      const auto data = ReadAll(dir_ + "/" + tn);
      LoomBuffer& b = Alloc(data.size());
      gpu_.H2D(b, data.data(), data.size());
      tabs_.push_back({b.handle, 0, data.size()});
    }
    hidden_ = &Alloc(kHidden * 4);
    normed_ = &Alloc(kHidden * 4);
    qg_ = &Alloc(kQProj * 4);
    q_ = &Alloc(kAttn * 4);
    gate_ = &Alloc(kAttn * 4);
    kb_ = &Alloc(kKv * 4);
    vb_ = &Alloc(kKv * 4);
    aout_ = &Alloc(kAttn * 4);
    qkv_ = &Alloc(kQkv * 4);
    alpha_ = &Alloc(kTs * 4);
    beta_ = &Alloc(kTs * 4);
    ssmout_ = &Alloc(kInner * 4);
    ffnact_ = &Alloc(kFfn * 4);
    logits_ = &Alloc(std::size_t{kVocab} * 4);
    sink_ = &Alloc(4);
    sparams_ = &Alloc(16);
    toks_ = &Alloc((std::size_t{T_} + 1) * 4);
    posarr_ = &Alloc(std::size_t{T_} * 4);
    eps_ = &Alloc(4);
    c32a_ = &Alloc(kKv * 4);
    c32b_ = &Alloc(kKv * 4);
    c16a_ = &Alloc(kKv * 2);
    c16b_ = &Alloc(kKv * 2);
    acc_ = &Alloc(std::size_t{npg} * kHeads * 256 * 4);
    ml_ = &Alloc(std::size_t{npg} * kHeads * 2 * 4);
    if (quant())
      for (std::uint32_t i = 0; i < full_layers(); ++i) vopen_.push_back(Ref(Alloc(std::size_t{kKv} * 16 * 2)));
    const float e = 1.0e-6f;
    gpu_.H2D(*eps_, &e, 4);
    std::vector<std::int32_t> pv(T_);
    for (std::uint32_t i = 0; i < T_; ++i) pv[i] = static_cast<std::int32_t>(i);
    gpu_.H2D(*posarr_, pv.data(), pv.size() * 4);
    trace_ = std::getenv("YAH_DEC_TRACE") != nullptr;
    Load("sample");   // now, not in the first sampled request (a set emitted before sampling stops here)
  }

  [[nodiscard]] std::uint32_t context() const { return T_; }
  // KV bits of this set: (16, 16) for fp16, else (8 | 4, 8 | 4).
  [[nodiscard]] std::pair<std::uint32_t, std::uint32_t> kv_bits() const { return {kbits_, vbits_}; }
  [[nodiscard]] bool quant() const { return kbits_ != 16; }
  // Quantized V: layer ai's open 16-key tile ([1024 dims][16 keys] f16), for seeding after a prefill that ends mid-tile.
  [[nodiscard]] hrx_buffer_ref_t OpenTile(std::uint32_t ai) const { return vopen_.at(ai); }
  // The next Step may start at pos even inside a tile: the open tiles hold that tile's earlier keys (seeded).
  void ResumeAt(std::uint32_t pos) { next_pos_ = pos; }
  // Quantized pool bytes per layer (gen_kvq.py layouts).
  std::size_t KqBytes() const { return std::size_t{T_} * (kbits_ == 4 ? 512 : 1024); }
  std::size_t KsBytes() const { return std::size_t{T_} * (kbits_ == 4 ? 128 : 32); }
  std::size_t VqBytes() const { return std::size_t{T_} * (vbits_ == 4 ? 512 : 1024); }
  std::size_t VsBytes() const { return std::size_t{T_} * 256; }
  std::uint32_t full_layers() const { return cfg_.main_block_count() / cfg_.full_attention_interval; }
  std::uint32_t recurrent_layers() const { return cfg_.main_block_count() - full_layers(); }

  // The decoder's own pools / page table (identity) / zeroed recurrent state.
  LoomDecoderState OwnState() {
    LoomDecoderState st;
    const std::uint32_t npg = T_ / 256;
    const std::size_t poolb = std::size_t{T_} * kKv * 2;
    for (std::uint32_t i = 0; i < full_layers(); ++i) {
      if (quant()) {
        st.kq.push_back(Ref(Alloc(KqBytes())));
        st.ks.push_back(Ref(Alloc(KsBytes())));
        st.km.push_back(Ref(Alloc(4096)));
        st.vq.push_back(Ref(Alloc(VqBytes())));
        st.vs.push_back(Ref(Alloc(VsBytes())));
      } else {
        st.kpool.push_back(Ref(Alloc(poolb)));
        st.vtpool.push_back(Ref(Alloc(poolb)));
      }
    }
    LoomBuffer& pt = Alloc(std::size_t{npg} * 4);
    std::vector<std::int32_t> pages(npg);
    for (std::uint32_t i = 0; i < npg; ++i) pages[i] = static_cast<std::int32_t>(i);
    gpu_.H2D(pt, pages.data(), pages.size() * 4);
    st.ptab = Ref(pt);
    st.convstate = Alloc(std::size_t{recurrent_layers()} * kConvState * 4).handle;
    st.dstate = Alloc(std::size_t{recurrent_layers()} * kStateElems * 4).handle;
    return st;
  }

  // External state: sizes are checked against this set's context.
  void Bind(const LoomDecoderState& st) {
    const std::size_t poolb = std::size_t{T_} * kKv * 2;
    const std::uint32_t nf = full_layers();
    if (quant()) {
      if (st.kq.size() != nf || st.ks.size() != nf || st.km.size() != nf || st.vq.size() != nf || st.vs.size() != nf)
        throw LoomError("decoder: quantized K / V pools per full-attention layer expected");
      for (std::uint32_t i = 0; i < nf; ++i)
        if (st.kq[i].length < KqBytes() || st.ks[i].length < KsBytes() || st.km[i].length < 4096 ||
            st.vq[i].length < VqBytes() || st.vs[i].length < VsBytes())
          throw LoomError("decoder: quantized KV pool smaller than the decode set's context " + std::to_string(T_));
    } else {
      if (st.kpool.size() != nf || st.vtpool.size() != nf)
        throw LoomError("decoder: one K and one V^T pool per full-attention layer expected");
      for (std::uint32_t i = 0; i < nf; ++i)
        if (st.kpool[i].length < poolb || st.vtpool[i].length < poolb)
          throw LoomError("decoder: KV pool smaller than the decode set's context " + std::to_string(T_));
    }
    if (st.ptab.length < std::size_t{T_ / 256} * 4) throw LoomError("decoder: page table too small");
    st_ = st;
    next_pos_ = kAnyTile;
    cs_[0] = st.convstate;
    if (!cs_[1]) cs_[1] = Alloc(std::size_t{recurrent_layers()} * kConvState * 4).handle;
    cs_cur_ = 0;
  }

  // Tokens into the device stream at positions at .. at + n - 1.
  void SetTokens(const std::uint32_t* toks, std::size_t n, std::uint32_t at) {
    if (at + n > T_ + 1) throw LoomError("decoder: token stream overflow");
    gpu_.H2D(*toks_, toks, n * 4, std::size_t{at} * 4);
  }
  std::vector<std::uint32_t> Tokens(std::uint32_t from, std::uint32_t to) {
    std::vector<std::uint32_t> out(to - from);
    gpu_.D2H(*toks_, out.data(), out.size() * 4, std::size_t{from} * 4);
    for (auto t : out)
      if (t >= kVocab) throw LoomError("decoder: token id out of range: " + std::to_string(t));
    return out;
  }
  // Token position pos of the device token stream (a step's argmax target).
  [[nodiscard]] hrx_buffer_ref_t TokenRef(std::uint32_t pos) const { return {toks_->handle, std::size_t{pos} * 4, 4}; }
  void CopyLogits(float* host) { gpu_.D2H(*logits_, host, std::size_t{kVocab} * 4); }

  // Queues the step at position pos. It reads toks[pos]; its argmax goes to toks[pos + 1] if pos + 1 >= keep_from.
  // Prompt positions (pos + 1 < keep_from) write the argmax to a sink.
  void Step(std::uint32_t pos, std::uint32_t keep_from) {
    if (pos >= T_) throw LoomError("decoder: position past the set's context");
    if (st_.kpool.empty() && st_.kq.empty()) throw LoomError("decoder: no state bound");
    // The open V tile holds this tile's earlier keys only if the steps were contiguous.
    if (quant() && pos % 16 && pos != next_pos_)
      throw LoomError("decoder: quantized KV steps must be contiguous from a 16-key tile boundary");
    next_pos_ = pos + 1;
    cur_pos_ = pos;
    const hrx_buffer_ref_t dposr{posarr_->handle, std::size_t{pos} * 4, 4};
    Dispatch(Load("embed"), 1, 1, kHidden / 16,
             {TRef("token_embd.weight"), {toks_->handle, std::size_t{pos} * 4, 4}, Ref(*hidden_)});
    const std::uint32_t nl = cfg_.main_block_count();
    for (std::uint32_t l = 0; l < nl; ++l) {
      const std::string pre = "blk." + std::to_string(l) + ".";
      Rmsnorm(*hidden_, pre + "attn_norm.weight", *normed_);
      Tr("in", l, *hidden_, kHidden);
      if (cfg_.IsFullAttention(l)) {
        const std::uint32_t ai = l / cfg_.full_attention_interval;
        Project({pre + "attn_q.weight", pre + "attn_k.weight", pre + "attn_v.weight"}, {qg_, kb_, vb_});
        Dispatch(Load("unpack"), kHeads, 1, 256, {Ref(*qg_), Ref(*q_), Ref(*gate_)});
        Dispatch(Load("rope"), kHeads + kKvHeads, 1, 256,
                 {Ref(*q_), Ref(*kb_), Ref(*vb_), TRef(pre + "attn_q_norm.weight"), TRef(pre + "attn_k_norm.weight"),
                  Ref(*q_), Ref(*kb_), Ref(*c32a_), Ref(*c32b_), Ref(*c16a_), Ref(*c16b_), dposr, Ref(*eps_)});
        if (quant()) {
          Dispatch(Load("dattn_kappend_q"), 1, 1, 128,
                   {Ref(*kb_), st_.km[ai], st_.kq[ai], st_.ks[ai], st_.ptab, dposr});
          gpu_.NoBarrierNext();  // independent of the K append
          Dispatch(Load("dattn_vappend_q"), kKvHeads, 1, 256,
                   {Ref(*vb_), vopen_[ai], st_.vq[ai], st_.vs[ai], st_.ptab, dposr});
          Dispatch(Load("dattn_part_q"), kKvHeads, pos / 256 + 1, 256,
                   {Ref(*q_), st_.kq[ai], st_.ks[ai], st_.vq[ai], st_.vs[ai], vopen_[ai], st_.ptab, dposr, Ref(*acc_),
                    Ref(*ml_)});
        } else {
          Dispatch(Load("dattn_kvappend"), kKvHeads, 1, 256,
                   {Ref(*kb_), Ref(*vb_), st_.kpool[ai], st_.vtpool[ai], st_.ptab, dposr});
          Dispatch(Load("dattn_part"), kKvHeads, pos / 256 + 1, 256,
                   {Ref(*q_), st_.kpool[ai], st_.vtpool[ai], st_.ptab, dposr, Ref(*acc_), Ref(*ml_)});
        }
        Dispatch(Load("dattn_reduce"), kHeads, 1, 256, {Ref(*acc_), Ref(*ml_), Ref(*gate_), dposr, Ref(*aout_)});
        Tr("attn", l, *aout_, kAttn);
        Gemv("resid", {pre + "attn_output.weight"}, *aout_, *hidden_);
      } else {
        const std::uint32_t si = l - l / cfg_.full_attention_interval;
        Project({pre + "attn_qkv.weight", pre + "attn_gate.weight", pre + "ssm_alpha.weight", pre + "ssm_beta.weight"},
                {qkv_, gate_, alpha_, beta_});
        const hrx_buffer_ref_t dst{st_.dstate, std::size_t{si} * kStateElems * 4, std::size_t{kStateElems} * 4};
        // Conv state ping-pong: this step reads cs_[cs_cur_] and writes the other one.
        const std::size_t co = std::size_t{si} * kConvState * 4, cb = std::size_t{kConvState} * 4;
        Dispatch(Load("deltanet_conv"), kHeadsV, 1, 512,
                 {{qkv_->handle, 0, std::size_t{kQkv} * 4},
                  TRef(pre + "ssm_conv1d.weight"),
                  {cs_[cs_cur_], co, cb},
                  {cs_[1 - cs_cur_], co, cb},
                  dst,
                  Ref(*alpha_),
                  Ref(*beta_),
                  TRef(pre + "ssm_a"),
                  TRef(pre + "ssm_dt.bias"),
                  TRef(pre + "ssm_norm.weight"),
                  Ref(*gate_),
                  Ref(*ssmout_)});
        Tr("ssm", l, *ssmout_, kInner);
        Gemv("resid", {pre + "ssm_out.weight"}, *ssmout_, *hidden_);
      }
      Rmsnorm(*hidden_, pre + "post_attention_norm.weight", *normed_);
      Gemv("swiglu", {pre + "ffn_gate.weight", pre + "ffn_up.weight"}, *normed_, *ffnact_);
      Tr("ffnact", l, *ffnact_, kFfn);
      Gemv("resid", {pre + "ffn_down.weight"}, *ffnact_, *hidden_);
    }
    Rmsnorm(*hidden_, "output_norm.weight", *normed_);
    Gemv("plain", {"output.weight"}, *normed_, *logits_);
    Tr("logits", 99, *logits_, kVocab);
    cs_cur_ = 1 - cs_cur_;
    if (sampling_ && pos + 1 >= keep_from)
      Sample(Ref(*logits_), pos, TokenRef(pos + 1));
    else
      Dispatch(Load("argmax"), 1, 1, 1024, {Ref(*logits_), pos + 1 >= keep_from ? TokenRef(pos + 1) : Ref(*sink_)});
  }

  // From now on the steps whose token is kept sample it (gen_decode_misc.gen_sample: softmax(logits / t) cut to its
  // top_p nucleus, an exact draw seeded by seed and the step's position) instead of taking the argmax; t <= 0: argmax.
  // In stream order, so a run of steps queued before keeps its mode.
  void SetSampling(float t, float top_p, std::uint64_t seed) {
    sampling_ = t > 0.0f;
    if (!sampling_) return;
    Load("sample");
    std::uint32_t p[4];
    std::memcpy(&p[0], &t, 4), std::memcpy(&p[1], &top_p, 4);
    p[2] = static_cast<std::uint32_t>(seed), p[3] = static_cast<std::uint32_t>(seed >> 32);
    gpu_.Update(*sparams_, p, sizeof p);
  }
  // Samples a token from logits (SetSampling's parameters) into dst; at: the position of the step that produced the
  // logits (the draw's counter).
  void Sample(const hrx_buffer_ref_t& logits, std::uint32_t at, const hrx_buffer_ref_t& dst) {
    if (at >= T_) throw LoomError("decoder: position past the set's context");
    Dispatch(Load("sample"), 1, 1, 1024, {logits, Ref(*sparams_), {posarr_->handle, std::size_t{at} * 4, 4}, dst});
  }

 private:
  struct Fmt {
    const char* name;
    std::uint32_t qk, bb, tables;
  };
  // GGML type -> GEMV name, block elements, block bytes, table bits.
  // Table bits follow gen_gemv.TABLE_ORDER: grid_iq3s, grid_iq3xxs, grid_iq2xxs, grid_iq2xs, ksigns.
  static bool FmtOf(std::uint32_t type, Fmt* f) {
    switch (type) {
      case 8:
        *f = {"q8_0", 32, 34, 0};
        return true;
      case 10:
        *f = {"q2k", 256, 84, 0};
        return true;
      case 11:
        *f = {"q3k", 256, 110, 0};
        return true;
      case 12:
        *f = {"q4k", 256, 144, 0};
        return true;
      case 13:
        *f = {"q5k", 256, 176, 0};
        return true;
      case 14:
        *f = {"q6k", 256, 210, 0};
        return true;
      case 16:
        *f = {"iq2xxs", 256, 66, 4 | 16};
        return true;
      case 17:
        *f = {"iq2xs", 256, 74, 8 | 16};
        return true;
      case 18:
        *f = {"iq3xxs", 256, 98, 2 | 16};
        return true;
      case 21:
        *f = {"iq3s", 256, 110, 1};
        return true;
      case 23:
        *f = {"iq4xs", 256, 136, 0};
        return true;
      default:
        return false;
    }
  }
  static std::vector<char> ReadAll(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw LoomError("cannot open " + path);
    return std::vector<char>(std::istreambuf_iterator<char>(in), {});
  }
  static hrx_buffer_ref_t Ref(const LoomBuffer& b) { return {b.handle, 0, b.size}; }
  LoomBuffer& Alloc(std::size_t bytes) {
    keep_.push_back(gpu_.Allocate(bytes));
    std::vector<char> z(bytes, 0);
    gpu_.H2D(keep_.back(), z.data(), bytes);
    return keep_.back();
  }
  const core::TensorInfo* Find(const std::string& n) const {
    const auto* t = gguf_.Find(n);
    if (!t) throw LoomError("tensor not found: " + n);
    return t;
  }
  hrx_buffer_ref_t TRef(const std::string& n) const {
    const auto* t = Find(n);
    return {weights_, delta_ + static_cast<std::size_t>(t->offset), static_cast<std::size_t>(t->bytes)};
  }
  LoomExecutable& Load(const std::string& name) {
    auto it = exes_.find(name);
    if (it == exes_.end()) it = exes_.emplace(name, gpu_.Load(dir_ + "/" + name + ".hal")).first;
    return it->second;
  }
  // Workgroup size and binding count must match the compiled export.
  // Example: argmax is a 1024-lane kernel; with 32 lanes its per-wave LDS slots stay unwritten and it returns garbage.
  void Dispatch(LoomExecutable& e, std::uint32_t gx, std::uint32_t gy, std::uint32_t wg,
                const std::vector<hrx_buffer_ref_t>& b) {
    const std::uint32_t cw = e.WorkgroupSize(0), cb = e.BindingCount(0);
    if ((cw && cw != wg) || (cb && cb != b.size()))
      throw LoomError("dispatch mismatch: workgroup " + std::to_string(wg) + " vs compiled " + std::to_string(cw) +
                      ", bindings " + std::to_string(b.size()) + " vs " + std::to_string(cb));
    gpu_.Dispatch(e, 0, LoomDevice::Config(gx, gy, 1, wg, 1, 1), nullptr, 0, b.data(), b.size());
  }
  // GEMV kernel names and weight footprints follow tools/gen_gemv.py.
  void Gemv(const char* kind, const std::vector<std::string>& ws, const LoomBuffer& x, const LoomBuffer& y) {
    std::string name = std::string("gv_") + kind;
    std::uint32_t tbits = 0, M = 0, K = 0;
    std::vector<hrx_buffer_ref_t> b;
    for (const auto& w : ws) {
      const auto* t = Find(w);
      Fmt f{};
      if (!FmtOf(static_cast<std::uint32_t>(t->type), &f)) throw LoomError("no GEMV format for " + w);
      K = static_cast<std::uint32_t>(t->dims[0]);
      M = static_cast<std::uint32_t>(t->dims[1]);
      if (t->bytes != std::uint64_t{K} / f.qk * f.bb * M) throw LoomError("footprint mismatch on " + w);
      name += std::string("_") + f.name;
      tbits |= f.tables;
      b.push_back(TRef(w));
    }
    name += "_" + std::to_string(M) + "_" + std::to_string(K);
    for (int i = 0; i < 5; ++i)
      if (tbits & (1u << i)) b.push_back(tabs_[i]);
    if (x.size < std::size_t{K} * 4 || y.size < std::size_t{M} * 4) throw LoomError("GEMV operand too small: " + name);
    b.push_back({x.handle, 0, std::size_t{K} * 4});
    b.push_back({y.handle, 0, std::size_t{M} * 4});
    LoomExecutable& exe = Load(name);
    auto [R, W] = Rw(kind);
    const std::uint32_t gx = M / (R * W);
    CheckGrid(name, gx);
    Dispatch(exe, gx, 1, 32 * W, b);
  }
  // A layer's input projections of normed_: one band-fused GEMV (gen_gemv gen_bands).
  void Project(const std::vector<std::string>& ws, const std::vector<LoomBuffer*>& ys) {
    std::string fn, mn;
    std::uint32_t tbits = 0, K = 0, rows = 0;
    std::vector<hrx_buffer_ref_t> b;
    for (std::size_t i = 0; i < ws.size(); ++i) {
      const auto* t = Find(ws[i]);
      Fmt f{};
      if (!FmtOf(static_cast<std::uint32_t>(t->type), &f)) throw LoomError("no GEMV format for " + ws[i]);
      const std::uint32_t k = static_cast<std::uint32_t>(t->dims[0]), M = static_cast<std::uint32_t>(t->dims[1]);
      if (i && k != K) throw LoomError("bands need one K");
      K = k;
      if (t->bytes != std::uint64_t{K} / f.qk * f.bb * M) throw LoomError("footprint mismatch on " + ws[i]);
      if (M % (Rw("bands").first * Rw("bands").second) || ys[i]->size < std::size_t{M} * 4)
        throw LoomError("band output: " + ws[i]);
      fn += std::string("_") + f.name;
      mn += "_" + std::to_string(M);
      tbits |= f.tables;
      rows += M;
      b.push_back(TRef(ws[i]));
    }
    for (int i = 0; i < 5; ++i)
      if (tbits & (1u << i)) b.push_back(tabs_[i]);
    b.push_back({normed_->handle, 0, std::size_t{K} * 4});
    for (std::size_t i = 0; i < ws.size(); ++i) {
      const auto* t = Find(ws[i]);
      b.push_back({ys[i]->handle, 0, static_cast<std::size_t>(t->dims[1]) * 4});
    }
    auto [R, W] = Rw("bands");
    const std::string bn = "gb" + fn + mn + "_" + std::to_string(K);
    CheckGrid(bn, rows / (R * W));
    Dispatch(Load(bn), rows / (R * W), 1, 32 * W, b);
  }
  // (R, W) of a GEMV kind in this set (decode.txt).
  std::pair<std::uint32_t, std::uint32_t> Rw(const std::string& kind) const {
    auto it = rw_.find(kind);
    return it == rw_.end() ? std::make_pair(kR, kW) : it->second;
  }
  // Loom drops clamps it proves redundant from the launch grid: launch exactly the grid recorded in decode.txt.
  void CheckGrid(const std::string& name, std::uint32_t gx) const {
    auto it = grid_.find(name);
    if (it == grid_.end()) throw LoomError("decode set has no grid record for " + name + " (re-emit)");
    if (it->second != gx)
      throw LoomError("grid mismatch for " + name + ": " + std::to_string(gx) + " vs compiled " +
                      std::to_string(it->second));
  }
  void Rmsnorm(const LoomBuffer& x, const std::string& w, const LoomBuffer& out) {
    Dispatch(Load("rmsnorm"), 1, 1, 512, {Ref(x), TRef(w), Ref(out)});
  }
  // YAH_DEC_TRACE=1: at step 0, sync after every stage and print max |x| and the NaN count.
  void Tr(const char* what, std::uint32_t l, const LoomBuffer& b, std::size_t n) {
    if (!trace_ || cur_pos_ != 0) return;
    gpu_.Synchronize();
    std::vector<float> h(n);
    gpu_.D2H(b, h.data(), n * 4);
    double mx = 0;
    std::size_t nan = 0;
    for (float v : h) {
      if (v != v)
        ++nan;
      else
        mx = std::max(mx, static_cast<double>(std::fabs(v)));
    }
    std::fprintf(stderr, "trace l%-2u %-10s max %.4g nan %zu\n", l, what, mx, nan);
  }

  LoomDevice& gpu_;
  const core::Gguf& gguf_;
  const core::Qwen35Config& cfg_;
  std::string dir_;
  hrx_buffer_t weights_;
  std::size_t delta_;
  static constexpr std::uint32_t kAnyTile = 0xffffffffu;
  std::uint32_t T_ = 0, cur_pos_ = 0, kbits_ = 16, vbits_ = 16, next_pos_ = kAnyTile;
  std::vector<hrx_buffer_ref_t> vopen_;  // quantized V: each layer's open tile, [1024][16] f16
  bool trace_ = false;
  bool sampling_ = false;  // SetSampling
  hrx_buffer_t cs_[2] = {nullptr, nullptr};  // conv state ping-pong
  int cs_cur_ = 0;
  std::deque<LoomBuffer> keep_;  // deque: Alloc hands out references that must stay valid
  std::map<std::string, LoomExecutable> exes_;
  std::map<std::string, std::pair<std::uint32_t, std::uint32_t>> rw_;
  std::map<std::string, std::uint32_t> grid_;
  std::vector<hrx_buffer_ref_t> tabs_;
  LoomDecoderState st_;
  LoomBuffer *hidden_, *normed_, *qg_, *q_, *gate_, *kb_, *vb_, *aout_, *qkv_, *alpha_, *beta_, *ssmout_, *ffnact_,
      *logits_, *sink_, *sparams_, *toks_, *posarr_, *eps_, *c32a_, *c32b_, *c16a_, *c16b_, *acc_, *ml_;
};

}  // namespace yah::model

#endif  // YAH_MODEL_LOOM_DECODER_HPP_
