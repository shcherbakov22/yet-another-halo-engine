# Accuracy gate

Checks that a candidate HAL set still computes the model correctly, from all-position logits on held-out text. Use it for any change that is not bit-identical (hidden md5).

- Corpus: 16 disjoint 2048-token windows and one 8192-token window of wikitext-2 (`corpus/`, md5s in `corpus/manifest.json`), tokenized with llama.cpp `llama-tokenize --ids --no-bos`.
- Golden: a frozen Loom output of a reference set (positions 1536..2047 of each window), kept in `~/yah-scratch/golden4` (windows 00-03, the production set of 2026-10-04; it differs from the Oct 1 set by mean KL 5.7e-7, so compare against a golden of the current default).
- Tiers: T1 (rounding level, windows 00-03) for exact-math rewrites with a different rounding; T2 (quantization level, windows 00-15 plus the 8K window) for changes that quantize. The limits are in `thresholds_T1.json` / `thresholds_T2.json`. T1 was calibrated from the distance between the old HIP engine and the golden (`reference_distance` inside the file): mean KL 5.3e-7, p99.9 1.8e-5, 0 top-1 flips. T2 is provisional.
- N (`thresholds_N.json`, the default bar for non-bit-exact changes): mean KL <= 2e-6, p99.9 <= 2e-4, 0 top-1 flips, |ln PPL ratio| <= 1e-4, always on all 16 windows (`--windows 00,...,15`, WINDOWS for gate_run.sh) against a golden of the current default. The budget is cumulative: gate a candidate together with every non-bit-exact change already shipped. Calibrated on two legitimate engine versions (Oct 1 vs Oct 4: mean KL 1.95e-6, p99.9 1.8e-4, 0 flips on 12 windows); 4 windows undersample the tail.

Run:

```
gate_run.sh <set> <golden_dir>                       # once, for a new golden (WINDOWS="00 01 02 03" or 00..15; L0 needs SET8K)
gate_run.sh <candidate set> <cand_dir>
../accgate2.py check <golden_dir> <cand_dir> thresholds_T1.json
../accgate2.py stats <golden_dir> <cand_dir>         # the raw distances
```

Reading the result:

- `kl_mean` saturates at ~4e-7 for any attention change that is not bit-exact; a single rounding change in attention (o * (1/sum) instead of o / sum) already gives 4.24e-7. Judge such changes on `kl_p999`, top-1 flips and PPL.
- Exact-change references (hidden md5 of the current default sets, `ids2048` / `ids8192` prompts): pp2048 `ac36332b6b5092a4`, pp8192 `963b7396625e2333`.
- Recalibrating T1 needs a second independent reference; the HIP engine that produced it is gone (tag `hip-final`).
