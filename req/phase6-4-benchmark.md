# Phase 6.4 decoder benchmark

The opt-in `:jmlx-examples:benchmarkDecode` task measures a local checkpoint. It downloads
nothing. The first model load in a fresh JVM is timed before any checkpoint hashing; this is a
first-in-process measurement with the OS page cache in its observed state, not a flushed-cache
measurement. Public `forward` evaluates its outputs before each prefill and one-token decode
timer stops. Sustained generation samples active native bytes after every emitted token and reports
the allocator's peak active bytes separately from cached bytes.

On a bootstrapped macOS Apple Silicon host, run each synthetic family in a fresh JVM:

```sh
./scripts/bootstrap-native.sh
for family in llama qwen2 mistral gemma phi3 mixtral; do
  ./gradlew :jmlx-examples:benchmarkDecode \
    --args="${PWD}/tools/hf-reference/goldens/checkpoints/${family} ${PWD}/build/phase6-4-${family} 1,7,42,3,19,5 32 5 2 full"
done
```

Each run writes `<output-prefix>.json` and `<output-prefix>.md`. The JSON records the checkpoint
file hashes, runtime pins, sample distributions, tokens/s, load/prefill/decode times, peak active
native memory, and per-token active-memory samples. These tiny checkpoints measure synthetic
fixture performance; their numbers do not represent production-size models. Repeat with pinned
Tier-B Llama, Qwen2, and Mistral directories when they are available on the runner, saving raw
JSON, commands, hardware, and checkpoint hashes. Inspect the Tier-B Mistral `sliding_window`
setting before running or labeling an eviction benchmark.

The implementation environment for this change is Linux without the pinned macOS MLX runtime.
Consequently no native throughput or memory results are recorded here. The macOS CI benchmark
smoke step validates output fields without speed thresholds; performance comparisons remain an
explicit native-run acceptance item.

Quantized KV retention is not exposed as a policy yet. The pinned runtime's quantize/dequantize
representation, SDPA input requirements, and transient dequantization peak still require a native
probe before an accuracy or memory contract can be published. A preallocated `slice_update`
buffer or sliding ring buffer likewise remains a post-baseline optimization candidate.
