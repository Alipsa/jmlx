# Phase 6.4 decoder benchmark

The opt-in `:jmlx-examples:benchmarkDecode` task measures a local checkpoint. It downloads
nothing. The first model load in a fresh JVM is timed before any checkpoint hashing; this is a
first-in-process measurement with the OS page cache in its observed state, not a flushed-cache
measurement. Public `forward` evaluates its outputs before each prefill and one-token decode
timer stops. That forward computes logits for every prompt position, whereas generation projects
only the last hidden state, so the reported `prefill_ns` and prefill peak memory overstate real
prefill cost (the report records this as `"prefill_logits": "all_positions"`); compare them only
with each other. Sustained generation samples active native bytes after every emitted token and reports
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

## Quantized KV retention probe

The pinned macOS runtime probe ran in [CI run 125](https://github.com/Alipsa/jmlx/actions/runs/36777592834)
and the padded-head follow-up in [CI run 126](https://github.com/Alipsa/jmlx/actions/runs/36778017501).
Both native jobs and the required-suite assertions passed. Their `kv-cache-quantization-probe`
artifacts contain the raw JUnit XML and measurements. All seven tiny HF reference checkpoints use
16-dimensional K/V heads.

| Cache shape and affine setting | Packed bytes for one 64-token K | Float bytes | Maximum round-trip error | Direct packed SDPA |
| --- | ---: | ---: | ---: | --- |
| D=16, group=16, 4 or 8 bits | — | 4,096 | — | Quantize rejected: native groups are 32, 64, 128 |
| D=16 padded to 32, group=32, 4 bits | 1,536 | 4,096 | 0.00832210 | Not attempted; packed last dimension is 4, not the query's 16 |
| D=32, group=32, 4 bits | 1,536 | 8,192 | 0.00924163 | Rejected: packed last dimension 4, query dimension 32 |
| D=64, group=64, 4 bits | 2,560 | 16,384 | 0.0214118 | Rejected: packed last dimension 8, query dimension 64 |

Eight-bit D=32/64 cases also packed successfully, with smaller round-trip error (0.000670463
for group 32), but direct SDPA rejected their packed dimensions. The padded D=16 probe observed a
26,128-byte process-wide peak during dequantization; that figure includes the still-live source
and padding arrays, so it is **not** a controlled float-versus-quantized peak comparison. The
representation itself proves the limiting transient: a D=16 cache padded to D=32 needs 1,536
packed bytes plus an 8,192-byte dequantized tensor for each 64-token K, before attention's other
buffers. A float D=16 K needs 4,096 bytes. For D=32/64, the dequantized tensor alone is the size
of the original float cache, and packed storage remains live alongside it. Full-cache
dequantization also adds work to every token.

The probe therefore does not establish a useful decode-memory or throughput path on the pinned
runtime. `GenerationCachePolicy.quantized(bits, groupSize)` and `KVCachePolicy.quantized(bits,
groupSize)` fail immediately with a named unsupported-capability exception. No request silently
uses float retention. Append, reorder, fork, and scope ownership of a quantized cache were not
probed because the representation fails the attention-memory gate first. A future fused packed-KV
attention kernel would require a new capability probe and accuracy/ownership tests before enabling
the policy. A preallocated `slice_update` buffer or sliding ring buffer remains a post-baseline
float-cache optimization candidate.
