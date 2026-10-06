# Phase 7.2 implementation evidence — 2026-10-05

Implementation and local validation are complete. The new macOS 26 CI acceptance is pending;
no publication, native repin or generated-binding change is included.

Phase 7.1's prerequisite passed on the starting HEAD
`961c644eae7b92d1f3ff1fa6a24a8b2733df5b91`, including macOS 26 native-suite assertions:
https://github.com/Alipsa/jmlx/actions/runs/37343853857. That run does not validate these new
working-tree changes. Local checks used Apple M5 Max, macOS 27.0.1, aarch64 and Java 25.

## Delivered work packages

1. CPU float32 eager references, tiny BERT/T5 checkpoints, semantic probes/source provenance and
   actual sentence-transformers 5.1.2 references. The universal hash lock preserves Linux CPU
   Torch and permits Darwin Torch. Existing decoder goldens are unchanged.
2. Real Flan-T5 Precompiled Darts charsmap/alignment, malformed-table rejection, BERT pair API
   and all three truncation strategies. Locked Rust cases cover Unicode, errors/ties and
   per-input offsets/sequence IDs. Legacy expected fixtures are byte-identical. Derived inputs
   reproduce exactly and include source hashes, sizes and Apache-2.0 attribution.
3. Key-only masks, rectangular explicit-scale attention with additive bias, pre/post-norm blocks
   and evaluated request-owned static cross K/V. Existing constructor defaults remain unchanged.
4. Java downloader, full JSON parser retaining strict inventory behavior, safe nested paths,
   Redirect.NORMAL, final-response checks and bounded streaming. Non-2xx final statuses reject
   unfollowed 3xx. Local cross-host HTTP redirects and the JDK redirect policy are tested; no
   local TLS/downgrade integration claim. The opt-in task has no tracked outputs/cache state.
5. Validated BERT tensor/config plans, embeddings/post-LN blocks, sentence pooling/Normalize,
   tanh-pooler sequence head, token head, defensive results, labels and pre-native input checks.
6. T5 RMS/pre-norm stacks, rectangular projections, logarithmic buckets, shared relative bias,
   ReLU/gated-GELU, tied/untied heads and full/cached forward agreement.
7. Generation sampler/output integration, decoder-only penalties, separate source/target budgets,
   request ownership, cancellation/failure cleanup, projection counts and explicit scheduler
   and sliding-policy rejection.
8. Required-suite result directories, separate weekly task jobs, API examples/changelogs,
   compatibility/tokenizer amendments and measured resource/precision evidence. Versions stay
   unchanged; the parent's scheduler 4 MiB follow-up remains separate.

## Config and tensor mapping

| Component | HF keys/config | Java mapping |
| --- | --- | --- |
| BERT dimensions | vocab_size, hidden_size, intermediate_size, layer/head/type/position counts | Positive integer schema, head divisibility, positive finite epsilon and task-specific architecture selection |
| BERT embeddings | embeddings.{word,position,token_type}_embeddings.weight; embeddings.LayerNorm | Sum with input-index absolute positions, then LayerNorm |
| BERT blocks | encoder.layer.N attention query/key/value, output dense/LN, intermediate dense, output dense/LN | Weight/bias shape plan, scale 1/sqrt(headDim), GELU and post-norm residuals |
| BERT heads | pooler.dense, classifier; bert. prefix for task backbones | CLS index 0/tanh for sequence head; final per-token hidden states for token head |
| BERT buffers | embeddings.position_ids / token_type_ids | Optional I32/I64 [1,positions] arange/zeros checked from Java header/payload; not floating parameters |
| Labels | num_labels, problem_type, id2label/label2id | Complete consistent numeric ordering, LABEL_n fallback; softmax/sigmoid; implicit and declared regression rejected |
| Sentence pipeline | modules.json, Pooling config, sentence_bert_config | Root Transformer → one pooling → optional Normalize; dimension/path/limit checks; explicit encode policy overrides defaults |
| T5 dimensions | d_model, d_kv, num_heads, d_ff, encoder/decoder layer counts | q/k/v width heads*d_kv independent of d_model; source limit independent of positions |
| T5 embedding/head | shared.weight, encoder/decoder aliases, lm_head.weight | Equal alias shape/value checks; tied hidden scaling d_model^-0.5; untied independent head |
| T5 attention | SelfAttention and decoder EncDecAttention q/k/v/o | Bias-free projections, scale 1; causal target/key-only source masks; once-per-layer static cross K/V |
| T5 bias/norm | block0 SelfAttention relative_attention_bias; layer/final norms | Signed/log buckets shared per stack; RMS without mean subtraction or bias |
| T5 FFN | encoder layer1 / decoder layer2 DenseReluDense wi/wo or wi_0/wi_1/wo | ReLU or gated gelu_new selected by feed_forward_proj |
| T5 start | generation_config decoder_start_token_id then config fallback | In-vocabulary integer checked before native loading; decoder history only |

## Real artifact and margin evidence

Immutable pins, files, licenses, source hashes and full reference arrays are in `tools/tier-b`.
Intel MRPC at `844e804944328e45c3f3bf69b35467e0ce0b26af` lacks safetensors; the permitted
Apache-2.0 SST-2 fallback is selected. Pair encoding remains independently oracle-verified.

| Artifact | Bytes | Maximum absolute error | JVM RSS snapshot KiB | Native peak bytes | Minimum gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| all-MiniLM-L6-v2 | 91346791 | 1.9371509552001953e-7 | 304576 | 99257899 | Vector only |
| bert-base-uncased-sst2 | 438431729 | 4.76837158203125e-7 | 643216 | 470252210 | 14.706037521362305 |
| flan-t5-small | 310308220 | 3.814697265625e-5 | 651104 | 317606623 | 0.22959327697753906 |

RSS is post-comparison, not a high-water measurement. Jobs run separately; macOS 26 runner
measurements remain necessary. MiniLM uses actual SentenceTransformer. Flan compares every
full CPU logit history, complete greedy IDs and finish reason, permitting early EOS.

The two-stage gate is complete locally: WP1 selected candidates using provisional epsilon=1e-4
and margin=1e-4; WP5/WP7 measured real-checkpoint errors below epsilon and rechecked every gap
above `2 * epsilon + margin` = 0.0003. No input replacement was needed. Numerical comparison
always runs on other hosts; tuple mismatch never becomes structural-only. Future multi-label
fixtures must satisfy `abs(logit) > epsilon + margin` at threshold 0. Reference regeneration
resets acceptance and requires new measurements.

## Precision and memory

Strict encoder/core/T5 references retain 1e-4 in full-float32 JVMs with TF32 off. Separate
ordinary tests cover BERT task heads and T5 full/cached outputs with bound 0.03: measured BERT
0.00014835596084594727, T5 ReLU 0.006804823875427246 and T5 gated 0.020420074462890625.
These tiny-fixture bounds are not universal checkpoint claims.

Repeated completion, bounded caches, zero targets, cancellation, listener failure, injected
MLXException at the projection boundary and output-tokenizer failure returned to warm native
baseline 41984 bytes, with zero repeated-request retained growth. Injection verifies cleanup
and propagation, not a real device-failure diagnostic. A models observer counts actual cache
initialization once per layer/request and distinct identities for a different second source.

The tiny fixture uses 192 bytes per source/target K/V position and 64 bytes per retained encoder
position. Peak deltas were 302528 bytes for source32/target1, 87296 for source4/target32 and 302464
for source32/target32. Tested peak budget: 32 × derived retained bytes  + 64 KiB fixed warm margin.
This includes temporary attention/FFN overhead for those tiny widths, not a general linear peak
law; encoder attention is quadratic. Retention separately has 64 KiB margin and <=4 KiB late
growth. Static source dimensions equal encoder output length before decode. Bounded target
capacity rejects before source encoding. T5 sliding is rejected and makes no plateau claim.

## Validation and remaining acceptance

Passed locally: full `./gradlew build`; independent `./gradlew -p buildSrc check`; tokenizer
Java/Java21 bytecode checks; strict/default references and current decoder/scheduler regressions;
all three opt-in artifact tasks; inventory/call-site/hash verification; pure-Java contracts under
an unstaged native path; tokenizer oracle and
fixture verification. Optional MLX oracle verified 7.1 and explicitly skipped Phase 6 because that
profile requires exactly macOS 26. New CI assertions use matching result directories. Committed
external tokenizer/HF fixtures are
explicit Gradle test inputs; opt-in Tier-B always executes instead of reusing another host result.

Supported macOS 26 native and weekly Tier-B acceptance for these changes remains pending.
Ordinary builds execute no Python oracle/reference/download commands and retain existing
Java/Gradle and optional native-bootstrap behavior. Unsupported: training, quantization, beams,
decoder prefixes, arbitrary encoder families/custom pooling/non-root Transformer modules and
seq2seq scheduler cohorts. Model README records deliberate OMIT source-special behavior,
rejection instead of automatic truncation, caller EOS policies and separate target penalties/cache
accounting. No artifact is published by implementation.
