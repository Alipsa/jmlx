# Phase 7.3b — Qwen3-VL image and text inference

- **Status:** proposed, 2026-10-08. Source inspection completed for planning; artifact pins,
  executable reference probes and implementation acceptance remain work below.
- **Parent:** `req/plans/phase7-plan.md` §7.3b, public decisions 1–4, 6–8 and
  Cross-cutting verification.
- **Dependencies:** accepted 7.0 Qwen3 backbone and completed 7.3a multimodal contracts,
  expansion/accounting, embedding entry point and lifecycle tests. Reuse 7.1 core layers.

## 1. Outcome and boundaries

Deliver local dense Qwen3-VL image+text generation through `TextGenerationModels.load` and
`TextGenerationModel.generate`, using the existing sampling, streaming, cancellation, result and
cache-policy contracts. Support one or multiple decoded images in prompt order, including text
between images and text-only requests through the same model. Implement dynamic-resolution
preprocessing, the vision tower and mergers, interleaved three-axis RoPE and DeepStack.

The Tier-B candidate is [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct),
whose current model card identifies Apache-2.0 licensing. WP1 must verify it is the smallest
public licensed dense instruct artifact suitable for this gate and pin it immutably. Current
`main` discovery is not an artifact pin, loader acceptance or inference evidence.

Float safetensors are the correctness baseline. MLX affine support is also required for the
parent's manual 8B 4-bit gate, with its precise supported tensor scope fixed in WP1/WP6.
Exclude video, `qwen3_vl_moe`, remote image fetching, training, image generation, public model
batching, multimodal scheduler cohorts, GPTQ/AWQ/GGUF, quantized KV and per-layer quantization
overrides. Reject unsupported inputs/configurations explicitly before model return or inference.
No native-pin change or generated-binding edit is planned.

## 2. Repository baseline and prerequisite gate

At planning time, `jmlx-vision` contains `RgbImage`, image decode/resample/normalize and the
SmolVLM processor, with image-oracle provenance and Java 21 enforcement. Qwen3 text dispatch and
per-head Q/K normalization already exist. 7.3a WP3 delivered immutable
`GenerationRequest.withImages`/`images`, `InputModality`, TEXT/IMAGE metadata and structured
chat validation via PR #42. Reuse those contracts. `GenerationResult.promptPositions` also
exists in the current uncommitted working tree; reverify its merged state at implementation
start rather than treating it as a new 7.3b API. Remaining integration gaps are:

- `TextGenerationModels` dispatches T5 and ordinary decoders, with no SmolVLM model.
- `DecoderModel` embeds token IDs internally and has the existing scheduler `EmbeddingHook`;
  no production embedding-start entry point or post-layer visual injection exists.
- `RopeSpec` and decoder attention support scalar/per-row positions, not three-axis coordinates.

Thus delivered 7.3a preprocessing does not establish completed 7.3a model execution. Finish
7.3a WP4–WP8 before the integration work here: acceptance of `promptPositions`, pure expansion
and resource limits,
decoder embedding entry point, scheduler rejection, shared generation mechanics and exact
before/after recorder. Do not create competing versions of these contracts in 7.3b.

Reference discovery and pure Qwen preprocessing may proceed while these prerequisites finish.
At implementation start, record the actual base commit, completed prerequisite suites and any
publication since this plan. Public changes after publication must remain additive.

## 3. Reference and artifact gate (WP1)

Use the existing hash-locked Transformers 4.57.6 CPU float32/eval reference toolchain. That tag
contains [Qwen3-VL modeling](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)
and its [modular source](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3_vl/modular_qwen3_vl.py).
Inspect configuration, [processor](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3_vl/processing_qwen3_vl.py),
the resolved image processor and generation preparation as one path. The slow
[Qwen2-VL image processor](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py)
is the explicitly selected jmlx preprocessing reference. The candidate artifact declares
`Qwen2VLImageProcessorFast`; record and reverify that declaration at the pinned revision.
**2026-10-08 decision:** use the slow PIL processor as a documented deviation from that
declaration. Instantiate `Qwen2VLImageProcessor` directly with the pinned artifact's options
in both reference tools, and supply it explicitly to the multimodal processor; do not rely
on `AutoProcessor` selection or a missing-torchvision fallback. Existing HF/image locks
remain the slow-reference environments. Tier-A and Tier-B exact IDs are relative to this
slow preprocessing, not a claim of equality with the fast processor.

Probe construction of `Qwen3VLProcessor` with the explicit slow image processor inside the
locked, torchvision-free HF venv. Its `video_processor` attribute must not introduce an
implicit torchvision requirement. Record whether `video_processor=None` is accepted and
the minimal supported construction if a video processor is required. If construction cannot
work in that environment, call the pinned tokenizer/template and pure image expansion path
without the wrapper, verifying the same image/text IDs and geometry. Archive the selected
construction and executable image/text probes; video remains rejected. Do not silently add
torchvision to the slow-reference locks to make wrapper construction succeed.

WP1 must measure slow-versus-fast uint8 pixel, normalized-patch, prefill/cached-logit and
greedy-ID divergence once on pinned representative inputs. Use a separate comparison venv
under `tools/image-oracle/fast-comparison/`, with reviewed hash-locked Torch/torchvision and
the same Transformers/Pillow/NumPy pins. Record versions, explicit processor classes/options,
commands and source/output hashes. The fast-comparison venv produces processed patches only;
it never computes model logits. Export both slow and fast patch sets losslessly with dtype,
shape, grid and SHA-256, then feed both through the same HF-reference model runtime, using
the same checkpoint, Torch version, FLOAT32/eager settings and explicit positions. Compute
prefill/cached logits and greedy IDs there, so the reported divergence isolates preprocessing
and does not mix Torch runtimes. Record that runtime's lock digest and exact Torch version
(currently 2.6.0) in the comparison provenance. This environment is opt-in, never an ordinary-build
dependency. Archive the differences and state the deviation in compatibility/docs and Tier-B
provenance; fallback warnings or source inspection alone do not satisfy this gate.
When adding its lock, extend root `verifyImageOracleDependencyPins` to compare the
fast-comparison lock's Pillow/NumPy/Transformers versions and environment markers against
both existing locks, and declare it as a task input. Missing pins or mismatches must fail
root `check` without Python. Verify the guard with deliberate version/marker drift in a
disposable lock copy. This guard extension is a WP1 prerequisite to comparison acceptance.

The current [2B config](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/main/config.json)
provides discovery values, to be reverified at the chosen revision:

| Area | Candidate values |
| --- | --- |
| Outer/text types | `qwen3_vl` / `qwen3_vl_text` |
| Text | hidden 2048, 28 layers, 16 query / 8 KV heads, head dimension 128, tied embeddings |
| Text RoPE | theta 5000000, default type, interleaved MRoPE sections `[24,20,20]` |
| Vision | hidden 1024, depth 24, 16 heads, intermediate 4096, output hidden 2048 |
| Patch geometry | spatial patch 16, temporal patch 2, spatial merge 2 |
| Vision positions | 2304 learned positions; DeepStack extraction indices `[5,11,17]` |
| Special tokens | image/video and vision-start/end IDs declared in outer config |

Create `req/plans/phase7-3b-reference-findings.md` with executable tiny probes and:

1. Full HF commit, license/access evidence, file SHA-256s, shard/tensor inventory, storage dtype,
   download size and tokenizer/template/processor/generation files. Populate a supported,
   explicitly inert or rejected key table for each config, including nested RoPE keys.
2. Resolved processor class/options, raw PIL decode/color path, resize geometry and rounding,
   normalization dtype/order, temporal duplication and patch permutation. Probe RGB, grayscale,
   alpha and palette inputs against the existing decoder contract; document any mismatch and
   resolve it before claiming parity. Do not silently reuse SmolVLM-specific color behavior.
   The existing `ImageDecoder` composites RGBA/LA over white; the candidate Qwen2-VL slow
   processor uses PIL `convert("RGB")`, dropping alpha instead. `RgbImage` has already lost
   alpha and cannot undo that compositing. If confirmed for the pinned artifact, add an
   explicit decoder option that preserves source color channels while dropping alpha,
   keeping existing decode overloads' white-compositing behavior. Freeze its additive API
   and palette/tRNS semantics in WP1 before WP2. Qwen file-to-processor parity fixtures and
   usage examples must select this option. Arbitrary caller-supplied `RgbImage` values are
   accepted as final RGB input; document that their original file/color provenance cannot
   be recovered or validated by the model.
3. Vision attention segmentation, normalization/activation/biases, learned-position interpolation,
   two-axis rotary layout, merger permutations and intermediate tensor shapes.
4. Unexpanded/expanded prompt IDs, feature destinations, all three position axes, RoPE delta,
   logical cache positions and DeepStack timing. Include hand-checkable asymmetric grids.
5. Actual quantized 8B candidate: immutable revision, license, quantization metadata, every
   packed/float tensor category and tied aliases. Reject incompatible formats rather than
   approximating their meaning.
   Use `mlx-vlm` as the independent MLX-affine reference, in a new opt-in
   `tools/qwen-vl-reference/` CPython 3.12 environment. Before probes, commit a reviewed
   hash-locked `requirements.lock` for an exact compatible `mlx-vlm` release and its MLX,
   Transformers and processor dependencies, with installer, runner and provenance. Select
   actual versions by executable compatibility probes; unpinned installs are not evidence.
   Record converter identity/version, original HF revision, conversion options and tensor
   layout; missing provenance blocks that artifact's acceptance. Run with the same explicit
   slow preprocessing/processed patches as jmlx and record compute/cache dtype and kernels.
   Prefer `mlx==0.31.2` and `mlx-metal==0.31.2` to match the current native and MLX-oracle
   pins; reverify the actual pins at implementation start. Probe whether a Qwen3-VL-capable
   `mlx-vlm` release supports that pair. Record both wheel versions/hashes, native jmlx
   pins and `mlx-vlm` version in every quantized-reference reproducibility tuple. If no
   compatible release exists, review/pin the required runtime separately and record the
   mismatch and possible affine/dequantization numerical differences in the 8B evidence;
   do not claim same-runtime parity or change jmlx's native pin implicitly.

Use eager attention for reproducible reference math and record it explicitly. Verify 4.57.6
works by running tiny prefill and cached decode; a toolchain bump requires a separate reviewed
lock/provenance/golden change. Preserve image/HF shared dependency-pin guards. Source reading
alone does not close WP1; archive probe commands/results and populate the mapping tables first.

## 4. Dynamic image processing (WP2)

Add `Qwen3VlProcessorConfig`, `Qwen3VlImageProcessor` and an immutable processor result in
`se.alipsa.jmlx.vision`. Keep the module pure Java 21 with no native gates. Reuse Pillow kernels,
RGB primitives and normalization only when their operation order agrees with the reference.
Implement the WP1 decoder option with RGBA/LA and palette/tRNS oracle fixtures, including
transparent pixels with non-white source colors. Assert both the Qwen-selected conversion
and unchanged default SmolVLM conversion before claiming file-to-processor parity.
The result owns flattened patch data, patch shape, `grid_thw` and merged-token counts; buffer
accessors are defensive, with a bulk-copy path for models. Do not describe flattened patches
as ordinary NCHW/NHWC image tensors.

Provide a pure geometry-planning method shared by processing and zero-token generation.
Validate positive dimensions, pixel budgets, channels, patch/temporal/merge sizes, finite
normalization parameters and supported resize options. Use checked long arithmetic before
array sizing or conversion to int. Bound total input images, decoded/resized pixels, patch rows,
merged rows and expanded positions; keep processor pixel budgets distinct from application
resource limits. Report which budget and geometry failed.

**2026-10-08 processor-budget deviation:** default effective `size.longest_edge` is
**1,048,576 pixels** (`4096 * 16 * 16` for the candidate's patch size), with the pinned
`size.shortest_edge` minimum retained. This is an explicit override of the artifact's
16,777,216-pixel maximum, in addition to the slow-PIL deviation. Give the same effective
size to Java geometry/processing and every slow/fast/HF/quantized reference. Record original
and effective size, patch geometry and application budget in Tier-A/Tier-B provenance.
Derive the area cap from the measured raw-row budget and actual patch size with checked
arithmetic. Run smart-resize to downscale ordinary photos rather than rejecting their
artifact-default resized geometry. Add a 1920-by-1080 input fixture producing
1344-by-768 pixels (4032 raw rows), a 1280-by-960 input producing 1152-by-864
(3888 rows), a portrait 1080-by-1920 input producing 768-by-1344 (4032 rows), and
1024-by-1024 at the exact 4096-row cap. Record these geometries in oracle fixtures.

The default hard budget remains **4096 raw patch rows per image** after effective resize.
Use noncausal `MLXFast.scaledDotProductAttention`
(`mlx_fast_scaled_dot_product_attention`) per image. WP1 must probe its fused/fallback
behavior and WP5 measure the evaluated per-image peak; do not assume fusion from the name.
At 4096 rows and 16 heads, one FLOAT32 score matrix is 1 GiB; conservatively budget four
such matrices (4 GiB) plus Q/K/V, block/merger activations and recorded allocator workspace.
This is an attention-workspace planning bound, not a claim that the model fits a 7 GB host.
Freeze the measured peak and supported fallback in findings before accepting WP5; if it
exceeds the budget, lower the limit or introduce separately probed chunking before acceptance.
Sequential image execution bounds concurrent attention workspace to one image; also bound
aggregate feature/patch storage. Caller processor overrides must not bypass the measured
hard budget. Enforce two separate checks before pixel/native work, including the
zero-new-token path:

- Configuration check: effective maximum area must be at most
  `rawRowBudget * patchSize * patchSize`. A `size.longest_edge` override of 1,049,600
  pixels must fail against the default 1,048,576-pixel budget, before image planning.
- Per-image check: the planned raw patch rows must be at most `rawRowBudget`. With resize
  disabled and an otherwise valid configuration, a 160-by-6560-pixel image yields a
  10-by-410 raw-patch grid (4100 rows, merge size 2, aspect ratio 41) and must fail this
  check. This fixture must reach image planning rather than fail the configuration check.
  Set its decoded/resized and aggregate application pixel limits above 1,049,600 pixels,
  retaining the valid 1,048,576 processor area setting and the 4096-row hard budget.
  With resize disabled, the processor area setting governs resize configuration, not a
  separate pixel rejection before the raw-row check. Assert that the rejection explicitly
  names the raw-row budget, with actual 4100 and limit 4096; a generic rejection or a
  decoded/resized-pixel budget error does not satisfy this test.

Raising the hard budget requires separate measured memory evidence and an explicit
application override. Never inherit the artifact's
65,536-row allowance as an application default. If a lower measured budget is needed,
lower the effective area cap too and regenerate/review matching reference provenance.

Port smart-resize rounding, min/max-area adjustment and aspect-ratio validation exactly,
including Python ties-to-even. Preserve Python's binary64 operation order with `Math.rint`
and `Math.sqrt`: in particular `floor(h / beta / factor)` must retain its two sequential
divisions, not become `floor(h / (beta * factor))`. Keep the same order in area scaling,
ceil/floor and aspect checks; no FLOAT32 intermediates or algebraic reassociation. Add
half-factor rounding ties and a hash-pinned boundary fixture sensitive to division order,
recording reference binary64 intermediates and final geometry. The geometry factor is derived
from patch and merge sizes;
never copy Qwen2's default factor into Qwen3. Probe very small dimensions and non-divisible
inputs. Establish precedence of legacy min/max-pixel fields versus size fields from the
resolved processor, and reject contradictory unsupported settings.

For still images, reproduce the reference temporal duplication and channel/temporal/spatial
flattening, grouping patches in merge-block order. Assert `grid_t == 1` for supported still
images, raw rows `t*h*w`, and merged rows `t*h*w/(merge*merge)`, with divisibility checks.
Use index-coded channels/pixels to prove row and inner-patch order, including rectangular
grids and two differently sized images. Never pad unrelated images into one attention sequence.

Extend `tools/image-oracle` with a separate Qwen family path; do not change SmolVLM fixtures.
Pin processor config, images and case specs in provenance. Cover rounding ties, min/max bounds,
portrait/landscape, extreme legal/rejected ratios, resize-disabled validation, patch order,
normalization and mixed-image ordering. Compare geometry/ordering/uint8 resize exactly and
normalized float bits where the reference order permits; record any justified float tolerance
independently of model-logit tolerances. Add a shared hash-pinned processed-patch fixture consumed
by both image and HF tools, so model-math tests cannot conceal preprocessing errors.

## 5. Prompt expansion and position planning (WP3)

Add a package-private pure `Qwen3VlPromptExpander` beside the 7.3a expander. Inputs are original
IDs, validated special IDs and image geometry. Outputs own expanded IDs, ordered feature ranges,
three-axis prefill positions and post-prefill delta/state. Resolve special IDs through the
tokenizer and cross-check outer config/vocabulary; never hard-code candidate numeric IDs.

Accept one unexpanded image placeholder in each valid vision-start/end wrapper and repeat it
to the merged feature count. Match images to wrappers in prompt order. Validate the wrapper
grammar, unmatched/nested markers, image-count mismatches, video markers and pre-expanded input;
do not guess whether a repeated marker run represents one image or several. Define the exact
accepted pretokenized grammar in WP1 and document it. Template goldens cover structured messages,
system/multi-turn chat, text-only, adjacent images, text between images and applicable template
options, through the real processor-owned template if present.

Keep original IDs in result/abort payloads. Charge expanded IDs to sampling history, context
limits and cache capacity; report expanded length through 7.3a's `promptPositions`. For positive
generation budgets, cache demand is `expandedLength + maxNewTokens - 1`. For zero new tokens,
validate markers, geometry and context, report positions, and skip pixel processing/native/KV
work. Retain 7.3a's capacity-zero rule in the zero-new-token path.

Position planning must preserve three independent concepts: physical expanded sequence order
for causal masking, monotonic logical positions for cache accounting, and multimodal coordinates
for RoPE. Image coordinates need not increase like token positions. Build coordinate axes and
delta by matching the pinned `get_rope_index` probes; continue decode coordinates from logical
cache position plus that request's delta. Do not derive them from retained KV length after
eviction. Test negative deltas, rectangular grids, text after an image, two different grids and
text-only zero delta. Reject coordinate/arithmetic overflow before creating GPU arrays.

## 6. Core RoPE and decoder integration (WP4)

Capture all existing Tier-A decoder prefill/decode raw float bits and exact greedy IDs before
changing the decoder or RoPE path, in both TF32 modes. Reuse the 7.3a recorder, with commit,
native pins, device/OS, batch settings, fixture/request and precision tuple. Extend its inputs
only if necessary; recordings remain device-specific PR evidence.

Add an explicit multi-axis RoPE path in `jmlx-core`, with validated `[3,B,T]` positions and a
family-specific section/interleaving specification. Preserve existing overloads and route absent
multimodal inputs through the identical old graph. Do not globally reinterpret `RopeSpec.Base`
or coerce `qwen3_vl_text` to `qwen3` and lose its rotary layout. Choose an additive overload or
dedicated core rotary component after WP1; freeze its signatures in the findings before coding.
`RopeSpec` gains no new permitted variant: its public sealed hierarchy must preserve source
compatibility for exhaustive consumer switches.

Kernel decision: Qwen3-VL uses a dedicated explicit FLOAT32 frequency/cos/sin and half-rotation
path for multimodal prefill, cached decode, text-only prefill and fresh full forward alike.
Existing `mlx_fast_rope`/`mlx_fast_rope_dynamic` remain on ordinary decoder paths; their
offset-plus-arange contract cannot express unequal multimodal coordinates. Freeze this
per-path kernel matrix and compute/cast order in WP1 findings. Equal axes reduce
mathematically to 1-D RoPE, but do not assume bit equality with the fast kernels. Test
decode axes at `cachePosition + delta`, including negative deltas with valid shifted
positions. Any later fast-kernel optimization requires a dated amendment, shifted-offset
probes, strict/default cached/full equivalence and performance evidence; never silently
mix kernels between the compared paths.

Implement the pinned interleaved frequency selection, half rotation and compute/cast order,
with asymmetric-axis/index-coded tests. Validate section bounds, supported rotary dimension
and head dimension. Cover identical axes, unequal axes, nonzero decode positions and invalid
shapes. Causal masks and cache preflight/postflight remain based on sequence/cache positions.

Extend the internal decoder embedding-start path with request-supplied rotary state and a
post-block injection callback. The source places DeepStack additions after each selected early
decoder block, before final normalization; distinguish vision extraction indices from decoder
injection indices. Keep the scheduler `EmbeddingHook` at its original post-lookup/pre-scale
point. Production ordinary decoders use no-op hooks. Preserve evaluation ordering, valid-length
checks, cache poisoning and tied projection. No public forward-from-embeddings API is required.

Implement visual replacement and masked residual addition through existing core facades,
using slice/concatenate or supported indexing; model code never calls FFI. If a new primitive
is demonstrably required, first record its exact mlx-c declaration, probe native semantics,
update inventory overrides/call-site coverage and test scope/error behavior. Do not preemptively
add general indexing APIs. After the refactor, require exact old-path bit and greedy-ID equality
under the same reproducibility tuple, not just tolerance-golden passes.

## 7. Vision tower and checkpoint assembly (WP5–WP6)

Add `Qwen3VlModel implements TextGenerationModel`, composing the decoder so the scheduler's
existing non-DecoderModel check rejects it. Dispatch `qwen3_vl` before generic decoder config
parsing; explicitly reject `qwen3_vl_moe`. Metadata advertises TEXT and IMAGE, without exposing
MLXArray results. Reuse shared generation mechanics and the 7.3a state/accounting contracts.

Build family-specific config/descriptor/tensor plans for vision and nested text. Verify required,
optional and forbidden names, shapes, biases, aliases and all unexpected tensors before model
return. Do not infer prefixes from the ordinary Qwen3 checkpoint. Validate output/text hidden
dimensions, DeepStack count/index ranges and learned-position table geometry.
Maintain separate tensor plans selected by explicit checkpoint provenance: HF float
safetensors and MLX/`mlx-vlm` converted affine checkpoints. Each plan owns prefixes,
aliases, packed categories and conv layout; probe actual converted names/layout rather
than assuming they match HF. Reject unknown provenance/layout combinations before loading.

The inspected vision source includes temporal Conv3d patch embedding, interpolated learned
positions in addition to two-axis RoPE, segmented bidirectional attention, GELU-tanh blocks and
an exact-GELU merger. Main and DeepStack mergers differ in normalization order. These are
explicit reference-probe targets, not interchangeable SmolVLM components.

Use reusable `nn` modules in core for patch projection, rotary vision attention and mergers;
family sequencing/weight mapping stays in models. For patch projection, compare `Conv3d` with
an equivalent flattened linear formulation on identical ordered patches. Select one documented
implementation after strict intermediate probes, preserving bias/dtype and weight permutation;
do not replace temporal kernels with Conv2d. Choose conv layout by provenance, verify by
shape, never heuristics; an already channels-last MLX conversion must not be transposed
as though it were PyTorch. Test both provenance plans with equivalent weights. Additive
core layers must follow Module ownership and train/eval rules.

Process images independently or segment packed attention so one image cannot attend to another.
Probe learned-position interpolation and merger order on unequal rectangular grids. Capture
patch embedding, selected block outputs, final merged features and every DeepStack level in
the tiny HF reference to localize errors before end-to-end logits.

Float correctness initially computes in FLOAT32, including conversion of stored BF16/FP16
weights; document memory cost. Add MLX affine loading only for verified eligible tensors, with
required `.scales`/`.biases`, dimensions and group-size checks; norm/position state remains float.
If the 8B artifact packs vision matrices, implement and test those categories too or select a
compatible artifact and record why. Synthetic quantized tests compare against the same
dequantized weights. Never compare quantized weights against unrelated HF float goldens or
silently dequantize the entire 8B model to satisfy the manual run.

## 8. Request lifecycle and cache policies (WP7)

Adopt parent decision 8's 2026-10-08 lifetime amendment, also adopted by 7.3a.
Model fields hold weights/config only. Every request owns KV caches and its post-prefill
position state in the generation scope. Final visual features and all DeepStack levels live
in a request-owned prefill child scope, with explicit hoisting from shorter vision activation
children when needed. Evaluate prefill logits and every retained KV array before closing
the prefill scope, so lazy graphs cannot retain unevaluated visual work. Close that scope
and release visual features, processed patches and injection callbacks before cached decode;
only KV and position state survive. Full-forward reference tests independently recompute
visual features. Decode activation scopes read only surviving state; no request state
accumulates on model scope. Unlike the Python model's mutable `rope_deltas` field, Java
position state belongs to the request.

Vision and visual injection run once during prefill, never for ordinary cached decode.
Generated image-special tokens are ordinary generated history and do not consume image payloads
or trigger the tower. Text-only requests skip all vision work. Test two sequential requests
with different image sizes/deltas to expose stale state, plus image → text-only → image reuse.
Check cancellation before preprocessing, between images, before prefill and between decode steps.
Close scopes for EOS, stop, max tokens, listener/tokenizer/native errors and prefill failure.

FULL and capacity-bounded FULL are required. Dense Qwen3-VL does not establish sliding support
merely because the core cache can evict: default to rejecting a sliding request with a named
error. Any extension must have a dated policy amendment and position/retention equivalence
evidence. Capacity errors include original/expanded lengths and occur before native work.

Add `Qwen3VlMemoryTest` using evaluated arrays and `MLXMemory.activeBytes()` after warm-up.
Exercise repeated same/different images, mixed sizes, text-only, cancellation and injected
post-prefill/listener/tokenizer/native failures. For float batch one, self-KV bytes per position
are `2 * decoderLayers * kvHeads * headDim * 4`. During prefill, final/DeepStack features
cost `totalMergedRows * textHidden * (1 + deepstackLevels) * 4`, in addition to the bounded
vision workspace above; after prefill their retained cost is zero. Bound late/early decode
slopes by KV bytes plus recorded metadata/allocator margins. Assert evaluated KV/position
validity after prefill-scope closure, a post-prefill release of visual storage within a
measured margin, and return to baseline on failure/request closure. Full(capacity) rejects
one-over-budget rather than evicting.

## 9. Acceptance evidence (WP8)

Generate a deterministic tiny checkpoint in `tools/hf-reference`, with at least two DeepStack
levels, unequal H/W grids, nontrivial learned positions and RoPE sections, and enough decoder
layers to exercise every injection. Store checkpoint/input/processed-patch hashes, seeds,
resolved configs, attention implementation and provenance. Ordinary Gradle checks verify
committed digests and run Java; Python execution remains explicit.

For each case below, record expanded IDs, grids, all position axes/delta, intermediate visual
features, prefill logits, several cached greedy-decode logits and exact generated IDs:

- one image;
- two different-size images;
- image, intervening text, second image and trailing text;
- text-only through the same Qwen3-VL model.

For every decode step compare incremental logits to a fresh full uncached forward over the
same original images and expanded prompt plus generated IDs. Reapply visual features/DeepStack
only at their original destinations for that full reference. Cover one/many/zero new tokens,
EOS/stop, sampling penalties/filtering, streaming/log probabilities and abort result IDs.

Add deterministic forced continuations containing each image/video/vision-start/vision-end
special ID, plus a complete wrapper-like generated sequence, for image-bearing and text-only
prompts. Treat these IDs as ordinary generated history even when their sequence resembles an
input wrapper. Assert no new image consumption, expansion, tower calls or DeepStack injection.
Compare cached and uncached logits at every forced step in both precision modes. The HF full
reference must pass explicit three-axis positions (original prefill axes followed by logical
positions plus the original delta) and explicit visual masks/destinations restricted to the
original prompt through its embedding/DeepStack path. Do not rerun token-based placeholder
detection or `get_rope_index` over generated markers. Archive this reference invocation and
the forced IDs/positions/masks in provenance so it tests the declared generation contract.

Separate `Qwen3VlGoldenTest` (`full-float32`, strict maximum absolute error `1e-4`) from
`Qwen3VlDefaultPrecisionTest` (ordinary TF32 mode, measured named family bound). Cached/full
agreement has both gates. Record default-mode errors in `req/phase6-golden-precision.md`;
never weaken the strict gate or regenerate goldens to fit reduced precision. Add core rotary
and vision reference suites to the appropriate precision tasks and CI required-execution list.

Pin a Tier-B manifest with processor/tokenizer/template/image hashes, original and expanded
prompt IDs, image limits, cache/compute dtype, exact greedy IDs and reference provenance.
Include at least one multi-image prompt and an exact decoded-output assertion under the pinned
tokenizer. Measure top-logit margins so exact IDs have credible stability. Download through the
existing Java manifest/SHA path. Measure native peak memory, wall time and disk use before
adding a weekly job; the recorded 7 GB RAM / 14 GB disk runner is not assumed to fit a 2B model
converted to FLOAT32. A resource-blocked CI run is recorded as pending, not verified.

Pin and run a compatible public 8B 4-bit artifact manually on suitable hardware. Record
image+text output, an independent comparison using the locked `tools/qwen-vl-reference/`
`mlx-vlm` runner from WP1, artifact/converter provenance and tensor-plan identity, quantized
coverage, prefill
and decode throughput, image sizes/counts, expanded length, peak native memory and the complete
reproducibility tuple in `req/phase6-4-benchmark.md`. Keep this evidence-only unless runner fit
is measured. Neither this run nor Tier-B is optional for claiming the parent milestone done.

Update compatibility, Tier-A/Tier-B documentation, models/vision READMEs, CHANGELOGs and AGENTS
architecture notes with the exact implemented scope. Distinguish synthetic, real-artifact,
quantized manual and pending CI evidence. No release or publication is part of this plan.

## 10. Work packages and completion checklist

| Package | Deliverable | Depends on | Exit evidence |
| --- | --- | --- | --- |
| WP0 | Complete/reverify 7.0 and 7.3a integration prerequisites | Existing milestone plans | Accepted contracts, SmolVLM execution/lifecycle suites and base recorder |
| WP1 | Reference probes, immutable artifact/tensor/config mappings | Discovery only | Populated reference-findings document; executable tiny prefill/decode |
| WP2 | Qwen processor and image oracle | WP1 | Exact geometry/permutation/resize; pinned processed-patch fixtures |
| WP3 | Expansion, templates, three-axis position planner | WP0–WP2 | Pure tests and HF template/expanded-ID/position goldens |
| WP4 | Core multi-axis RoPE and decoder injection seam | WP0, WP1, WP3 | Strict rotary tests; exact before/after decoder bits in both modes |
| WP5 | Vision tower, main/DeepStack mergers | WP1, WP2, core primitives | HF intermediate-feature references |
| WP6 | Loader, composed model and affine scope | WP3–WP5 | Complete tensor validation; float/quantized synthetic loading tests |
| WP7 | Generation, ownership and cache accounting | WP6 | Cached/full equivalence, lifecycle/failure/memory tests |
| WP8 | Tier-B, manual 8B, CI and documentation | WP7 | Parent numerical/artifact/performance/matrix gates |

Implement in reviewable PRs in this order; source discovery and pure processor work may overlap
with WP0. Do not mark later gates complete based only on compile success or skipped native tests.

Required validation includes `:jmlx-vision:check`, `:jmlx-tokenizer:check`, relevant Jinja checks,
`:jmlx-core:check`, `:jmlx-models:check`, root `check`, Spotless/checkstyle, inventory freshness,
`verifyMlxApiCallSites`, `verifyHfReferenceGoldens` and image fixture/pin guards. Explicitly run
both Python oracle verifications and new reference generation/probes in their locked venvs.
Run the opt-in fast-comparison and quantized-reference runners in their separately reviewed
locked environments; record their outputs and lock/provenance digests as WP1/WP8 evidence.
Run `./gradlew -p buildSrc check` if downloader/manifest build logic changes. Add every mandatory
native suite to `.github/workflows/ci.yml` with its correct ordinary/float32 result directory;
obtain supported macOS 26 execution evidence. Existing native skip gates remain authoritative.

- [ ] WP0 prerequisites accepted; no competing multimodal contracts introduced.
- [ ] Immutable artifacts, license evidence and all config/tensor mappings recorded.
- [ ] Slow-PIL and effective-area deviations, processor-construction probe and guarded fast-comparison lock accepted.
- [ ] Locked `mlx-vlm` reference and separate HF/MLX provenance tensor plans accepted.
- [ ] Qwen decode option, unchanged default conversion and exact preprocessing/prompt/position/template fixtures pass.
- [ ] Old one-dimensional decoder path remains bit-identical in both precision modes.
- [ ] Qwen kernel matrix, delta-shifted decode tests and unchanged `RopeSpec` variants verified.
- [ ] Vision/merger/DeepStack intermediate references and quantized synthetic tests pass.
- [ ] All four Tier-A cases and forced generated-marker continuations pass strict/default cached/full-forward checks.
- [ ] Resource validation, request reuse, cancellation/failure cleanup and memory bounds pass.
- [ ] Default per-image attention limit/peak, one-over rejection and post-prefill visual release verified.
- [ ] Multi-image Tier-B exact output and manual 8B 4-bit benchmark recorded.
- [ ] Required CI suites executed; matrix/docs state actual modality and verification limits.
