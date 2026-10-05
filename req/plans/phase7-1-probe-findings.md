# Phase 7.1 pinned-runtime findings

Recorded 2026-10-05 on Apple M5 Max (CPU and GPU), Darwin 27.0.0, CPython 3.12.15,
Gradle Java 25.
Native pins: MLX/MLX-metal 0.31.2, mlx-c fba4470b89073180056c9ea46c443051375f7399.
Python probes: `MLX_ENABLE_TF32=0 tools/mlx-oracle/.venv/bin/python
 tools/mlx-oracle/probes/phase71.py` (run outside a sandbox that hides Metal).
References use CPU, Java uses its existing GPU streams. Both precision modes have explicit tasks.

## Confirmed contracts and amendments

- Padding supports constant and edge. Reflect/symmetric mode strings throw
  `ValueError: Invalid padding mode (...) passed to pad`. Equal widths are supported independently.
- A contiguous-normalized row-contiguous slice `[3:7]` of `arange(10)` strided at offset zero
  returns
  `[3,4,5,6]`. Offset 3 with stride -1 returns `[6,5,4,3]`; zero stride at offset 1 returns four 4s.
  The Java GPU equivalents run in the required ordinary `StridedViewSafetyTest`, including
transpose,
  first/last bounds, overflow, empty and scalar views. No out-of-bounds read is probed.
- Tile accepts zero repetitions and leading rank extension; repeat accepts zero and negative axes.
  Inverted clip bounds return the upper bound. Java requires two array bounds; use min/max for
  one-sided clipping. No empty-array bound convention is promised.
- All six convolution ranks evaluate on CPU. Weights are `[out, kernel..., in/groups]` for both
  ordinary and transpose variants. Differential cases distinguish channels and spatial dimensions.
- Three-dimensional grouped convolution is rejected by native:
  `[conv] Can only handle groups != 1 in 1D or 2D convolutions.` This also affects transpose3d.
  The facade retains groups and native errors; it does not claim support for this combination.
- **GPU defect:** grouped transpose2d with stride 2 differs from CPU. With input `arange(32)` shaped
  `[1,4,4,2]`, weight ones `[4,2,2,1]`, groups 2, padding/output-padding zero, CPU sum is 3968 and
  GPU sum is 960. Padding 1/output-padding 1 gives CPU 3458 and GPU 8172. Stride 1 matches in the
  four tested padding/output-padding combinations. Java explicitly rejects grouped transpose2d
  with non-unit stride. This is an amended unsupported-runtime combination, not tolerance drift.
- MaxPool1d(size=2, window=3, stride=2) computes zero extent and native max rejects the empty array:
  `[max] Cannot max reduce zero size array.` Preserve that error; floor division remains required.
  Python negative extent (size=1, window=4, stride=1) attempts an enormous allocation and errors.
  Java rejects negative extent before striding, naming size/window/stride.
- Pinned sinusoidal dims=2 returns NaNs (missing denominator guard). Java deliberately rejects
  dims<4. Odd dims returns twice floor(dims/2), as pinned. Default scale is sqrt(2/dims).
- Upsample linear align_corners=True with output size one raises Python ZeroDivisionError.
  Java rejects it with a named IllegalArgumentException before gathering.
- GroupNorm defaults to interleaved channels. PyTorch grouping must be explicitly selected.
  BatchNorm supports ranks 2-4; eval without tracking uses freshly computed batch statistics.
  Its floating running statistics remain differentiated by ModuleGrad in Java.
- **Loss amendment:** KL inputs and targets are both log probabilities. The pinned source computes
  sum(exp(target)*(target-input), axis). There is no selectable log-target option. Zero target
  probability can produce NaN from the literal formula; sentinels preserve that classification.
- CE smoothing is [0,1), weights have the exact reduced output shape. BCE inputs/targets and weights
  have equal shapes; probability logarithms are clipped at -100, avoiding endpoint NaNs.
  MSE/L1/smooth-L1 require equal shapes. Cosine defaults axis=1 and clips the norm product by
epsilon.
- ALiBi uses symmetric negative absolute distance and adds to scores; offset shifts queries.
  QuickGELU uses the explicit HF formula x*sigmoid(1.702*x), with a formula reference in the
dispatcher.

## Evidence and remaining acceptance

`Phase71OracleTest` compares committed CPU JSON shape/value references against Java GPU execution
with combined atol=1e-4, rtol=1e-5; nonfinite classifications are exact. Development-host strict
comparison passed after the grouped transpose2d limitation was isolated. Phase 6 fixture outputs
show drift at seven decimals on this host: array softmax 0.665241 -> 0.6652409 and sampling
log-probability -0.4076059 -> -0.4076061. Generation initially exposed this difference; the legacy
files were restored byte-for-byte and are intentionally not updated. The global
verifyMlxOracleFixtures gate therefore remains failing on this host. This is a separately reviewed
Phase 6 reproducibility issue, not permission to rewrite legacy goldens. The drift also occurs
with TF32=1, so selecting default precision does not resolve it.

The environment verifier and runner enforce per-profile OS policies: Phase 6 GPU generation and
verification require exactly macOS 26; Phase 7.1 CPU fixtures allow macOS 26+. Unfiltered commands
on macOS 27 fail before rewriting any fixture. Selecting `-PmlxOracleFamily=phase7-1` verifies only
the compatible CPU profile. Per-family CPU/GPU profiles and precision are validated independently.
Cross-host byte-exact verification remains a CI acceptance gate; no second host result is claimed
here.

The isolated launcher ran Phase71ConvolutionProbe: exit 0, executed 1, skipped 0, failures 0,
executed-success. It asserted the native grouped-3D error in a disposable JVM. XML/log/summary are
under build/phase71-probes/Phase71ConvolutionProbe; missing or skipped reports are infrastructure
errors. No expected process termination is declared for current cases.

## Verification results

- `./gradlew build`: passed, including both precision tasks and existing model/decoder suites.
- `./gradlew :jmlx-core:check :jmlx-models:check`: passed; new native correctness suites executed
  without skips. Existing unrelated Checkstyle warnings remain; new sources have no warnings.
- Inventory generation, call-site guard and staged header coverage: passed; bindings unchanged.
- `verifyMlxOracleFixtures -PmlxOracleFamily=phase7-1`: passed byte-exact regeneration of 125 cases.
- Core publication POM/module metadata guard: passed; no fixture variant/artifact or Jackson/JUnit
  dependency was published. Production dependency remains jmlx-ffi only.
- Fixture input invalidation: same filtered core/models ordinary/float32 command ran once, then all
  four tasks were UP-TO-DATE; appending a temporary newline to the 7.1 input made all four execute,
  each reporting that input changed. The fixture was restored exactly afterward.
- Isolated launcher: executed 1, skipped 0, failures 0, aggregate passed=true.
- Global legacy oracle verification: fails with the recorded Phase 6 drift; originals preserved.
- CPU byte-exact verification on a second CI host: pending, no result claimed.

## Measured reference errors

125 fixed cases, with atol=1e-4 plus rtol=1e-5 in both development-host precision modes.
Maximum absolute errors per operation (float32 values read back from Java):

| Operation | Full float32 | Default precision |
| --- | ---: | ---: |
| abs | 0 | 0 |
| alibi | 5.9604645e-08 | 5.9604645e-08 |
| as_strided | 0 | 0 |
| avg_pool1d | 3.7252903e-08 | 3.7252903e-08 |
| avg_pool2d | 5.9604645e-08 | 5.9604645e-08 |
| batch_norm | 4.7683716e-07 | 4.7683716e-07 |
| binary_cross_entropy | 2.3841858e-07 | 2.3841858e-07 |
| clip | 0 | 0 |
| conv1d | 0 | 0 |
| conv2d | 0 | 0 |
| conv3d | 0 | 0 |
| conv_general | 0 | 0 |
| conv_transpose1d | 0 | 0 |
| conv_transpose2d | 0 | 0 |
| conv_transpose3d | 0 | 0 |
| cosine_similarity_loss | 5.9604645e-08 | 5.9604645e-08 |
| cross_entropy | 2.3841858e-07 | 2.3841858e-07 |
| dropout | 0 | 0 |
| elu | 5.9604645e-08 | 5.9604645e-08 |
| floor | 0 | 0 |
| group_norm | 4.7683716e-07 | 4.7683716e-07 |
| hardswish | 3.3527613e-08 | 3.3527613e-08 |
| instance_norm | 5.9604645e-08 | 5.9604645e-08 |
| kl_div_loss | 2.3841858e-07 | 2.3841858e-07 |
| l1_loss | 4.8428774e-08 | 4.8428774e-08 |
| leaky_relu | 7.450581e-09 | 7.450581e-09 |
| log1p | 9.536743e-07 | 9.536743e-07 |
| log_softmax | 4.4703484e-08 | 4.4703484e-08 |
| max | 0 | 0 |
| max_pool1d | 0 | 0 |
| max_pool2d | 0 | 0 |
| minimum | 0 | 0 |
| mish | 2.9802322e-08 | 2.9802322e-08 |
| mse_loss | 4.656613e-09 | 4.656613e-09 |
| nll_loss | 0 | 0 |
| pad | 0 | 0 |
| pad_symmetric | 0 | 0 |
| quick_gelu | 5.9604645e-08 | 5.9604645e-08 |
| relu | 0 | 0 |
| repeat_axis | 0 | 0 |
| selu | 1.1920929e-07 | 1.1920929e-07 |
| sigmoid | 5.9604645e-08 | 5.9604645e-08 |
| sinusoidal | 4.2021275e-06 | 4.2021275e-06 |
| smooth_l1_loss | 5.9604645e-08 | 5.9604645e-08 |
| softmax | 5.9604645e-08 | 5.9604645e-08 |
| softplus | 5.9604645e-08 | 5.9604645e-08 |
| tanh | 5.9604645e-08 | 5.9604645e-08 |
| tile | 0 | 0 |
| upsample | 2.3841858e-07 | 2.3841858e-07 |
| var | 0 | 0 |

Grouped 2-D `convGeneral` also exposes the input-dilation defect: reviewer CPU/GPU comparisons
with groups=2 found maximum absolute differences of 13.66 for dilation=(2,2), 14.06 for (1,2),
and 7.63 for (2,1) with flip. Java rejects all grouped 2-D non-unit input dilation. Grouped
1-D dilation and ungrouped 2-D dilation matched; the restriction remains specific to 2-D groups.
ALiBi now constructs distance and broadcast slopes on-device and casts bias to the scores' dtype.

ALiBi casts both slopes and distances to the scores' dtype before multiplication, matching pinned
MLX's reduced-precision rounding order. The float16 oracle regression uses 12 heads, query
offset 4032, and shape [1,12,2,16]; Java matches the reference exactly. Running this case against
the previous float32-multiply-then-cast implementation fails the exact comparison.

Exact oracle comparison is now explicitly selected with `exact: true` per case. The bfloat16 twin
of the 12-head, offset-4032 ALiBi fixture also matches exactly. Host-policy checks covered macOS
26/27 acceptance and macOS 25 rejection; both unfiltered Gradle generation and verification refused
macOS 27, with all expected fixture hashes unchanged.
