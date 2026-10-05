# jmlx-core

## Core inference layers (Phase 7.1)

Convolution inputs are channels-last NLC/NHWC/NDHWC. Checkpoint weights, including transpose
convolution, are `[out, kernel..., in/groups]`; constructors copy spatial options and read registered
weights fresh on every forward. `MLXConv` exposes strides, padding, dilation, groups and transpose
output padding; `convGeneral` uses low/high padding and input/kernel dilation. Pinned MLX does not
support grouped 3-D convolution, and grouped transpose2d with non-unit stride is rejected because
its GPU results disagree with CPU. Grouped 2-D `convGeneral` with non-unit input dilation
is rejected for the same defect. See [probe findings](../req/plans/phase7-1-probe-findings.md).

`Sequential` applies fixed unary children; `ModuleList` stores arbitrary fixed modules. Both use
ordered decimal parameter paths. Modules default to eval. `train(boolean)` propagates recursively
and newly registered children inherit the mode. Mode is independent of `freeze()`. Dropout is
identity in eval; Dropout and BatchNorm training throw until Phase 11. Cycles/shared-parent trees
retain the existing unsupported behavior.

```java
try (MLXScope model = new MLXScope()) {
  Conv2d conv = new Conv2d(model, checkpointWeight, checkpointBias);
  Sequential encoder = new Sequential(model, conv, new ReLU(model),
      new AvgPool2d(model, new int[] {2, 2}));
  try (MLXScope step = model.newChild()) {
    MLXArray features = encoder.forward(imageInStepScope);
    MLX.eval(features);
  }
}
```

GroupNorm defaults to MLX interleaved grouping. PyTorch checkpoint loaders must explicitly select
`pytorchCompatible=true`. InstanceNorm reduces spatial axes per example/channel. BatchNorm accepts
ranks 2–4 and loaded running statistics, or computes batch statistics in eval when tracking is
false. Optional affine parameters are a weight/bias pair. Running mean/variance are registered
floating parameters: `ModuleGrad` differentiates them even after freezing; non-trainable buffers
are deferred to Phase 11.

Pool sizes use floor division. Negative extents are rejected before creating windows; max pooling
preserves the native error for empty outputs. Average padding counts zero-padded samples in its
window denominator. Upsample supports nearest and linear over spatial axes, clips indices before
gather, and rejects single-element aligned linear outputs (the pinned denominator is undefined). Pad modes are CONSTANT and EDGE;
PadSymmetric means equal widths, not reflection. `asStrided` always normalizes row contiguity and
checks every reachable element address, with checked arithmetic and no unchecked variant.

ReLU, LeakyReLU, ELU, SELU, Tanh, Sigmoid, Softplus, Mish, HardSwish and QuickGELU are parameter-free.
Softplus is stable at large magnitudes; QuickGELU is HF's `x * sigmoid(1.702*x)`.
`logSoftmax` is composed from log-sum-exp and subtraction. SinusoidalPositionalEncoding returns
features rather than adding to embeddings; dims<4 is deliberately rejected to avoid pinned NaNs.
ALiBi builds its bias with device operations, preserves the scores' dtype, and adds MLX's symmetric
negative absolute-distance bias, with a query offset for cached scores.
Position-sized temporaries are allocated in the input scope.

`Losses` provides CE, BCE, NLL, MSE, L1, smooth-L1, KL and cosine similarity for evaluation.
Reductions are NONE/MEAN/SUM. CE/NLL/KL/cosine default NONE; BCE/MSE/L1/smooth-L1 default MEAN.
CE accepts class labels or same-shaped probabilities, smoothing in [0,1), and exact output-shaped
weights. Label validation synchronizes INT32 tensors before gathering; invalid indices never reach
native. BCE accepts logits (default) or probabilities, same-shaped targets/weights, and clips
probability logarithms at -100. MSE/L1/smooth-L1 require equal shapes. KL takes **log probabilities
for both inputs and targets**. Cosine returns similarity, axis 1 by default, clipping the product
of norms by epsilon. These compositions have no Phase 11 gradient guarantee.

Strict CPU oracle references run against Java GPU in `float32GoldenTest`, with separate default-mode
coverage. The reader is an internal Gradle test fixture; both fixture publication variants are
skipped and POM/module metadata dependencies are checked by `verifyPublishedDependencies`.
