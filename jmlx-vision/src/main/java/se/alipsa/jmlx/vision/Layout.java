package se.alipsa.jmlx.vision;

/** Tensor sample layout for {@link ImageTensor}. */
public enum Layout {
  /** Height, width, channels: shape {@code [height, width, 3]}. */
  HWC,
  /** Batch, height, width, channels: shape {@code [1, height, width, 3]}. */
  NHWC
}
