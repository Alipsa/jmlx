package se.alipsa.jmlx.core;

import se.alipsa.jmlx.ffi.mlx_h;

/**
 * The mlx-c dtype constants supported by this slice. See req/initial-plan.md, Out of scope, and
 * req/phase4-plan.md §4.
 */
public enum DType {
  FLOAT32(mlx_h.MLX_FLOAT32(), true),
  INT32(mlx_h.MLX_INT32(), false),
  BOOL(mlx_h.MLX_BOOL(), false),
  UINT32(mlx_h.MLX_UINT32(), false),
  FLOAT16(mlx_h.MLX_FLOAT16(), true),
  BFLOAT16(mlx_h.MLX_BFLOAT16(), true);

  private final int nativeValue;
  private final boolean inexact;

  DType(int nativeValue, boolean inexact) {
    this.nativeValue = nativeValue;
    this.inexact = inexact;
  }

  int nativeValue() {
    return nativeValue;
  }

  /**
   * Mirrors native's own {@code issubdtype(dtype, inexact)} predicate (upstream {@code
   * ops.cpp:3222-3230}): float16, bfloat16, float32 and complex64 report {@code true}; integer and
   * boolean dtypes report {@code false}. Written as an allowlist field on each constant rather than
   * {@code return this != INT32 && this != UINT32 && this != BOOL}, so adding a future non-inexact
   * dtype (e.g. {@code INT8}, {@code INT64}) cannot silently start reporting {@code true}.
   */
  public boolean isInexact() {
    return inexact;
  }

  static DType fromNative(int value) {
    for (DType d : values()) {
      if (d.nativeValue == value) {
        return d;
      }
    }
    throw new IllegalArgumentException(
        "mlx dtype "
            + value
            + " is not supported by this slice (only float32/int32/bool/uint32/float16/bfloat16"
            + " are)");
  }
}
