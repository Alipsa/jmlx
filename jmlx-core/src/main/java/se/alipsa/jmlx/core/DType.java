package se.alipsa.jmlx.core;

import se.alipsa.jmlx.ffi.mlx_h;

/** mlx-c dtype constants supported by this slice. See req/initial-plan.md, Out of scope. */
public enum DType {
    FLOAT32(mlx_h.MLX_FLOAT32()),
    INT32(mlx_h.MLX_INT32());

    private final int nativeValue;

    DType(int nativeValue) {
        this.nativeValue = nativeValue;
    }

    int nativeValue() {
        return nativeValue;
    }

    static DType fromNative(int value) {
        for (DType d : values()) {
            if (d.nativeValue == value) {
                return d;
            }
        }
        throw new IllegalArgumentException(
            "mlx dtype " + value + " is not supported by this slice (only float32/int32 are)");
    }
}
