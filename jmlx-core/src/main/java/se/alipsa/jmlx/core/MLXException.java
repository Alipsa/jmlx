package se.alipsa.jmlx.core;

/** Thrown when an mlx-c call returns a non-zero status. See {@link MLX#check}. */
public final class MLXException extends RuntimeException {

    public MLXException(String message) {
        super(message);
    }
}
