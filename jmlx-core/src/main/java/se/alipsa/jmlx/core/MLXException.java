package se.alipsa.jmlx.core;

/** Thrown when an mlx-c call returns a non-zero status. See {@link MLX#check}. */
public final class MLXException extends RuntimeException {

  /** Creates an exception carrying the mlx-c failure message. */
  public MLXException(String message) {
    super(message);
  }
}
