package se.alipsa.jmlx.core;

/** Thrown when an mlx-c call returns a non-zero status. See {@link MLX#checked}. */
public final class MLXException extends RuntimeException {

  /** Creates an exception carrying the mlx-c failure message. */
  public MLXException(String message) {
    super(message);
  }

  /** Creates an exception carrying {@code cause}'s mlx-c failure message as context. */
  public MLXException(String message, Throwable cause) {
    super(message, cause);
  }
}
