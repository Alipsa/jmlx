package se.alipsa.jmlx.models;

/**
 * A thread-safe signal that the generation-scope owner polls before prefill and between decode
 * steps. Cancellation does not interrupt an in-progress prefill or invoke native code on the
 * cancelling thread.
 */
@FunctionalInterface
public interface CancellationToken {
  /** A token which is never cancelled. */
  CancellationToken NONE = () -> false;

  /** Returns whether the caller has requested cancellation. */
  boolean isCancelled();
}
