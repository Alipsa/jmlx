package se.alipsa.jmlx.models;

import java.util.Objects;
import java.util.function.BooleanSupplier;

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

  /** Wraps a thread-safe cancellation predicate. */
  static CancellationToken of(BooleanSupplier supplier) {
    Objects.requireNonNull(supplier, "supplier");
    return supplier::getAsBoolean;
  }
}
