package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.MLXArray;

/**
 * A module whose {@link #forward} takes exactly one input array and produces exactly one output
 * array. Most layers ({@code Linear}, activations, ...) fit this shape; it exists as a separate
 * interface from {@link Module} so a layer can implement it without committing to any particular
 * base-class hierarchy for composite modules that don't (e.g. ones with multiple inputs/outputs).
 */
public interface UnaryModule {

  /** Computes this module's output for input {@code x}. */
  MLXArray forward(MLXArray x);
}
