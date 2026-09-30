package se.alipsa.jmlx.models;

import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;

/** Instance-scoped evaluation boundary, injectable from package tests. */
@FunctionalInterface
interface StepBoundaryEvaluator {
  StepBoundaryEvaluator NATIVE = MLX::eval;

  void evaluate(MLXArray... arrays);
}
