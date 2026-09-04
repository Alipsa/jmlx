package se.alipsa.jmlx.models;

import java.util.Objects;

/** Shared validation for model architecture identifiers. */
final class ModelTypes {
  private ModelTypes() {}

  static String requireValid(String modelType) {
    Objects.requireNonNull(modelType, "modelType");
    if (modelType.isBlank()) {
      throw new IllegalArgumentException("modelType must not be blank");
    }
    return modelType;
  }
}
