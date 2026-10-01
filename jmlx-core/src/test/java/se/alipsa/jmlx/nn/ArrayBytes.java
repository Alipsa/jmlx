package se.alipsa.jmlx.nn;

import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLXArray;

/** Byte sizes for the memory probes. The switch is exhaustive so a new dtype fails to compile. */
final class ArrayBytes {
  private ArrayBytes() {}

  static int elementBytes(DType dtype) {
    return switch (dtype) {
      case BOOL -> 1;
      case FLOAT16, BFLOAT16 -> 2;
      case FLOAT32, INT32, UINT32 -> 4;
    };
  }

  static long bytes(MLXArray array) {
    return array.size() * elementBytes(array.dtype());
  }
}
