package se.alipsa.jmlx.models;

final class ResultArrays {
  private ResultArrays() {}

  static float[][] copy(float[][] rows) {
    float[][] result = new float[rows.length][];
    for (int i = 0; i < rows.length; i++) {
      result[i] = rows[i].clone();
    }
    return result;
  }
}
