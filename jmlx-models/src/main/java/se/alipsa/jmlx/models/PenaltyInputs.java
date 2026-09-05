package se.alipsa.jmlx.models;

import java.util.LinkedHashMap;
import java.util.Map;

/** Compact immutable token-frequency input for sparse penalty application. */
record PenaltyInputs(int[] tokenIds, float[] counts) {
  PenaltyInputs {
    tokenIds = tokenIds.clone();
    counts = counts.clone();
    if (tokenIds.length != counts.length) {
      throw new IllegalArgumentException("penalty token IDs and counts must have equal length");
    }
  }

  static PenaltyInputs from(Map<Integer, Integer> frequencies, int vocabularySize) {
    int[] ids = new int[frequencies.size()];
    float[] counts = new float[frequencies.size()];
    int index = 0;
    for (Map.Entry<Integer, Integer> entry : frequencies.entrySet()) {
      int id = entry.getKey();
      int count = entry.getValue();
      if (id < 0 || id >= vocabularySize) {
        throw new IllegalArgumentException(
            "history token ID " + id + " outside vocabulary [0, " + vocabularySize + ")");
      }
      if (count <= 0) {
        throw new IllegalArgumentException("history count must be positive for token " + id);
      }
      ids[index] = id;
      counts[index] = count;
      index++;
    }
    return new PenaltyInputs(ids, counts);
  }

  static LinkedHashMap<Integer, Integer> frequencies(int[] prompt) {
    LinkedHashMap<Integer, Integer> result = new LinkedHashMap<>();
    for (int token : prompt) {
      result.merge(token, 1, Integer::sum);
    }
    return result;
  }

  @Override
  public int[] tokenIds() {
    return tokenIds.clone();
  }

  @Override
  public float[] counts() {
    return counts.clone();
  }
}
