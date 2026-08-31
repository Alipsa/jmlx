package se.alipsa.jmlx.tokenizer;

import java.util.Map;
import java.util.Objects;

/**
 * {@code tokenizer.json}'s {@code model} object, scoped to the byte-level-BPE fields this port
 * uses.
 */
public record BpeModelConfig(
    Map<String, Integer> vocab, Map<String, Integer> mergeRank, boolean ignoreMerges) {

  /**
   * Defensively copies {@code vocab}/{@code mergeRank}: {@link BpeMerger} holds a reference to this
   * record and re-reads both maps directly on every {@link BpeMerger#merge} call rather than
   * caching a snapshot, so without {@code Map.copyOf}, a caller mutating the map it passed in after
   * construction would silently change tokenization mid-flight -- the same class of bug {@link
   * SpecialTokenInfo}'s own compact constructor was fixed for (PR #14 review round 7, finding 3).
   */
  public BpeModelConfig {
    Objects.requireNonNull(vocab, "BpeModelConfig: vocab must not be null");
    Objects.requireNonNull(mergeRank, "BpeModelConfig: mergeRank must not be null");
    vocab = Map.copyOf(vocab);
    mergeRank = Map.copyOf(mergeRank);
  }
}
