package se.alipsa.jmlx.tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** The byte-level BPE merge algorithm: repeatedly applies the lowest-rank adjacent-pair merge. */
public final class BpeMerger {

  private final BpeModelConfig model;

  /** Prepares a BPE model's vocabulary and merge-rank tables for token merging. */
  public BpeMerger(BpeModelConfig model) {
    this.model = Objects.requireNonNull(model, "BpeMerger: model must not be null");
  }

  /** Merges one byte-level-encoded pre-token chunk into its final BPE symbol sequence. */
  public List<String> merge(String byteLevelWord) {
    Objects.requireNonNull(byteLevelWord, "BpeMerger.merge: byteLevelWord must not be null");
    if (model.ignoreMerges() && model.vocab().containsKey(byteLevelWord)) {
      return List.of(byteLevelWord);
    }
    List<String> symbols = new ArrayList<>();
    byteLevelWord.codePoints().forEach(cp -> symbols.add(new String(Character.toChars(cp))));
    while (symbols.size() > 1) {
      int bestRank = Integer.MAX_VALUE;
      int bestIndex = -1;
      for (int i = 0; i < symbols.size() - 1; i++) {
        Integer rank = model.mergeRank().get(symbols.get(i) + " " + symbols.get(i + 1));
        if (rank != null && rank < bestRank) {
          bestRank = rank;
          bestIndex = i;
        }
      }
      if (bestIndex == -1) {
        break;
      }
      String merged = symbols.get(bestIndex) + symbols.get(bestIndex + 1);
      symbols.set(bestIndex, merged);
      symbols.remove(bestIndex + 1);
    }
    for (String symbol : symbols) {
      if (!model.vocab().containsKey(symbol)) {
        throw new TokenizerException(
            "BpeMerger.merge: merged symbol '"
                + symbol
                + "' has no vocabulary entry (byte_fallback is assumed false for this port's target"
                + " models — see req/plans/phase5-m2-plan.md)");
      }
    }
    return symbols;
  }
}
