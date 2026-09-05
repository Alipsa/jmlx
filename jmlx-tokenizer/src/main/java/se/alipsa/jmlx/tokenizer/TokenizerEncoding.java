package se.alipsa.jmlx.tokenizer;

import java.util.List;
import java.util.Objects;

/**
 * Immutable columns produced by single-sequence tokenization.
 *
 * @param ids token IDs
 * @param typeIds sequence/type IDs
 * @param attentionMask one for attended tokens and zero for padding
 * @param specialTokensMask one for special tokens and zero otherwise
 * @param offsets original-input UTF-8 byte ranges
 */
public record TokenizerEncoding(
    List<Integer> ids,
    List<Integer> typeIds,
    List<Integer> attentionMask,
    List<Integer> specialTokensMask,
    List<TokenOffset> offsets) {

  /** Defensively copies the columns and requires equal cardinality. */
  public TokenizerEncoding {
    ids = List.copyOf(Objects.requireNonNull(ids, "ids"));
    typeIds = List.copyOf(Objects.requireNonNull(typeIds, "typeIds"));
    attentionMask = List.copyOf(Objects.requireNonNull(attentionMask, "attentionMask"));
    specialTokensMask = List.copyOf(Objects.requireNonNull(specialTokensMask, "specialTokensMask"));
    offsets = List.copyOf(Objects.requireNonNull(offsets, "offsets"));
    int size = ids.size();
    if (typeIds.size() != size
        || attentionMask.size() != size
        || specialTokensMask.size() != size
        || offsets.size() != size) {
      throw new IllegalArgumentException("TokenizerEncoding columns must have equal cardinality");
    }
  }
}
