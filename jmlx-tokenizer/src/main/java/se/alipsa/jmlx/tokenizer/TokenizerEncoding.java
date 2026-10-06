package se.alipsa.jmlx.tokenizer;

import java.util.List;
import java.util.Objects;

/**
 * Immutable columns produced by single-sequence or supported BERT pair tokenization.
 *
 * @param ids token IDs
 * @param typeIds sequence/type IDs
 * @param attentionMask one for attended tokens and zero for padding
 * @param specialTokensMask one for special tokens and zero otherwise
 * @param offsets input-local original UTF-8 byte ranges; pair identity follows BERT type IDs/masks
 * @param tokens emitted token strings
 */
public record TokenizerEncoding(
    List<Integer> ids,
    List<Integer> typeIds,
    List<Integer> attentionMask,
    List<Integer> specialTokensMask,
    List<TokenOffset> offsets,
    List<String> tokens) {

  /**
   * Compatibility constructor for callers that do not supply token strings; each token string is
   * empty.
   *
   * @param ids token IDs
   * @param typeIds sequence/type IDs
   * @param attentionMask one for attended tokens and zero for padding
   * @param specialTokensMask one for special tokens and zero otherwise
   * @param offsets original-input UTF-8 byte ranges
   */
  public TokenizerEncoding(
      List<Integer> ids,
      List<Integer> typeIds,
      List<Integer> attentionMask,
      List<Integer> specialTokensMask,
      List<TokenOffset> offsets) {
    this(
        ids,
        typeIds,
        attentionMask,
        specialTokensMask,
        offsets,
        java.util.Collections.nCopies(ids.size(), ""));
  }

  /** Defensively copies the columns and requires equal cardinality. */
  public TokenizerEncoding {
    ids = List.copyOf(Objects.requireNonNull(ids, "ids"));
    typeIds = List.copyOf(Objects.requireNonNull(typeIds, "typeIds"));
    attentionMask = List.copyOf(Objects.requireNonNull(attentionMask, "attentionMask"));
    specialTokensMask = List.copyOf(Objects.requireNonNull(specialTokensMask, "specialTokensMask"));
    offsets = List.copyOf(Objects.requireNonNull(offsets, "offsets"));
    tokens = List.copyOf(Objects.requireNonNull(tokens, "tokens"));
    int size = ids.size();
    if (typeIds.size() != size
        || attentionMask.size() != size
        || specialTokensMask.size() != size
        || offsets.size() != size
        || tokens.size() != size) {
      throw new IllegalArgumentException("TokenizerEncoding columns must have equal cardinality");
    }
  }
}
