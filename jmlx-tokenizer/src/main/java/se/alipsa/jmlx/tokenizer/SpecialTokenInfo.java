package se.alipsa.jmlx.tokenizer;

import java.util.List;
import java.util.Objects;

/**
 * One entry of a {@code TemplateProcessing} step's {@code special_tokens} map. {@code ids} and
 * {@code tokens} must be non-empty and equal-length: {@link PostProcessorApplier#apply} pairs them
 * positionally, so an empty pair would make the special token vanish silently (e.g. a BOS token
 * simply not appearing) rather than fail loudly, and an unequal-length pair would misalign every
 * entry after the shorter list runs out. Validating here, not just at the {@code
 * TokenizerJsonLoader} call site, means the invariant holds for this record no matter how it's
 * constructed (PR #14 review round 3, finding 1).
 *
 * @param id special token's JSON {@code id} field
 * @param ids token ids to emit
 * @param tokens token texts paired with {@code ids}
 */
public record SpecialTokenInfo(String id, List<Integer> ids, List<String> tokens) {

  /**
   * Enforces the non-empty, equal-length invariant described in the class javadoc, and defensively
   * copies both lists: without {@code List.copyOf}, a caller mutating the list it passed in after
   * construction (e.g. clearing it) would silently violate the very invariant just validated, since
   * a record stores its constructor arguments by reference (PR #14 review round 4, finding 10).
   */
  public SpecialTokenInfo {
    Objects.requireNonNull(id, "SpecialTokenInfo: id must not be null");
    Objects.requireNonNull(ids, "SpecialTokenInfo: ids must not be null");
    Objects.requireNonNull(tokens, "SpecialTokenInfo: tokens must not be null");
    if (ids.isEmpty() || tokens.isEmpty() || ids.size() != tokens.size()) {
      throw new TokenizerException(
          "SpecialTokenInfo: '"
              + id
              + "' has "
              + ids.size()
              + " ids but "
              + tokens.size()
              + " tokens -- PostProcessorApplier pairs them positionally and requires a non-empty,"
              + " equal-length pair of lists");
    }
    ids = List.copyOf(ids);
    tokens = List.copyOf(tokens);
  }
}
