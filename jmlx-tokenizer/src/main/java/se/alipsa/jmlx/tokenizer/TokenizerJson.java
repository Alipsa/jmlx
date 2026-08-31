package se.alipsa.jmlx.tokenizer;

import java.util.List;
import java.util.Objects;

/** The parsed, byte-level-BPE-scoped contents of a {@code tokenizer.json} file. */
public record TokenizerJson(
    NormalizerKind normalizer,
    PreTokenizerConfig preTokenizer,
    List<PostProcessorStep> postProcessor,
    BpeModelConfig model,
    List<AddedToken> addedTokens) {

  /**
   * Defensively copies {@code postProcessor}/{@code addedTokens}, mirroring {@link
   * SpecialTokenInfo}'s and {@link BpeModelConfig}'s own compact constructors: {@link HfTokenizer}
   * retains both lists for its whole lifetime (its constructor and every {@code encode}/{@code
   * decode} call), so a caller mutating the list it passed to {@link TokenizerJsonLoader#load}'s
   * result afterward would otherwise corrupt tokenization mid-flight (PR #14 review round 7,
   * finding 3).
   */
  public TokenizerJson {
    Objects.requireNonNull(postProcessor, "TokenizerJson: postProcessor must not be null");
    Objects.requireNonNull(addedTokens, "TokenizerJson: addedTokens must not be null");
    postProcessor = List.copyOf(postProcessor);
    addedTokens = List.copyOf(addedTokens);
  }
}
