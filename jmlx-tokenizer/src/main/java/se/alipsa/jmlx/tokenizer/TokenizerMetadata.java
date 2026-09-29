package se.alipsa.jmlx.tokenizer;

import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalLong;

/**
 * Immutable metadata loaded from a tokenizer directory.
 *
 * @param bosToken configured beginning-of-sequence token
 * @param eosToken configured end-of-sequence token
 * @param padToken configured padding token
 * @param unknownToken configured unknown token
 * @param separatorToken configured separator token
 * @param classificationToken configured classification token
 * @param maskToken configured mask token
 * @param modelMaximumLength meaningful configured model maximum, when present
 * @param paddingSide configured padding side
 * @param truncationSide configured truncation side
 * @param chatTemplateNames available template names
 */
public record TokenizerMetadata(
    Optional<String> bosToken,
    Optional<String> eosToken,
    Optional<String> padToken,
    Optional<String> unknownToken,
    Optional<String> separatorToken,
    Optional<String> classificationToken,
    Optional<String> maskToken,
    OptionalLong modelMaximumLength,
    Direction paddingSide,
    Direction truncationSide,
    List<String> chatTemplateNames) {

  /** Defensively copies all optional/list values. */
  public TokenizerMetadata {
    bosToken = Objects.requireNonNull(bosToken, "bosToken");
    eosToken = Objects.requireNonNull(eosToken, "eosToken");
    padToken = Objects.requireNonNull(padToken, "padToken");
    unknownToken = Objects.requireNonNull(unknownToken, "unknownToken");
    separatorToken = Objects.requireNonNull(separatorToken, "separatorToken");
    classificationToken = Objects.requireNonNull(classificationToken, "classificationToken");
    maskToken = Objects.requireNonNull(maskToken, "maskToken");
    modelMaximumLength = Objects.requireNonNull(modelMaximumLength, "modelMaximumLength");
    paddingSide = Objects.requireNonNull(paddingSide, "paddingSide");
    truncationSide = Objects.requireNonNull(truncationSide, "truncationSide");
    chatTemplateNames = List.copyOf(chatTemplateNames);
  }

  static TokenizerMetadata empty() {
    return new TokenizerMetadata(
        Optional.empty(),
        Optional.empty(),
        Optional.empty(),
        Optional.empty(),
        Optional.empty(),
        Optional.empty(),
        Optional.empty(),
        OptionalLong.empty(),
        Direction.RIGHT,
        Direction.RIGHT,
        List.of());
  }
}
