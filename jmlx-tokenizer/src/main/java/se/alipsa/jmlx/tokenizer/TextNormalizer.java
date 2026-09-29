package se.alipsa.jmlx.tokenizer;

import java.text.Normalizer;
import java.util.Objects;

/**
 * Applies a {@code tokenizer.json} normalizer ({@code None} or {@code NFC}) to input text.
 *
 * @deprecated no longer used by {@link HfTokenizer}, whose pipeline supports more components than
 *     this class; use {@link HfTokenizer} for text normalization that matches Hugging Face
 */
@Deprecated
public final class TextNormalizer {

  private TextNormalizer() {}

  /**
   * Normalizes {@code text} per {@code kind}.
   *
   * @param kind normalizer to apply
   * @param text input text
   * @return normalized text
   */
  public static String normalize(NormalizerKind kind, String text) {
    Objects.requireNonNull(kind, "TextNormalizer.normalize: kind must not be null");
    Objects.requireNonNull(text, "TextNormalizer.normalize: text must not be null");
    return switch (kind) {
      case NONE -> text;
      case NFC -> Normalizer.normalize(text, Normalizer.Form.NFC);
    };
  }
}
