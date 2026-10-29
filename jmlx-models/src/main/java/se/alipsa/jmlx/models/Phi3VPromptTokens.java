package se.alipsa.jmlx.models;

import java.util.regex.Pattern;

/**
 * The prompt-side marker contract of Phi-3.5-Vision ({@code phi3_v}).
 *
 * <p>The reference expands at the string level: the bundled processor splits the rendered prompt on
 * {@code re.split(r"<\|image_\d+\|>", text)} and tokenizes each chunk separately (findings §7.1,
 * {@code req/plans/phase7-3c-reference-findings.md}), so the per-image marker is plain text, not an
 * added token — no BPE merge ever crosses a marker boundary. This class owns the jmlx-side contract
 * for {@code phi3_v} chat prompts (plan WP3): the marker pattern, the 1-based prompt-order marker
 * rendering, and the negative expanded-ID sentinel the model's row-replacement seam injects at.
 *
 * <p>Two distinctions bind the contract. First, marker indices are 1-based and appear in prompt
 * order: marker {@code i} stands for the i-th image part and expands to {@code -i} sentinel rows.
 * Second, the literal single-word token {@code <|image|>} (added token id 32044 in the pinned
 * artifact) is <b>not</b> a marker: the pattern requires {@code image_} followed by at least one
 * digit, so the literal never matches, creates no image slot in the reference, and is passed
 * through as an ordinary token (findings §7.1). The source never holds a complete marker literal:
 * both the pattern and {@link #marker(int)} compose it from parts, for the same transport reason
 * the reference generators build the strings from character codes.
 */
final class Phi3VPromptTokens {

  /** The reference's marker pattern: {@code <|image_} + one or more decimal digits + {@code |>}. */
  static final Pattern MARKER_PATTERN = Pattern.compile("<\\|image_(\\d+)\\|>");

  private Phi3VPromptTokens() {}

  /**
   * Renders marker {@code index} (1-based prompt order) as the reference's marker text.
   *
   * @param index the image's 1-based index in prompt order
   * @return the marker text
   * @throws IllegalArgumentException when the index is not 1-based
   */
  static String marker(int index) {
    if (index < 1) {
      throw new IllegalArgumentException("marker indices are 1-based: " + index);
    }
    return "<|image_" + index + "|>";
  }

  /**
   * The expanded-ID sentinel for image {@code index} (1-based prompt order): the reference's
   * negative pad id {@code -index}, one row per image token.
   *
   * @param index the image's 1-based index in prompt order
   * @return the sentinel id
   * @throws IllegalArgumentException when the index is not 1-based
   */
  static int sentinel(int index) {
    if (index < 1) {
      throw new IllegalArgumentException("marker indices are 1-based: " + index);
    }
    return -index;
  }
}
