package se.alipsa.jmlx.tokenizer;

/**
 * Matches Rust's {@code char::is_whitespace} (the Unicode {@code White_Space} property), used by HF
 * for whitespace detection everywhere except the byte-level GPT-2 regex and {@code onig}-backed
 * {@code Split}/{@code Replace} patterns, which have their own {@code \s} semantics. {@link
 * Character#isWhitespace} is not equivalent: it is false for U+00A0 (NBSP), U+2007, and U+202F (all
 * {@code White_Space}), and true for the FS/GS/RS/US control characters U+001C-U+001F (none of
 * which are {@code White_Space}) (PR #24 review round 2, finding 3).
 */
final class UnicodeWhitespace {

  private UnicodeWhitespace() {}

  static boolean isWhitespace(int codePoint) {
    return switch (codePoint) {
      case 0x0009,
          0x000A,
          0x000B,
          0x000C,
          0x000D,
          0x0020,
          0x0085,
          0x00A0,
          0x1680,
          0x2028,
          0x2029,
          0x202F,
          0x205F,
          0x3000 ->
          true;
      default -> codePoint >= 0x2000 && codePoint <= 0x200A;
    };
  }
}
