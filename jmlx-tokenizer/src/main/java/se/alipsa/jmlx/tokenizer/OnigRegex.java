package se.alipsa.jmlx.tokenizer;

/** Translates the whitespace escapes used by supported onig tokenizer patterns to Java regex. */
final class OnigRegex {

  private OnigRegex() {}

  static String whitespace(String expression) {
    return expression.replace("\\s", "\\p{IsWhite_Space}").replace("\\S", "\\P{IsWhite_Space}");
  }
}
