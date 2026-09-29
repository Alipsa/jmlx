package se.alipsa.jmlx.tokenizer;

/** Translates supported onig character classes to Java's Unicode regex classes. */
final class OnigRegex {

  private OnigRegex() {}

  static String whitespace(String expression) {
    return translate(expression);
  }

  static String translate(String expression) {
    StringBuilder translated = new StringBuilder("(?U)");
    for (int index = 0; index < expression.length(); index++) {
      char current = expression.charAt(index);
      if (current == '\\' && index + 1 < expression.length()) {
        char next = expression.charAt(++index);
        if (next == 's') {
          translated.append("\\p{IsWhite_Space}");
        } else if (next == 'S') {
          translated.append("\\P{IsWhite_Space}");
        } else {
          translated.append('\\').append(next);
        }
      } else if (expression.startsWith("[[:", index)) {
        int end = expression.indexOf(":]]", index + 3);
        if (end < 0) {
          throw new TokenizerException("OnigRegex: unsupported POSIX character class");
        }
        String name = expression.substring(index + 3, end);
        if (!"digit".equals(name)) {
          throw new TokenizerException("OnigRegex: unsupported POSIX character class: " + name);
        }
        translated.append("\\p{Nd}");
        index = end + 2;
      } else {
        translated.append(current);
      }
    }
    return translated.toString();
  }
}
