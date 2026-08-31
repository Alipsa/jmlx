package se.alipsa.jmlx.jinja.internal.lexer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;
import se.alipsa.jmlx.jinja.ErrorCategory;
import se.alipsa.jmlx.jinja.SourceLocation;
import se.alipsa.jmlx.jinja.TemplateOptions;
import se.alipsa.jmlx.jinja.TemplateRenderException;
import se.alipsa.jmlx.jinja.TemplateSyntaxException;

/**
 * Scanner ported from upstream {@code lexer.ts}: whitespace controls, tokens, and source spans.
 *
 * <p>Reported {@link SourceLocation}s refer to the caller's original string. Preprocessing (the
 * trailing newline strip and, when enabled, {@code trim_blocks}/{@code lstrip_blocks}/the {@code
 * generation} tag strip) records every removal in a compact origin map, and scanner positions are
 * mapped back through it before a location is constructed.
 */
public final class Lexer {
  /**
   * The ECMA-262 WhiteSpace and LineTerminator character set, as a regex bracket-class body built
   * from explicit backslash-u escapes. It is wider than Java's default whitespace notions: it adds
   * NBSP, the Unicode {@code Zs} space separators, U+2028/U+2029, and U+FEFF, none of which {@link
   * Character#isWhitespace} or {@link Pattern}'s default {@code \s} treat as whitespace. Kept in
   * sync by hand with {@link #isJsWhitespace(char)} below — a bracket character class and a
   * per-char predicate can't share one definition, since the class needs the {@code X-Y} range
   * syntax that a plain predicate expresses as a numeric comparison instead.
   */
  private static final String JS_WHITESPACE_CHAR_CLASS =
      "\\t\\n"
          + "\\u000B\\f\\r"
          + "\\u0020\\u00A0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000\\uFEFF";

  /*
   * ECMAScript's multiline ^ recognizes only LF, CR, LS, and PS as line terminators. Java also
   * recognizes NEL (U+0085), so use an explicit preceding-character assertion rather than (?m).
   */
  private static final Pattern LSTRIP_BLOCKS_PATTERN =
      Pattern.compile("(?:^|(?<=[\\n\\r\\u2028\\u2029]))[ \\t]*(\\{[#%-])");
  private static final Pattern TRIM_BLOCKS_PATTERN = Pattern.compile("([#%-]\\})\\n");

  /**
   * Strips the HF-transformers-specific {@code {% generation %}}/{@code {% endgeneration %}} tags,
   * honoring their {@code -} whitespace-control modifiers. Upstream compiles this with JS's {@code
   * gs} flags. Its literal <code>&#123;%</code> prefix is deliberately matched before surrounding
   * whitespace is examined, preventing unsuccessful searches through long whitespace runs from
   * becoming quadratic.
   */
  private static final Pattern GENERATION_TAG_PATTERN =
      Pattern.compile(
          "\\{%(-?)["
              + JS_WHITESPACE_CHAR_CLASS
              + "]*(?:end)?generation["
              + JS_WHITESPACE_CHAR_CLASS
              + "]*(-?)%\\}");

  private static final Map<Character, Character> ESCAPE_CHARACTERS =
      Map.of(
          'n', '\n', 't', '\t', 'r', '\r', 'b', '\b', 'f', '\f', 'v', '\u000B', '\'', '\'', '"',
          '"', '\\', '\\');

  /** Match order is behaviorally significant: longer prefixes must be tried before shorter ones. */
  private static final List<TokenPattern> ORDERED_MAPPING_TABLE =
      List.of(
          new TokenPattern("{%", TokenType.OpenStatement),
          new TokenPattern("%}", TokenType.CloseStatement),
          new TokenPattern("{{", TokenType.OpenExpression),
          new TokenPattern("}}", TokenType.CloseExpression),
          new TokenPattern("(", TokenType.OpenParen),
          new TokenPattern(")", TokenType.CloseParen),
          new TokenPattern("{", TokenType.OpenCurlyBracket),
          new TokenPattern("}", TokenType.CloseCurlyBracket),
          new TokenPattern("[", TokenType.OpenSquareBracket),
          new TokenPattern("]", TokenType.CloseSquareBracket),
          new TokenPattern(",", TokenType.Comma),
          new TokenPattern(".", TokenType.Dot),
          new TokenPattern(":", TokenType.Colon),
          new TokenPattern("|", TokenType.Pipe),
          new TokenPattern("<=", TokenType.ComparisonBinaryOperator),
          new TokenPattern(">=", TokenType.ComparisonBinaryOperator),
          new TokenPattern("==", TokenType.ComparisonBinaryOperator),
          new TokenPattern("!=", TokenType.ComparisonBinaryOperator),
          new TokenPattern("<", TokenType.ComparisonBinaryOperator),
          new TokenPattern(">", TokenType.ComparisonBinaryOperator),
          new TokenPattern("+", TokenType.AdditiveBinaryOperator),
          new TokenPattern("-", TokenType.AdditiveBinaryOperator),
          new TokenPattern("~", TokenType.AdditiveBinaryOperator),
          new TokenPattern("*", TokenType.MultiplicativeBinaryOperator),
          new TokenPattern("/", TokenType.MultiplicativeBinaryOperator),
          new TokenPattern("%", TokenType.MultiplicativeBinaryOperator),
          new TokenPattern("=", TokenType.Equals));

  private Lexer() {}

  /**
   * Scans {@code source} into a token list.
   *
   * @param source the template source
   * @param options parse-time limits and syntax options
   * @return the scanned tokens, in source order
   */
  public static List<Token> tokenize(String source, TemplateOptions options) {
    Objects.requireNonNull(source, "source");
    Objects.requireNonNull(options, "options");
    if (source.length() > options.maxSourceLength()) {
      throw new TemplateRenderException(
          "Source length "
              + source.length()
              + " exceeds the configured limit of "
              + options.maxSourceLength(),
          ErrorCategory.RESOURCE_LIMIT,
          null);
    }
    return new Scanner(source, options).scan();
  }

  private static PreprocessedSource preprocess(String source, TemplateOptions options) {
    var text = source;
    List<int[]> originalRemovals = null;
    if (text.endsWith("\n")) {
      originalRemovals = new ArrayList<>();
      originalRemovals.add(new int[] {text.length() - 1, text.length()});
      text = text.substring(0, text.length() - 1);
    }
    if (options.lstripBlocks()) {
      var matcher = LSTRIP_BLOCKS_PATTERN.matcher(text);
      List<int[]> ranges = null;
      while (matcher.find()) {
        if (matcher.start() != matcher.start(1)) {
          if (ranges == null) {
            ranges = new ArrayList<>();
          }
          ranges.add(new int[] {matcher.start(), matcher.start(1)});
        }
      }
      if (ranges != null) {
        originalRemovals = addOriginalRemovals(originalRemovals, ranges);
        text = removeRanges(text, ranges);
      }
    }
    if (options.trimBlocks()) {
      var matcher = TRIM_BLOCKS_PATTERN.matcher(text);
      List<int[]> ranges = null;
      while (matcher.find()) {
        if (matcher.end(1) != matcher.end()) {
          if (ranges == null) {
            ranges = new ArrayList<>();
          }
          ranges.add(new int[] {matcher.end(1), matcher.end()});
        }
      }
      if (ranges != null) {
        originalRemovals = addOriginalRemovals(originalRemovals, ranges);
        text = removeRanges(text, ranges);
      }
    }
    if (text.indexOf("generation") >= 0) {
      var ranges = generationRemovalRanges(text);
      if (!ranges.isEmpty()) {
        originalRemovals = addOriginalRemovals(originalRemovals, ranges);
        text = removeRanges(text, ranges);
      }
    }
    return originalRemovals == null
        ? new PreprocessedSource(text, null, null, null)
        : new PreprocessedSource(text, new OriginMap(originalRemovals), source, lineStarts(source));
  }

  private static String removeRanges(String source, List<int[]> ranges) {
    var text = new StringBuilder(source.length());
    var copyPosition = 0;
    for (var range : ranges) {
      text.append(source, copyPosition, range[0]);
      copyPosition = range[1];
    }
    return text.append(source, copyPosition, source.length()).toString();
  }

  /**
   * Converts {@code ranges} from current-text coordinates to original-source coordinates and merges
   * them with the removals already recorded there. Both inputs are ascending, so mapping and
   * merging use a shared walking cursor.
   */
  private static List<int[]> addOriginalRemovals(List<int[]> existing, List<int[]> ranges) {
    var mappedRanges = new ArrayList<int[]>(ranges.size());
    var existingIndex = 0;
    var cumulativeShift = 0;
    for (var range : ranges) {
      while (existing != null
          && existingIndex < existing.size()
          && range[0] >= existing.get(existingIndex)[0] - cumulativeShift) {
        var removal = existing.get(existingIndex++);
        cumulativeShift += removal[1] - removal[0];
      }
      var start = range[0] + cumulativeShift;
      while (existing != null
          && existingIndex < existing.size()
          && range[1] >= existing.get(existingIndex)[0] - cumulativeShift) {
        var removal = existing.get(existingIndex++);
        cumulativeShift += removal[1] - removal[0];
      }
      mappedRanges.add(new int[] {start, range[1] + cumulativeShift});
    }

    var merged = new ArrayList<int[]>();
    var existingRange = 0;
    var mappedRange = 0;
    while ((existing != null && existingRange < existing.size())
        || mappedRange < mappedRanges.size()) {
      var range =
          mappedRange == mappedRanges.size()
                  || (existing != null
                      && existingRange < existing.size()
                      && existing.get(existingRange)[0] <= mappedRanges.get(mappedRange)[0])
              ? existing.get(existingRange++)
              : mappedRanges.get(mappedRange++);
      if (!merged.isEmpty() && range[0] <= merged.getLast()[1]) {
        merged.getLast()[1] = Math.max(merged.getLast()[1], range[1]);
      } else {
        merged.add(new int[] {range[0], range[1]});
      }
    }
    return merged;
  }

  /**
   * Computes the character ranges removed by the {@code generation}-tag strip, in {@code template}
   * coordinates. The tag itself is always removed; the surrounding JS-whitespace runs are removed
   * only on the sides carrying a {@code -} modifier. {@code copyPosition} bounds each whitespace
   * scan to the region after the previous match, so the returned ranges are ascending and
   * non-overlapping.
   */
  private static List<int[]> generationRemovalRanges(String template) {
    var matcher = GENERATION_TAG_PATTERN.matcher(template);
    var ranges = new ArrayList<int[]>();
    var copyPosition = 0;
    while (matcher.find()) {
      var whitespaceBefore = matcher.start();
      while (whitespaceBefore > copyPosition
          && isJsWhitespace(template.charAt(whitespaceBefore - 1))) {
        whitespaceBefore--;
      }
      var whitespaceAfter = matcher.end();
      while (whitespaceAfter < template.length()
          && isJsWhitespace(template.charAt(whitespaceAfter))) {
        whitespaceAfter++;
      }
      ranges.add(
          new int[] {
            matcher.group(1).isEmpty() ? matcher.start() : whitespaceBefore,
            matcher.group(2).isEmpty() ? matcher.end() : whitespaceAfter
          });
      copyPosition = whitespaceAfter;
    }
    return ranges;
  }

  /**
   * Computes the original-source offsets at which each line begins, using the scanner's
   * line-terminator rules: LF, CR, LS, and PS each end a line, and CRLF counts as one break.
   */
  private static int[] lineStarts(String source) {
    var starts = new int[Math.min(source.length() + 1, 16)];
    var size = 1;
    starts[0] = 0;
    for (var i = 0; i < source.length(); i++) {
      var c = source.charAt(i);
      if (isLineTerminator(c)) {
        if (c == '\r' && i + 1 < source.length() && source.charAt(i + 1) == '\n') {
          i++;
        }
        if (size == starts.length) {
          starts = Arrays.copyOf(starts, starts.length * 2);
        }
        starts[size++] = i + 1;
      }
    }
    return Arrays.copyOf(starts, size);
  }

  private static boolean isLineTerminator(char c) {
    return c == '\n' || c == '\r' || c == '\u2028' || c == '\u2029';
  }

  /**
   * The preprocessed template text plus the data needed to map positions in it back to the caller's
   * original string.
   *
   * @param text the text the scanner works on, after all preprocessing removals
   * @param originMap compact mapping from preprocessed offsets to original-source offsets
   * @param source caller-provided source before preprocessing
   * @param lineStarts original-source offsets at which each line begins
   */
  private record PreprocessedSource(
      String text, OriginMap originMap, String source, int[] lineStarts) {

    SourceLocation locationAt(int preprocessedPosition) {
      var offset = originMap.originalOffsetAt(preprocessedPosition);
      if (offset > 0
          && offset < source.length()
          && source.charAt(offset) == '\n'
          && source.charAt(offset - 1) == '\r') {
        return new SourceLocation(offset, lineAt(offset + 1) + 1, 1);
      }
      var lineIndex = lineAt(offset);
      return new SourceLocation(offset, lineIndex + 1, offset - lineStarts[lineIndex] + 1);
    }

    private int lineAt(int offset) {
      var insertion = Arrays.binarySearch(lineStarts, offset);
      return insertion >= 0 ? insertion : -insertion - 2;
    }
  }

  /** A compact map from preprocessed offsets to original offsets, represented by removals. */
  private static final class OriginMap {
    private final int[] preprocessedBreakpoints;
    private final int[] cumulativeShifts;

    OriginMap(List<int[]> removals) {
      preprocessedBreakpoints = new int[removals.size()];
      cumulativeShifts = new int[removals.size()];
      var shift = 0;
      for (var i = 0; i < removals.size(); i++) {
        var removal = removals.get(i);
        preprocessedBreakpoints[i] = removal[0] - shift;
        shift += removal[1] - removal[0];
        cumulativeShifts[i] = shift;
      }
    }

    int originalOffsetAt(int preprocessedPosition) {
      var index = Arrays.binarySearch(preprocessedBreakpoints, preprocessedPosition);
      if (index < 0) {
        index = -index - 2;
      }
      return preprocessedPosition + (index < 0 ? 0 : cumulativeShifts[index]);
    }
  }

  private static boolean isWordChar(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
  }

  private static boolean isIntegerChar(char c) {
    return c >= '0' && c <= '9';
  }

  /** See {@link #JS_WHITESPACE_CHAR_CLASS} for why this can't share a definition with it. */
  private static boolean isJsWhitespace(char c) {
    return c == '\t'
        || c == '\n'
        || c == '\u000B'
        || c == '\f'
        || c == '\r'
        || c == '\u0020'
        || c == '\u00A0'
        || c == '\u1680'
        || (c >= '\u2000' && c <= '\u200A')
        || c == '\u2028'
        || c == '\u2029'
        || c == '\u202F'
        || c == '\u205F'
        || c == '\u3000'
        || c == '\uFEFF';
  }

  private record TokenPattern(String sequence, TokenType type) {}

  @FunctionalInterface
  private interface CharPredicate {
    boolean test(char c);
  }

  private static final class Scanner {
    private final PreprocessedSource preprocessed;
    private final String src;
    private final TemplateOptions options;
    private final List<Token> tokens = new ArrayList<>();
    private int cursorPosition;
    private int line = 1;
    private int column = 1;
    private int curlyBracketDepth;

    Scanner(String source, TemplateOptions options) {
      this.preprocessed = preprocess(source, options);
      this.src = preprocessed.text();
      this.options = options;
    }

    List<Token> scan() {
      main:
      while (cursorPosition < src.length()) {
        var lastType = lastTokenType();
        if (lastType == null
            || lastType == TokenType.CloseStatement
            || lastType == TokenType.CloseExpression
            || lastType == TokenType.Comment) {
          var start = currentLocation();
          var text = new StringBuilder();
          while (cursorPosition < src.length()
              && !(charAt(cursorPosition) == '{'
                  && (charAt(cursorPosition + 1) == '%'
                      || charAt(cursorPosition + 1) == '{'
                      || charAt(cursorPosition + 1) == '#'))) {
            text.append(charAt(cursorPosition));
            advance();
          }
          if (text.length() > 0) {
            addToken(TokenType.Text, text.toString(), start);
            continue;
          }
        }

        if (charAt(cursorPosition) == '{' && charAt(cursorPosition + 1) == '#') {
          var start = currentLocation();
          advance(2); // skip "{#"
          var stripBefore = charAt(cursorPosition) == '-';
          if (stripBefore) {
            advance();
          }
          var comment = new StringBuilder();
          while (charAt(cursorPosition) != '#' || charAt(cursorPosition + 1) != '}') {
            if (cursorPosition + 2 >= src.length()) {
              throw syntaxError("Missing end of comment tag", currentLocation());
            }
            comment.append(charAt(cursorPosition));
            advance();
          }
          var stripAfter = comment.length() > 0 && comment.charAt(comment.length() - 1) == '-';
          if (stripAfter) {
            comment.setLength(comment.length() - 1);
          }
          if (stripBefore) {
            stripTrailingWhitespace();
          }
          addToken(TokenType.Comment, comment.toString(), start);
          advance(2); // skip "#}"
          if (stripAfter) {
            skipLeadingWhitespace();
          }
          continue;
        }

        if (regionMatches("{%-")) {
          stripTrailingWhitespace();
          addToken(TokenType.OpenStatement, "{%", currentLocation());
          advance(3);
          continue;
        }

        if (regionMatches("{{-")) {
          stripTrailingWhitespace();
          addToken(TokenType.OpenExpression, "{{", currentLocation());
          curlyBracketDepth = 0;
          advance(3);
          continue;
        }

        consumeWhile(Lexer::isJsWhitespace);

        if (regionMatches("-%}")) {
          addToken(TokenType.CloseStatement, "%}", currentLocation());
          advance(3);
          skipLeadingWhitespace();
          continue;
        }

        if (regionMatches("-}}")) {
          addToken(TokenType.CloseExpression, "}}", currentLocation());
          advance(3);
          skipLeadingWhitespace();
          continue;
        }

        var currentChar = charAt(cursorPosition);

        if (currentChar == '-' || currentChar == '+') {
          var lastOperatorType = lastTokenType();
          if (lastOperatorType == null || lastOperatorType == TokenType.Text) {
            throw syntaxError("Unexpected character: " + currentChar, currentLocation());
          }
          switch (lastOperatorType) {
            case Identifier, NumericLiteral, StringLiteral, CloseParen, CloseSquareBracket -> {
              // Part of a binary operator: fall through to the mapping-table scan below.
            }
            default -> {
              var start = currentLocation();
              advance(); // consume the unary operator character
              var digits = consumeWhile(Lexer::isIntegerChar);
              addToken(
                  digits.isEmpty() ? TokenType.UnaryOperator : TokenType.NumericLiteral,
                  digits.isEmpty() ? String.valueOf(currentChar) : currentChar + digits,
                  start);
              continue main;
            }
          }
        }

        for (var pattern : ORDERED_MAPPING_TABLE) {
          if (pattern.sequence().equals("}}") && curlyBracketDepth > 0) {
            continue;
          }
          if (regionMatches(pattern.sequence())) {
            addToken(pattern.type(), pattern.sequence(), currentLocation());
            if (pattern.type() == TokenType.OpenExpression) {
              curlyBracketDepth = 0;
            } else if (pattern.type() == TokenType.OpenCurlyBracket) {
              curlyBracketDepth++;
            } else if (pattern.type() == TokenType.CloseCurlyBracket) {
              curlyBracketDepth--;
            }
            advance(pattern.sequence().length());
            continue main;
          }
        }

        if (currentChar == '\'' || currentChar == '"') {
          var quote = currentChar;
          var start = currentLocation();
          advance(); // skip opening quote
          var value = consumeWhile(c -> c != quote);
          addToken(TokenType.StringLiteral, value, start);
          advance(); // skip closing quote
          continue;
        }

        if (isIntegerChar(currentChar)) {
          var start = currentLocation();
          var number = consumeWhile(Lexer::isIntegerChar);
          if (lastTokenType() != TokenType.Dot
              && charAt(cursorPosition) == '.'
              && isIntegerChar(charAt(cursorPosition + 1))) {
            advance(); // consume '.'
            var fraction = consumeWhile(Lexer::isIntegerChar);
            number = number + "." + fraction;
          }
          addToken(TokenType.NumericLiteral, number, start);
          continue;
        }

        if (isWordChar(currentChar)) {
          var start = currentLocation();
          var word = consumeWhile(Lexer::isWordChar);
          addToken(TokenType.Identifier, word, start);
          continue;
        }

        throw syntaxError("Unexpected character: " + currentChar, currentLocation());
      }
      return List.copyOf(tokens);
    }

    private TokenType lastTokenType() {
      return tokens.isEmpty() ? null : tokens.get(tokens.size() - 1).type();
    }

    private char charAt(int index) {
      return index >= 0 && index < src.length() ? src.charAt(index) : '\0';
    }

    private boolean regionMatches(String literal) {
      return src.regionMatches(cursorPosition, literal, 0, literal.length());
    }

    private void advance() {
      var current = charAt(cursorPosition);
      if (isLineTerminator(current)) {
        if (current != '\n' || charAt(cursorPosition - 1) != '\r') {
          line++;
          column = 1;
        }
      } else {
        column++;
      }
      cursorPosition++;
    }

    private void advance(int count) {
      for (var i = 0; i < count; i++) {
        advance();
      }
    }

    private SourceLocation currentLocation() {
      return preprocessed.originMap() == null
          ? new SourceLocation(cursorPosition, line, column)
          : preprocessed.locationAt(cursorPosition);
    }

    /**
     * Consumes characters while {@code predicate} holds, resolving backslash escapes inline. Only
     * the string-literal call site ever exercises the escape branch: {@code '\\'} never satisfies
     * {@link #isJsWhitespace}, {@link #isIntegerChar}, or {@link #isWordChar}, so it's structurally
     * unreachable from the other three call sites. Faithfully preserves an upstream quirk: the "ran
     * off the end" check fires unconditionally right after consuming any character (escaped or not)
     * — this is upstream's only mechanism for detecting an unterminated tag, since more content (a
     * closing delimiter) is always grammatically expected while scanning inside one.
     */
    private String consumeWhile(CharPredicate predicate) {
      var text = new StringBuilder();
      while (predicate.test(charAt(cursorPosition))) {
        if (charAt(cursorPosition) == '\\') {
          var escapeLocation = currentLocation();
          advance();
          if (cursorPosition >= src.length()) {
            throw syntaxError("Unexpected end of input", escapeLocation);
          }
          var escaped = charAt(cursorPosition);
          advance();
          var unescaped = ESCAPE_CHARACTERS.get(escaped);
          if (unescaped == null) {
            throw syntaxError("Unexpected escaped character: " + escaped, escapeLocation);
          }
          text.append(unescaped.charValue());
          continue;
        }
        text.append(charAt(cursorPosition));
        advance();
        if (cursorPosition >= src.length()) {
          throw syntaxError("Unexpected end of input", currentLocation());
        }
      }
      return text.toString();
    }

    private void stripTrailingWhitespace() {
      if (tokens.isEmpty()) {
        return;
      }
      var lastIndex = tokens.size() - 1;
      var last = tokens.get(lastIndex);
      if (last.type() != TokenType.Text) {
        return;
      }
      var trimmed = trimTrailingJsWhitespace(last.value());
      if (trimmed.isEmpty()) {
        tokens.remove(lastIndex);
      } else {
        tokens.set(lastIndex, new Token(TokenType.Text, trimmed, last.start()));
      }
    }

    private static String trimTrailingJsWhitespace(String value) {
      var end = value.length();
      while (end > 0 && isJsWhitespace(value.charAt(end - 1))) {
        end--;
      }
      return value.substring(0, end);
    }

    private void skipLeadingWhitespace() {
      while (cursorPosition < src.length() && isJsWhitespace(charAt(cursorPosition))) {
        advance();
      }
    }

    private void addToken(TokenType type, String value, SourceLocation start) {
      if (tokens.size() >= options.maxTokenCount()) {
        throw new TemplateRenderException(
            "Token count exceeds the configured limit of " + options.maxTokenCount(),
            ErrorCategory.RESOURCE_LIMIT,
            start);
      }
      tokens.add(new Token(type, value, start));
    }

    private static TemplateSyntaxException syntaxError(String message, SourceLocation location) {
      return new TemplateSyntaxException(message, location);
    }
  }
}
