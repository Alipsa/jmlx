package se.alipsa.jmlx.jinja.internal.lexer;

/**
 * Token kinds understood by the lexer, ported one-to-one from upstream {@code TOKEN_TYPES}.
 *
 * <p>{@code CallOperator} is never emitted by the lexer itself; it is reserved for {@code
 * internal.parser} to synthesize, matching upstream's own comment. Declared {@code public}, unlike
 * the flat {@code internal} package's usual package-private convention, because the not-yet-written
 * {@code internal.parser} package needs cross-package access. It remains an internal implementation
 * type by package naming and API convention; Java module exports do not prevent classpath consumers
 * from accessing it.
 */
public enum TokenType {
  /**
   * Literal template text outside any {@code {{...}}}, {@code {%...%}}, or {@code {#...#}}
   * construct.
   */
  Text,
  /** A numeric literal, e.g. {@code 5} or {@code 5.0}. */
  NumericLiteral,
  /** A string literal, e.g. {@code 'a'} or {@code "a"}. */
  StringLiteral,
  /** A name: a variable, keyword, or built-in reference. */
  Identifier,
  /** The {@code =} assignment/keyword-argument operator. */
  Equals,
  /** The {@code (} delimiter. */
  OpenParen,
  /** The {@code )} delimiter. */
  CloseParen,
  /** The {@code &#123;%} opening statement delimiter. */
  OpenStatement,
  /** The {@code %&#125;} closing statement delimiter. */
  CloseStatement,
  /** The {@code &#123;&#123;} opening expression delimiter. */
  OpenExpression,
  /** The {@code &#125;&#125;} closing expression delimiter. */
  CloseExpression,
  /** The {@code [} delimiter. */
  OpenSquareBracket,
  /** The {@code ]} delimiter. */
  CloseSquareBracket,
  /** The {@code &#123;} delimiter. */
  OpenCurlyBracket,
  /** The {@code &#125;} delimiter. */
  CloseCurlyBracket,
  /** The {@code ,} separator. */
  Comma,
  /** The {@code .} member-access operator. */
  Dot,
  /** The {@code :} separator used in slices and object literals. */
  Colon,
  /** The {@code |} filter operator. */
  Pipe,
  /** Synthesized by the parser for a call following an expression; never emitted by the lexer. */
  CallOperator,
  /** {@code +}, {@code -}, or {@code ~} (string concatenation). */
  AdditiveBinaryOperator,
  /** {@code *}, {@code /}, or {@code %}. */
  MultiplicativeBinaryOperator,
  /** {@code ==}, {@code !=}, {@code <}, {@code <=}, {@code >}, or {@code >=}. */
  ComparisonBinaryOperator,
  /** {@code not}. */
  UnaryOperator,
  /** A {@code {# ... #}} comment. */
  Comment
}
