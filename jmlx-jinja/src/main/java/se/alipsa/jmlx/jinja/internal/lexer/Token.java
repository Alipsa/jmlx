package se.alipsa.jmlx.jinja.internal.lexer;

import java.util.Objects;
import se.alipsa.jmlx.jinja.SourceLocation;

/**
 * A single lexed token: its kind, raw source value, and start location.
 *
 * <p>Declared {@code public} for the same reason as {@link TokenType}: {@code internal.parser}
 * needs cross-package access. It is internal by package naming and API convention; Java module
 * exports do not prevent classpath consumers from accessing it.
 *
 * @param type the lexical kind of this token
 * @param value the raw source text this token was scanned from
 * @param start the location of the first character of {@code value}
 */
public record Token(TokenType type, String value, SourceLocation start) {
  /** Rejects a null {@code type}, {@code value}, or {@code start}. */
  public Token {
    Objects.requireNonNull(type, "type");
    Objects.requireNonNull(value, "value");
    Objects.requireNonNull(start, "start");
  }
}
