package se.alipsa.jmlx.jinja;

import java.io.Serializable;

/**
 * A zero-based source offset and one-based line and column in the caller's original template
 * string, before any preprocessing (the trailing newline strip and, when enabled, {@code
 * trim_blocks}/{@code lstrip_blocks}/the {@code generation} tag strip).
 *
 * @param offset zero-based character offset from the start of the source; diagnostics may point at
 *     or beyond the end of the source
 * @param line one-based line number
 * @param column one-based column number within {@code line}
 */
public record SourceLocation(int offset, int line, int column) implements Serializable {
  private static final long serialVersionUID = 1L;

  /** Rejects a negative offset, or a line or column less than 1. */
  public SourceLocation {
    if (offset < 0 || line < 1 || column < 1) {
      throw new IllegalArgumentException("Invalid source location");
    }
  }
}
