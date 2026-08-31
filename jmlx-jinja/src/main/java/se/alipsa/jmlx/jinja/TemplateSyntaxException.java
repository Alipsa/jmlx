package se.alipsa.jmlx.jinja;

/** A template source error. */
public final class TemplateSyntaxException extends HfJinjaException {
  private static final long serialVersionUID = 1L;

  /**
   * Creates a syntax exception.
   *
   * @param message the failure description
   * @param location where in the source the failure occurred
   */
  public TemplateSyntaxException(String message, SourceLocation location) {
    super(message, ErrorCategory.SYNTAX, location);
  }
}
