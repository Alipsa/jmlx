package se.alipsa.jmlx.jinja;

/** A host-boundary or render-time error. */
public final class TemplateRenderException extends JinjaException {
  private static final long serialVersionUID = 1L;

  /**
   * Creates an exception with no cause.
   *
   * @param message the failure description
   * @param category the failure's stable category
   * @param location where in the source the failure occurred, or null if not applicable
   */
  public TemplateRenderException(String message, ErrorCategory category, SourceLocation location) {
    super(message, category, location);
  }

  /**
   * Creates an exception wrapping a cause.
   *
   * @param message the failure description
   * @param cause the underlying cause
   * @param category the failure's stable category
   * @param location where in the source the failure occurred, or null if not applicable
   */
  public TemplateRenderException(
      String message, Throwable cause, ErrorCategory category, SourceLocation location) {
    super(message, cause, category, location);
  }
}
