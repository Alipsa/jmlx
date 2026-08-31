package se.alipsa.jmlx.jinja;

import java.util.Objects;
import java.util.Optional;

/** Base class for all documented hfjinja failures. */
public class HfJinjaException extends RuntimeException {
  private static final long serialVersionUID = 1L;

  /** This failure's stable category. */
  private final ErrorCategory category;

  /** Where in the source this failure occurred, or null if not applicable. */
  private final SourceLocation location;

  /**
   * Creates an exception with no cause.
   *
   * @param message the failure description
   * @param category the failure's stable category
   * @param location where in the source the failure occurred, or null if not applicable
   */
  public HfJinjaException(String message, ErrorCategory category, SourceLocation location) {
    super(message);
    this.category = Objects.requireNonNull(category, "category");
    this.location = location;
  }

  /**
   * Creates an exception wrapping a cause.
   *
   * @param message the failure description
   * @param cause the underlying cause
   * @param category the failure's stable category
   * @param location where in the source the failure occurred, or null if not applicable
   */
  public HfJinjaException(
      String message, Throwable cause, ErrorCategory category, SourceLocation location) {
    super(message, cause);
    this.category = Objects.requireNonNull(category, "category");
    this.location = location;
  }

  /**
   * Returns this failure's stable category.
   *
   * @return the category
   */
  public ErrorCategory category() {
    return category;
  }

  /**
   * Returns where in the source this failure occurred, if known.
   *
   * @return the location, or empty if not applicable
   */
  public Optional<SourceLocation> location() {
    return Optional.ofNullable(location);
  }
}
