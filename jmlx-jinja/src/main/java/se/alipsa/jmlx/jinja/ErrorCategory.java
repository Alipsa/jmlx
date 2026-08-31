package se.alipsa.jmlx.jinja;

/** Stable categories for template and host-boundary failures. */
public enum ErrorCategory {
  /** The template source could not be parsed. */
  SYNTAX,
  /** A name resolved to nothing, or an access target does not support the attempted operation. */
  UNDEFINED_OR_ACCESS,
  /** A value's runtime type does not support the attempted operation. */
  TYPE,
  /** A call was made with the wrong number of arguments. */
  ARITY,
  /** A value was outside the range or shape an operation requires. */
  VALUE,
  /** The template explicitly invoked {@code raise_exception}. */
  EXPLICIT_RAISE,
  /** A host function rejected its arguments or failed while executing. */
  HOST_FUNCTION,
  /** A value could not be converted across the host/template value boundary. */
  HOST_CONVERSION,
  /** A configured parse-time or render-time limit was exceeded. */
  RESOURCE_LIMIT,
  /** Producing rendered output failed. */
  OUTPUT
}
