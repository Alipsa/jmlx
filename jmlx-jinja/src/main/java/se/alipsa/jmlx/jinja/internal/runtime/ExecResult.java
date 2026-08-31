package se.alipsa.jmlx.jinja.internal.runtime;

sealed interface ExecResult permits ExecResult.Normal, ExecResult.Break, ExecResult.Continue {
  record Normal(String output) implements ExecResult {}

  enum Break implements ExecResult {
    INSTANCE
  }

  enum Continue implements ExecResult {
    INSTANCE
  }
}
