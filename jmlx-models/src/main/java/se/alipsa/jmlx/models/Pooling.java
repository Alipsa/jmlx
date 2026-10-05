package se.alipsa.jmlx.models;

import java.util.Objects;

/** Sentence pooling includes attended special tokens and excludes padding. */
public record Pooling(Mode mode, boolean l2Normalize) {
  /** Token reduction policy. */
  public enum Mode {
    CLS,
    MEAN,
    MAX
  }

  /** Requires a non-null mode. */
  public Pooling {
    Objects.requireNonNull(mode, "mode");
  }
}
