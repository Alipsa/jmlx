package se.alipsa.jmlx.jinja;

/** Immutable parse-time limits and syntax options. */
public final class TemplateOptions {
  private static final int DEFAULT_MAX_SOURCE_LENGTH = 1_048_576;
  private static final int DEFAULT_MAX_TOKEN_COUNT = 200_000;
  private static final int DEFAULT_MAX_AST_DEPTH = 256;

  /** Defaults matching upstream's public Template constructor. */
  public static final TemplateOptions DEFAULT =
      builder().trimBlocks(true).lstripBlocks(true).build();

  private final int maxSourceLength;
  private final int maxTokenCount;
  private final int maxAstDepth;
  private final boolean trimBlocks;
  private final boolean lstripBlocks;

  private TemplateOptions(
      int maxSourceLength,
      int maxTokenCount,
      int maxAstDepth,
      boolean trimBlocks,
      boolean lstripBlocks) {
    this.maxSourceLength = maxSourceLength;
    this.maxTokenCount = maxTokenCount;
    this.maxAstDepth = maxAstDepth;
    this.trimBlocks = trimBlocks;
    this.lstripBlocks = lstripBlocks;
  }

  /**
   * Starts construction of immutable parse-time options.
   *
   * @return a new builder with default options
   */
  public static Builder builder() {
    return new Builder();
  }

  /**
   * Returns the maximum accepted source length in {@code char}s, checked before preprocessing.
   *
   * @return the maximum source length
   */
  public int maxSourceLength() {
    return maxSourceLength;
  }

  /**
   * Returns the maximum accepted token count while scanning.
   *
   * @return the maximum token count
   */
  public int maxTokenCount() {
    return maxTokenCount;
  }

  /**
   * Returns the maximum nesting depth accepted by the parser.
   *
   * @return the maximum AST depth
   */
  public int maxAstDepth() {
    return maxAstDepth;
  }

  /**
   * Returns whether the first newline after a template tag is stripped automatically.
   *
   * @return whether block trimming is enabled
   */
  public boolean trimBlocks() {
    return trimBlocks;
  }

  /**
   * Returns whether leading spaces/tabs before a template tag are stripped automatically.
   *
   * @return whether block lstripping is enabled
   */
  public boolean lstripBlocks() {
    return lstripBlocks;
  }

  /** Builder for {@link TemplateOptions}. */
  public static final class Builder {
    private int maxSourceLength = DEFAULT_MAX_SOURCE_LENGTH;
    private int maxTokenCount = DEFAULT_MAX_TOKEN_COUNT;
    private int maxAstDepth = DEFAULT_MAX_AST_DEPTH;
    private boolean trimBlocks = true;
    private boolean lstripBlocks = true;

    private Builder() {}

    /**
     * Sets the maximum accepted source length.
     *
     * @param maxSourceLength the maximum source length in {@code char}s; must be positive
     * @return this builder
     */
    public Builder maxSourceLength(int maxSourceLength) {
      if (maxSourceLength <= 0) {
        throw new IllegalArgumentException("maxSourceLength must be positive");
      }
      this.maxSourceLength = maxSourceLength;
      return this;
    }

    /**
     * Sets the maximum accepted token count.
     *
     * @param maxTokenCount the maximum token count; must be positive
     * @return this builder
     */
    public Builder maxTokenCount(int maxTokenCount) {
      if (maxTokenCount <= 0) {
        throw new IllegalArgumentException("maxTokenCount must be positive");
      }
      this.maxTokenCount = maxTokenCount;
      return this;
    }

    /**
     * Sets the maximum nesting depth accepted by the parser.
     *
     * @param maxAstDepth the maximum AST depth; must be positive
     * @return this builder
     */
    public Builder maxAstDepth(int maxAstDepth) {
      if (maxAstDepth <= 0) {
        throw new IllegalArgumentException("maxAstDepth must be positive");
      }
      this.maxAstDepth = maxAstDepth;
      return this;
    }

    /**
     * Sets whether the first newline after a template tag is stripped automatically.
     *
     * @param trimBlocks whether to enable block trimming
     * @return this builder
     */
    public Builder trimBlocks(boolean trimBlocks) {
      this.trimBlocks = trimBlocks;
      return this;
    }

    /**
     * Sets whether leading spaces/tabs before a template tag are stripped automatically.
     *
     * @param lstripBlocks whether to enable block lstripping
     * @return this builder
     */
    public Builder lstripBlocks(boolean lstripBlocks) {
      this.lstripBlocks = lstripBlocks;
      return this;
    }

    /**
     * Creates immutable parse-time options.
     *
     * @return the immutable parse-time options
     */
    public TemplateOptions build() {
      return new TemplateOptions(
          maxSourceLength, maxTokenCount, maxAstDepth, trimBlocks, lstripBlocks);
    }
  }
}
