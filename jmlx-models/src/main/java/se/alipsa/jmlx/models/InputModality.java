package se.alipsa.jmlx.models;

/** The input modalities a generation model accepts, as reported by its metadata. */
public enum InputModality {
  /** Plain text prompts; accepted by every supported model. */
  TEXT,
  /**
   * RGB images carried on {@link GenerationRequest#withImages(java.util.List)}; accepted only by
   * vision-language models, which report this in {@code inputModalities()}.
   */
  IMAGE
}
