package se.alipsa.jmlx.models;

import java.util.function.Consumer;

/**
 * Public local text-generation API. Native work runs only on the generation scope's owner thread.
 */
public interface TextGenerationModel extends AutoCloseable {
  /** Returns stable metadata without exposing an architecture-specific configuration type. */
  ModelMetadata metadata();

  /**
   * Generates tokens and synchronously sends token events followed by one terminal event to {@code
   * listener}. EOS has precedence when a token belongs to both EOS and stop sets. For legacy
   * compatibility, an EOS ID remains in {@link GenerationResult#generatedTokenIds()} and is sent as
   * a token event; an explicit stop-token ID is excluded from both. If a token listener throws,
   * generation aborts, its native scopes are closed, and a {@link GenerationAbortedException}
   * exposes the partial token sequence; no terminal event is sent after that listener failure. A
   * terminal-listener exception is logged because generation has already completed and its result
   * remains available. Sampling is request-local and requires an explicit seed; when requested,
   * token-event and result log probabilities describe the selected token after every policy filter.
   */
  GenerationResult generate(GenerationRequest request, Consumer<GenerationEvent> listener);

  /**
   * Closes resources owned directly by this model. Every current implementation is loaded into a
   * caller-owned scope and owns no independent resources, so this is currently a no-op; a future
   * owning loader may provide an implementation that closes its own scope.
   */
  @Override
  default void close() {}
}
