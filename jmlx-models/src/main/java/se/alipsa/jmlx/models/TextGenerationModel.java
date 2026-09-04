package se.alipsa.jmlx.models;

import java.util.function.Consumer;

/**
 * Public local text-generation API. Native work runs only on the generation scope's owner thread.
 */
public interface TextGenerationModel extends AutoCloseable {
  /**
   * Generates tokens and synchronously sends token events followed by one terminal event to {@code
   * listener}. If the listener throws, generation aborts, its native scopes are closed, and no
   * terminal callback is promised because the listener has already declined delivery.
   */
  GenerationResult generate(GenerationRequest request, Consumer<GenerationEvent> listener);

  /** Closes only resources owned directly by this model. Scope-taking loaders own none. */
  @Override
  default void close() {}
}
