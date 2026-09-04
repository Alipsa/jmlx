package se.alipsa.jmlx.models;

import java.util.function.Consumer;

/**
 * Public local text-generation API. Native work runs only on the generation scope's owner thread.
 */
public interface TextGenerationModel extends AutoCloseable {
  /** Generates tokens and synchronously sends token and one terminal event to {@code listener}. */
  GenerationResult generate(GenerationRequest request, Consumer<GenerationEvent> listener);

  /** Closes only resources owned directly by this model. Scope-taking loaders own none. */
  @Override
  default void close() {}
}
