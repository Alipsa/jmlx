package se.alipsa.jmlx.models;

import java.util.function.Consumer;

/**
 * Public local text-generation API. Native work runs only on the generation scope's owner thread.
 */
public interface TextGenerationModel extends AutoCloseable {
  /**
   * Generates tokens and synchronously sends token events followed by one terminal event to {@code
   * listener}. EOS has precedence when a token belongs to both EOS and stop sets; neither
   * terminating token is sent as a token event. If a token listener throws, generation aborts and
   * its native scopes are closed. A terminal-listener exception is ignored because generation has
   * already completed and its result remains available.
   */
  GenerationResult generate(GenerationRequest request, Consumer<GenerationEvent> listener);

  /** Closes only resources owned directly by this model. Scope-taking loaders own none. */
  @Override
  default void close() {}
}
