package se.alipsa.jmlx.models;

import java.util.Arrays;
import java.util.Objects;

/** A tokenized generation request; prompt rendering and special-token policy stay explicit. */
public final class GenerationRequest {
  private final int[] promptTokenIds;
  private final GenerationConfig config;
  private final CancellationToken cancellationToken;

  /** Creates a request from already-rendered prompt token IDs. */
  public GenerationRequest(
      int[] promptTokenIds, GenerationConfig config, CancellationToken cancellationToken) {
    this.promptTokenIds =
        Arrays.copyOf(
            Objects.requireNonNull(promptTokenIds, "promptTokenIds"), promptTokenIds.length);
    if (this.promptTokenIds.length == 0) {
      throw new IllegalArgumentException("promptTokenIds must not be empty");
    }
    this.config = Objects.requireNonNull(config, "config");
    this.cancellationToken = Objects.requireNonNull(cancellationToken, "cancellationToken");
  }

  /** Returns a defensive copy of the prompt IDs. */
  public int[] promptTokenIds() {
    return Arrays.copyOf(promptTokenIds, promptTokenIds.length);
  }

  /** Returns the immutable generation policy. */
  public GenerationConfig config() {
    return config;
  }

  /** Returns the token polled by the generation-scope owner. */
  public CancellationToken cancellationToken() {
    return cancellationToken;
  }
}
