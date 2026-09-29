package se.alipsa.jmlx.models;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import se.alipsa.jmlx.tokenizer.ChatTemplateOptions;
import se.alipsa.jmlx.tokenizer.HfTokenizer;

/** A pretokenized, raw-text, or rendered-chat generation request with an explicit prompt policy. */
public final class GenerationRequest {
  private final int[] promptTokenIds;
  private final GenerationConfig config;
  private final CancellationToken cancellationToken;
  private final PromptSpecialTokens promptSpecialTokens;
  private final HfTokenizer tokenizer;

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
    this.promptSpecialTokens = PromptSpecialTokens.PRETOKENIZED;
    this.tokenizer = null;
  }

  private GenerationRequest(
      int[] promptTokenIds,
      GenerationConfig config,
      CancellationToken cancellationToken,
      PromptSpecialTokens promptSpecialTokens,
      HfTokenizer tokenizer) {
    this.promptTokenIds = Arrays.copyOf(promptTokenIds, promptTokenIds.length);
    if (this.promptTokenIds.length == 0) {
      throw new IllegalArgumentException("promptTokenIds must not be empty");
    }
    this.config = Objects.requireNonNull(config, "config");
    this.cancellationToken = Objects.requireNonNull(cancellationToken, "cancellationToken");
    this.promptSpecialTokens = Objects.requireNonNull(promptSpecialTokens, "promptSpecialTokens");
    this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
  }

  /**
   * Creates a request by eagerly tokenizing raw prompt text.
   *
   * @param tokenizer tokenizer paired with the checkpoint
   * @param prompt raw prompt text
   * @param specialTokens whether the tokenizer post-processor owns prompt special tokens
   * @param config generation policy
   * @param cancellationToken request cancellation token
   * @return tokenizer-backed request
   */
  public static GenerationRequest text(
      HfTokenizer tokenizer,
      String prompt,
      PromptSpecialTokens specialTokens,
      GenerationConfig config,
      CancellationToken cancellationToken) {
    Objects.requireNonNull(tokenizer, "tokenizer");
    Objects.requireNonNull(prompt, "prompt");
    Objects.requireNonNull(specialTokens, "specialTokens");
    if (specialTokens == PromptSpecialTokens.PRETOKENIZED) {
      throw new IllegalArgumentException("text requests require ADD or OMIT special-token policy");
    }
    List<Integer> ids = tokenizer.encode(prompt, specialTokens == PromptSpecialTokens.ADD);
    return tokenizerBacked(ids, config, cancellationToken, specialTokens, tokenizer);
  }

  /**
   * Creates a request by rendering and tokenizing a configured chat template. Templates own their
   * special markers, so this path always tokenizes with special-token insertion omitted.
   *
   * @param tokenizer tokenizer and template bundle paired with the checkpoint
   * @param messages ordered textual role/content messages
   * @param options chat-template options
   * @param config generation policy
   * @param cancellationToken request cancellation token
   * @return tokenizer-backed request
   */
  public static GenerationRequest chat(
      HfTokenizer tokenizer,
      List<Map<String, Object>> messages,
      ChatTemplateOptions options,
      GenerationConfig config,
      CancellationToken cancellationToken) {
    Objects.requireNonNull(tokenizer, "tokenizer");
    String prompt = tokenizer.renderChat(messages, options);
    List<Integer> ids = tokenizer.encode(prompt, false);
    return tokenizerBacked(ids, config, cancellationToken, PromptSpecialTokens.OMIT, tokenizer);
  }

  private static GenerationRequest tokenizerBacked(
      List<Integer> ids,
      GenerationConfig config,
      CancellationToken cancellationToken,
      PromptSpecialTokens specialTokens,
      HfTokenizer tokenizer) {
    return new GenerationRequest(
        ids.stream().mapToInt(Integer::intValue).toArray(),
        config,
        cancellationToken,
        specialTokens,
        tokenizer);
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

  /** Returns how prompt special tokens were handled. */
  public PromptSpecialTokens promptSpecialTokens() {
    return promptSpecialTokens;
  }

  HfTokenizer tokenizer() {
    return tokenizer;
  }
}
