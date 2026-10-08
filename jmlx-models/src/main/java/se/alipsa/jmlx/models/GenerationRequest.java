package se.alipsa.jmlx.models;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import se.alipsa.jmlx.tokenizer.ChatTemplateOptions;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.vision.RgbImage;

/**
 * A pretokenized, raw-text, or rendered-chat generation request with an explicit prompt policy.
 *
 * <p>Requests start image-free. Vision-language callers attach pixels with {@link
 * #withImages(java.util.List)}; the images are positional placeholders paired in order of
 * appearance with the {@code {"type": "image"}} parts of structured chat content (see {@link
 * HfTokenizer#renderChat}). Pixels never come from the chat messages themselves.
 */
public final class GenerationRequest {
  private final int[] promptTokenIds;
  private final GenerationConfig config;
  private final CancellationToken cancellationToken;
  private final PromptSpecialTokens promptSpecialTokens;
  private final HfTokenizer tokenizer;
  private final GenerationCachePolicy cachePolicy;
  private final List<RgbImage> images;

  /** Creates a request from already-rendered prompt token IDs. */
  public GenerationRequest(
      int[] promptTokenIds, GenerationConfig config, CancellationToken cancellationToken) {
    this(
        promptTokenIds,
        config,
        cancellationToken,
        PromptSpecialTokens.PRETOKENIZED,
        null,
        GenerationCachePolicy.full(),
        List.of());
  }

  private GenerationRequest(
      int[] promptTokenIds,
      GenerationConfig config,
      CancellationToken cancellationToken,
      PromptSpecialTokens promptSpecialTokens,
      HfTokenizer tokenizer,
      GenerationCachePolicy cachePolicy,
      List<RgbImage> images) {
    this.promptTokenIds =
        Arrays.copyOf(
            Objects.requireNonNull(promptTokenIds, "promptTokenIds"), promptTokenIds.length);
    if (this.promptTokenIds.length == 0) {
      throw new IllegalArgumentException("promptTokenIds must not be empty");
    }
    this.config = Objects.requireNonNull(config, "config");
    this.cancellationToken = Objects.requireNonNull(cancellationToken, "cancellationToken");
    this.promptSpecialTokens = Objects.requireNonNull(promptSpecialTokens, "promptSpecialTokens");
    this.tokenizer = tokenizer;
    this.cachePolicy = Objects.requireNonNull(cachePolicy, "cachePolicy");
    this.images = List.copyOf(Objects.requireNonNull(images, "images"));
    for (int i = 0; i < this.images.size(); i++) {
      Objects.requireNonNull(this.images.get(i), "images must not contain null");
    }
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
   * <p>Each message's {@code content} is plain text or a structured list of content parts: {@code
   * {"type": "text", "text": ...}} or {@code {"type": "image"}}. Parts keep their order, and image
   * parts are placeholders only — this method renders them to the template's image marker without
   * fetching any URL or path. Pixels belong on the returned request via {@link
   * #withImages(java.util.List)}, paired with the image parts in order of appearance.
   *
   * @param tokenizer tokenizer and template bundle paired with the checkpoint
   * @param messages ordered role/content messages; content is text or structured content parts
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
        tokenizer,
        GenerationCachePolicy.full(),
        List.of());
  }

  /**
   * Returns a copy of this request that carries the given images, in order of appearance. Every
   * other field — prompt, config, cancellation, prompt special tokens, tokenizer and cache policy —
   * is retained.
   *
   * <p>The list is copied and {@link RgbImage} is itself an immutable value (its constructor copies
   * the pixel buffer), so later mutation of the caller's list does not affect the returned request.
   * The list and its elements must not be null. Text-only models reject non-empty images at
   * generation time; see {@link ModelMetadata#inputModalities()}.
   *
   * @param images pixel payloads for the request's image content parts, in order of appearance
   * @return image-carrying copy of this request
   */
  public GenerationRequest withImages(List<RgbImage> images) {
    return new GenerationRequest(
        promptTokenIds,
        config,
        cancellationToken,
        promptSpecialTokens,
        tokenizer,
        cachePolicy,
        images);
  }

  /** Returns a copy of this request with an explicit cache policy. */
  public GenerationRequest withCachePolicy(GenerationCachePolicy policy) {
    return new GenerationRequest(
        promptTokenIds,
        config,
        cancellationToken,
        promptSpecialTokens,
        tokenizer,
        Objects.requireNonNull(policy, "policy"),
        images);
  }

  /** Returns this request's cache retention policy. */
  public GenerationCachePolicy cachePolicy() {
    return cachePolicy;
  }

  /**
   * Returns the immutable images this request carries, in order of appearance; empty for text-only
   * requests.
   */
  public List<RgbImage> images() {
    return images;
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
