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
 *
 * <p>Tokenizer-backed requests (the {@link #text} and {@link #chat} factories) also retain the
 * source text their IDs were encoded from — {@link #promptText()}. Pre-tokenized requests carry
 * {@code null}. Vision models whose prompt expansion operates on text rather than on a special
 * image token (Phi-3.5-Vision) require a tokenizer-backed request.
 */
public final class GenerationRequest {
  private final int[] promptTokenIds;
  private final GenerationConfig config;
  private final CancellationToken cancellationToken;
  private final PromptSpecialTokens promptSpecialTokens;
  private final HfTokenizer tokenizer;
  private final GenerationCachePolicy cachePolicy;
  private final List<RgbImage> images;
  private final String promptText;
  private final List<Map<String, Object>> chatMessages;
  private final ChatTemplateOptions chatOptions;

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
        List.of(),
        null,
        null,
        null);
  }

  private GenerationRequest(
      int[] promptTokenIds,
      GenerationConfig config,
      CancellationToken cancellationToken,
      PromptSpecialTokens promptSpecialTokens,
      HfTokenizer tokenizer,
      GenerationCachePolicy cachePolicy,
      List<RgbImage> images,
      String promptText,
      List<Map<String, Object>> chatMessages,
      ChatTemplateOptions chatOptions) {
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
    this.promptText = promptText;
    this.chatMessages = chatMessages;
    this.chatOptions = chatOptions;
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
    return tokenizerBacked(
        ids, config, cancellationToken, specialTokens, tokenizer, prompt, null, null);
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
   * <p>The retained messages and options are the request's snapshot, taken when the request is
   * built: a model whose prompt expansion re-renders them (Phi-3.5-Vision) must expand exactly what
   * {@link #promptTokenIds()} was encoded from, and a caller mutation after the build must not
   * change that. The deep copy therefore happens eagerly here — deferring it to the model's first
   * read would capture mutations made between request creation and generation and could silently
   * diverge the re-expansion from the retained IDs. It is paid by every chat request, including
   * ones a text-only model consumes; that is the deliberate cost of the build-time snapshot.
   *
   * <p>Phi-3.5-Vision re-flattens the messages whenever any one carries structured content, and on
   * that path a literal image marker ({@code <|image_i|>}) inside a plain-string message is
   * rejected: image markers are produced by image parts only. The same literal text is accepted
   * when every message is a plain string, since the rendered text is expanded as-is (the
   * marker-in-text prompt API); the rule is per path, and the two paths differ only by the content
   * shape of the other messages (see {@code Phi3VModel}).
   *
   * @param tokenizer tokenizer and template bundle paired with the checkpoint
   * @param messages ordered role/content messages; content is text or structured content parts
   * @param options chat-template options
   * @param config generation policy
   * @param cancellationToken request cancellation token
   * @return tokenizer-backed request retaining the messages and options, so a text-expanding vision
   *     model (Phi-3.5-Vision) can re-flatten structured image parts that the rendered prompt text
   *     cannot carry
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
    return tokenizerBacked(
        ids,
        config,
        cancellationToken,
        PromptSpecialTokens.OMIT,
        tokenizer,
        prompt,
        deepCopyMessages(messages),
        Objects.requireNonNull(options, "options"));
  }

  /**
   * Deep-copies the chat messages so a later caller mutation cannot change what the model expands:
   * like the images (whose pixel buffers {@link RgbImage} copies), the request must own its source
   * text. Delegates to {@link ChatTemplateOptions#copyContextValue(Object)}, the one shared
   * snapshot copy that the options' own extraContext uses, so the two snapshots cannot drift apart:
   * insertion-order unmodifiable wrappers (null values and non-String nested keys survive exactly
   * as they would the render), unmodifiable list copies of arrays, shared scalars, one shared copy
   * for a value referenced several times, and a cycle rejected with a clear error.
   */
  private static List<Map<String, Object>> deepCopyMessages(List<Map<String, Object>> messages) {
    return ChatTemplateOptions.copyContextValue(messages);
  }

  private static GenerationRequest tokenizerBacked(
      List<Integer> ids,
      GenerationConfig config,
      CancellationToken cancellationToken,
      PromptSpecialTokens specialTokens,
      HfTokenizer tokenizer,
      String promptText,
      List<Map<String, Object>> chatMessages,
      ChatTemplateOptions chatOptions) {
    return new GenerationRequest(
        ids.stream().mapToInt(Integer::intValue).toArray(),
        config,
        cancellationToken,
        specialTokens,
        tokenizer,
        GenerationCachePolicy.full(),
        List.of(),
        promptText,
        chatMessages,
        chatOptions);
  }

  /**
   * Returns a copy of this request that carries the given images, in order of appearance. Every
   * other field — prompt, prompt text, config, cancellation, prompt special tokens, tokenizer and
   * cache policy — is retained.
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
        images,
        promptText,
        chatMessages,
        chatOptions);
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
        images,
        promptText,
        chatMessages,
        chatOptions);
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

  /**
   * Returns the source text this request's IDs were encoded from — the prompt for a {@link #text}
   * request, the rendered chat text for a {@link #chat} request — or {@code null} for a
   * pre-tokenized request. Text is the ground truth for models whose prompt expansion operates on
   * text (Phi-3.5-Vision's image markers are literal prompt text, not special tokens, and the
   * tokenizer's decode is not the inverse of its encode at special-token boundaries, so the text
   * cannot be recovered from the IDs); such models reject pre-tokenized requests.
   */
  public String promptText() {
    return promptText;
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

  /**
   * Returns the chat messages this request was rendered from ({@link #chat}), or {@code null} for
   * {@link #text} and pre-tokenized requests. A model whose prompt expansion operates on text
   * rather than on a special image token (Phi-3.5-Vision) consults these when the rendered prompt
   * text cannot carry the image structure: the artifact's chat template string-concatenates {@code
   * message['content']}, so structured text/image content parts need the messages re-flattened to
   * {@code <|image_i|>} markers before rendering; plain-string content renders as-is and needs only
   * {@link #promptText()}.
   */
  List<Map<String, Object>> chatMessages() {
    return chatMessages;
  }

  /** Returns the chat-template options {@link #chat} rendered with, or {@code null} elsewhere. */
  ChatTemplateOptions chatOptions() {
    return chatOptions;
  }
}
