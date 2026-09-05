package se.alipsa.jmlx.tokenizer;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalInt;
import se.alipsa.jmlx.jinja.Template;

/**
 * A pure-Java Hugging Face tokenizer loaded from a {@code tokenizer.json} file or model directory.
 *
 * <p>The supported component families are ByteLevel/Metaspace BPE, Metaspace Unigram without a
 * Precompiled normalizer, and Bert/WordPiece. Instances are immutable and safe for concurrent
 * encoding, decoding, and chat rendering; per-call alignment and incremental-decode state is never
 * shared.
 */
public final class HfTokenizer {

  private static final List<String> BOS_TOKEN_NAMES =
      List.of("<|begin_of_text|>", "<s>", "<|bos|>");
  private static final List<String> EOS_TOKEN_NAMES =
      List.of("<|eot_id|>", "<|im_end|>", "<|end_of_text|>", "<|endoftext|>", "</s>");

  private final Vocabulary vocabulary;
  private final int baseVocabularyMaxKnownId;
  private final TokenizerRuntime runtime;
  private final TokenizerMetadata metadata;
  private final Map<String, Template> chatTemplates;

  private HfTokenizer(TokenizerDefinition definition) {
    this(definition, TokenizerMetadata.empty(), Map.of());
  }

  private HfTokenizer(
      TokenizerDefinition definition,
      TokenizerMetadata metadata,
      Map<String, Template> chatTemplates) {
    this.runtime = new TokenizerRuntime(definition);
    this.vocabulary = runtime.vocabulary();
    this.baseVocabularyMaxKnownId = runtime.vocabSize() - 1;
    this.metadata = Objects.requireNonNull(metadata, "metadata");
    this.chatTemplates = Map.copyOf(chatTemplates);
  }

  /**
   * Loads a tokenizer from a {@code tokenizer.json} file.
   *
   * @param tokenizerJsonPath tokenizer configuration file
   * @return loaded tokenizer
   */
  public static HfTokenizer fromFile(Path tokenizerJsonPath) {
    Objects.requireNonNull(
        tokenizerJsonPath, "HfTokenizer.fromFile: tokenizerJsonPath must not be null");
    return new HfTokenizer(TokenizerJsonLoader.loadDefinition(tokenizerJsonPath));
  }

  /**
   * Loads {@code tokenizer.json}, tokenizer metadata, and chat templates from a model directory.
   *
   * @param modelDirectory local model directory
   * @return loaded tokenizer bundle
   */
  public static HfTokenizer fromDirectory(Path modelDirectory) {
    Objects.requireNonNull(modelDirectory, "HfTokenizer.fromDirectory: modelDirectory");
    TokenizerDirectoryLoader.Bundle bundle = TokenizerDirectoryLoader.load(modelDirectory);
    return new HfTokenizer(bundle.definition(), bundle.metadata(), bundle.templates());
  }

  /**
   * Returns immutable metadata loaded with this tokenizer.
   *
   * @return tokenizer metadata; empty metadata for {@link #fromFile(Path)}
   */
  public TokenizerMetadata metadata() {
    return metadata;
  }

  /**
   * Renders one configured chat template.
   *
   * @param messages ordered textual role/content messages
   * @param options template name, generation-prompt flag, and extra context
   * @return rendered prompt text
   */
  public String renderChat(List<Map<String, Object>> messages, ChatTemplateOptions options) {
    Objects.requireNonNull(messages, "HfTokenizer.renderChat: messages");
    Objects.requireNonNull(options, "HfTokenizer.renderChat: options");
    List<Map<String, Object>> safeMessages =
        messages.stream().map(HfTokenizer::validatedMessage).toList();
    String name = options.templateName();
    if (name.isEmpty()) {
      if (chatTemplates.containsKey("default")) {
        name = "default";
      } else if (chatTemplates.size() == 1) {
        name = chatTemplates.keySet().iterator().next();
      } else {
        throw new TokenizerException(
            "HfTokenizer.renderChat: template name is required when no default exists");
      }
    }
    Template template = chatTemplates.get(name);
    if (template == null) {
      throw new TokenizerException("HfTokenizer.renderChat: unknown chat template '" + name + "'");
    }
    Map<String, Object> context = new HashMap<>(options.extraContext());
    context.put("messages", safeMessages);
    context.put("add_generation_prompt", options.addGenerationPrompt());
    putToken(context, "bos_token", metadata.bosToken());
    putToken(context, "eos_token", metadata.eosToken());
    putToken(context, "pad_token", metadata.padToken());
    putToken(context, "unk_token", metadata.unknownToken());
    putToken(context, "sep_token", metadata.separatorToken());
    putToken(context, "cls_token", metadata.classificationToken());
    putToken(context, "mask_token", metadata.maskToken());
    return ChatTemplateRenderer.render(template, context);
  }

  private static Map<String, Object> validatedMessage(Map<String, Object> message) {
    Objects.requireNonNull(message, "HfTokenizer.renderChat: message");
    Object role = message.get("role");
    Object content = message.get("content");
    if (!(role instanceof String roleText) || roleText.isBlank()) {
      throw new TokenizerException("HfTokenizer.renderChat: message role must be non-empty text");
    }
    if (!(content instanceof String)) {
      throw new TokenizerException("HfTokenizer.renderChat: message content must be text");
    }
    return Map.copyOf(message);
  }

  private static void putToken(
      Map<String, Object> context, String key, java.util.Optional<String> token) {
    token.ifPresent(value -> context.put(key, value));
  }

  /**
   * Finds a BOS id by supported Qwen/Llama naming conventions when the tokenizer marks it special.
   * Prefer {@link #bosTokenId(String)} when {@code tokenizer_config.json} supplies the BOS token.
   *
   * @return detected BOS id, if present
   */
  public OptionalInt bosTokenId() {
    return fallbackSpecialTokenId(BOS_TOKEN_NAMES);
  }

  /**
   * Resolves a caller-supplied BOS token, typically from {@code tokenizer_config.json}. {@code
   * null} returns empty; an unknown non-null token throws.
   *
   * @param bosToken configured BOS token
   * @return empty if {@code bosToken} is null; otherwise its resolved id
   */
  public OptionalInt bosTokenId(String bosToken) {
    return configuredTokenId(bosToken);
  }

  /**
   * Finds an EOS id by supported Qwen/Llama naming conventions when the tokenizer marks it special.
   * Prefer {@link #eosTokenId(String)} when {@code tokenizer_config.json} supplies the EOS token.
   *
   * @return detected EOS id, if present
   */
  public OptionalInt eosTokenId() {
    return fallbackSpecialTokenId(EOS_TOKEN_NAMES);
  }

  /**
   * Resolves a caller-supplied EOS token, typically from {@code tokenizer_config.json}. {@code
   * null} returns empty; an unknown non-null token throws.
   *
   * @param eosToken configured EOS token
   * @return empty if {@code eosToken} is null; otherwise its resolved id
   */
  public OptionalInt eosTokenId(String eosToken) {
    return configuredTokenId(eosToken);
  }

  /**
   * Resolves all caller-supplied EOS tokens, preserving order. Null and unknown entries throw. Use
   * this for checkpoints such as Llama 3.1 that declare multiple generation terminators.
   *
   * @param eosTokens configured EOS tokens
   * @return corresponding token ids
   */
  public List<Integer> eosTokenIds(List<String> eosTokens) {
    Objects.requireNonNull(eosTokens, "HfTokenizer.eosTokenIds: eosTokens must not be null");
    List<Integer> ids = new ArrayList<>(eosTokens.size());
    for (String eosToken : eosTokens) {
      Objects.requireNonNull(eosToken, "HfTokenizer.eosTokenIds: EOS tokens must not contain null");
      ids.add(configuredTokenId(eosToken).getAsInt());
    }
    return List.copyOf(ids);
  }

  /**
   * Returns the tokenizer's contiguous id range. This is not the model's output-head size; obtain
   * that from the checkpoint's {@code config.json}.
   *
   * @return one greater than the largest known token id
   */
  public int vocabSize() {
    return baseVocabularyMaxKnownId + 1;
  }

  private OptionalInt fallbackSpecialTokenId(List<String> names) {
    for (String name : names) {
      if (vocabulary.hasToken(name)) {
        int id = vocabulary.idOf(name);
        if (vocabulary.isSpecial(id)) {
          return OptionalInt.of(id);
        }
      }
    }
    return OptionalInt.empty();
  }

  private OptionalInt configuredTokenId(String token) {
    if (token == null) {
      return OptionalInt.empty();
    }
    if (!vocabulary.hasToken(token)) {
      throw new TokenizerException(
          "HfTokenizer: no vocabulary entry for configured special token '" + token + "'");
    }
    return OptionalInt.of(vocabulary.idOf(token));
  }

  /**
   * Encodes {@code text} into token ids.
   *
   * @param text text to encode
   * @param addSpecialTokens whether to apply post-processor special tokens
   * @return token ids
   */
  public List<Integer> encode(String text, boolean addSpecialTokens) {
    Objects.requireNonNull(text, "HfTokenizer.encode: text must not be null");
    return runtime.encode(text, EncodingOptions.unbounded(addSpecialTokens)).ids();
  }

  /**
   * Encodes text with explicit truncation, padding, and special-token options.
   *
   * @param text input text
   * @param options explicit encoding options
   * @return aligned encoding columns
   */
  public TokenizerEncoding encode(String text, EncodingOptions options) {
    Objects.requireNonNull(text, "HfTokenizer.encode: text must not be null");
    Objects.requireNonNull(options, "HfTokenizer.encode: options must not be null");
    return runtime.encode(text, options);
  }

  /**
   * Encodes text with {@code tokenizer.json}'s configured defaults and special tokens.
   *
   * @param text input text
   * @return aligned encoding columns
   */
  public TokenizerEncoding encodeWithDefaults(String text) {
    return encodeWithDefaults(text, true);
  }

  /**
   * Encodes text with {@code tokenizer.json}'s configured truncation and padding defaults.
   *
   * @param text input text
   * @param addSpecialTokens whether to apply post-processor special tokens
   * @return aligned encoding columns
   */
  public TokenizerEncoding encodeWithDefaults(String text, boolean addSpecialTokens) {
    Objects.requireNonNull(text, "HfTokenizer.encodeWithDefaults: text must not be null");
    return runtime.encode(text, runtime.configuredDefaults(addSpecialTokens));
  }

  /**
   * Creates a request-local generated-token decoder.
   *
   * @param skipSpecialTokens whether vocabulary-marked special tokens produce empty deltas
   * @return new independent decoder state
   */
  public IncrementalTokenDecoder newIncrementalDecoder(boolean skipSpecialTokens) {
    return runtime.newIncrementalDecoder(skipSpecialTokens);
  }

  /**
   * Decodes token ids back into text. An id above this tokenizer's own known-vocabulary boundary --
   * computed from {@code model.vocab} and {@code added_tokens} only, deliberately excluding any
   * sparse {@code TemplateProcessing}-only registration (see the constructor and {@link
   * Vocabulary#maxKnownId()}) -- is skipped rather than aborting the whole decode: e.g. Qwen2.5's
   * {@code config.json} {@code vocab_size} (152064) exceeds its tokenizer vocab (151665), so a
   * sampled logit can legitimately fall in that gap. An id within that boundary that still has no
   * entry is a real hole (the wrong tokenizer for the checkpoint, a mis-parsed {@code
   * added_tokens}, a {@link Vocabulary} bug) and still throws, so it can't be mistaken for the
   * above-vocab case. Described in prose rather than {@code {@link #baseVocabularyMaxKnownId}}: a
   * link to a private field does not resolve in generated public javadoc (PR #14 review round 7,
   * finding 8).
   *
   * @param ids token ids to decode
   * @param skipSpecialTokens whether to omit special tokens
   * @return decoded text
   */
  public String decode(List<Integer> ids, boolean skipSpecialTokens) {
    Objects.requireNonNull(ids, "HfTokenizer.decode: ids must not be null");
    return runtime.decode(ids, skipSpecialTokens);
  }
}
