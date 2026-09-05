package se.alipsa.jmlx.tokenizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/** Parses a {@code tokenizer.json} file into a {@link TokenizerJson}. */
public final class TokenizerJsonLoader {

  private static final ObjectMapper MAPPER = new ObjectMapper();

  private TokenizerJsonLoader() {}

  /**
   * Loads and parses {@code path} as a byte-level-BPE {@code tokenizer.json}.
   *
   * @param path tokenizer configuration file
   * @return parsed tokenizer configuration
   */
  public static TokenizerJson load(Path path) {
    Objects.requireNonNull(path, "TokenizerJsonLoader.load: path must not be null");
    try {
      JsonNode root = MAPPER.readTree(Files.newInputStream(path));
      requireByteLevelDecoder(root.path("decoder"));
      NormalizerKind normalizer = parseNormalizer(root.path("normalizer"));
      return new TokenizerJson(
          normalizer,
          parsePreTokenizer(root.get("pre_tokenizer")),
          parsePostProcessor(root.path("post_processor")),
          parseModel(root.path("model")),
          parseAddedTokens(root.path("added_tokens"), normalizer));
    } catch (IOException | JacksonException e) {
      throw new TokenizerException("TokenizerJsonLoader.load: failed to parse " + path, e);
    }
  }

  static TokenizerDefinition loadDefinition(Path path) {
    Objects.requireNonNull(path, "TokenizerJsonLoader.loadDefinition: path must not be null");
    try {
      JsonNode root = MAPPER.readTree(Files.newInputStream(path));
      JsonNode normalizer = nullableComponent(root, "normalizer");
      JsonNode preTokenizer = requiredComponent(root, "pre_tokenizer");
      JsonNode decoder = requiredComponent(root, "decoder");
      validateNormalizer(normalizer, "normalizer");
      validatePreTokenizer(preTokenizer, "pre_tokenizer");
      validateDecoder(decoder, "decoder");
      TokenizerDefinition.Model model = parseFlexibleModel(root.path("model"));
      List<AddedToken> addedTokens = parseFlexibleAddedTokens(root.path("added_tokens"));
      return new TokenizerDefinition(
          normalizer,
          preTokenizer,
          parseFlexiblePostProcessor(root.path("post_processor")),
          model,
          decoder,
          addedTokens,
          parseConfiguredDefaults(root, configuredVocabulary(model.vocab(), addedTokens)));
    } catch (IOException | JacksonException e) {
      throw new TokenizerException("TokenizerJsonLoader: failed to parse " + path, e);
    }
  }

  private static JsonNode nullableComponent(JsonNode root, String field) {
    JsonNode node = root.get(field);
    return node == null || node.isNull() ? null : node;
  }

  private static JsonNode requiredComponent(JsonNode root, String field) {
    JsonNode node = root.get(field);
    if (node == null || node.isNull() || !node.isObject()) {
      throw new TokenizerException("TokenizerJsonLoader: " + field + " must be an object");
    }
    return node;
  }

  private static String componentType(JsonNode node, String path) {
    JsonNode type = node.path("type");
    if (!type.isString() || type.asString().isEmpty()) {
      throw new TokenizerException("TokenizerJsonLoader: " + path + ".type must be a string");
    }
    return type.asString();
  }

  private static void validateNormalizer(JsonNode node, String path) {
    if (node == null) {
      return;
    }
    String type = componentType(node, path);
    if ("Precompiled".equals(type)) {
      throw new TokenizerException(
          "TokenizerJsonLoader: unsupported " + path + ".type 'Precompiled'");
    }
    if ("Sequence".equals(type)) {
      if (!node.path("normalizers").isArray()) {
        throw new TokenizerException(
            "TokenizerJsonLoader: " + path + ".normalizers must be an array");
      }
      int index = 0;
      for (JsonNode child : node.path("normalizers")) {
        validateNormalizer(child, path + ".normalizers[" + index++ + "]");
      }
      return;
    }
    if (!List.of(
            "NFC",
            "NFD",
            "NFKC",
            "NFKD",
            "Lowercase",
            "StripAccents",
            "Replace",
            "Strip",
            "Prepend",
            "BertNormalizer")
        .contains(type)) {
      throw new TokenizerException(
          "TokenizerJsonLoader: unsupported " + path + ".type '" + type + "'");
    }
    switch (type) {
      case "Replace" -> {
        validatePattern(node.path("pattern"), path + ".pattern");
        optionalString(node, "content", "", path);
      }
      case "Strip" -> {
        optionalBoolean(node, "strip_left", true, path);
        optionalBoolean(node, "strip_right", true, path);
      }
      case "Prepend" -> optionalString(node, "prepend", "", path);
      case "BertNormalizer" -> {
        optionalBoolean(node, "clean_text", true, path);
        optionalBoolean(node, "handle_chinese_chars", true, path);
        optionalBoolean(node, "lowercase", true, path);
        JsonNode accents = node.path("strip_accents");
        if (!accents.isMissingNode() && !accents.isNull() && !accents.isBoolean()) {
          throw new TokenizerException(
              "TokenizerJsonLoader: " + path + ".strip_accents must be boolean or null");
        }
      }
      default -> {
        // The Unicode-form, lowercase, and accent steps have no behavioral fields.
      }
    }
  }

  private static void validatePreTokenizer(JsonNode node, String path) {
    String type = componentType(node, path);
    if ("Sequence".equals(type)) {
      if (!node.path("pretokenizers").isArray()) {
        throw new TokenizerException(
            "TokenizerJsonLoader: " + path + ".pretokenizers must be an array");
      }
      int index = 0;
      for (JsonNode child : node.path("pretokenizers")) {
        validatePreTokenizer(child, path + ".pretokenizers[" + index++ + "]");
      }
      return;
    }
    if (!List.of(
            "ByteLevel", "Metaspace", "Whitespace", "WhitespaceSplit", "BertPreTokenizer", "Split")
        .contains(type)) {
      throw new TokenizerException(
          "TokenizerJsonLoader: unsupported " + path + ".type '" + type + "'");
    }
    switch (type) {
      case "ByteLevel" -> {
        optionalBoolean(node, "add_prefix_space", false, path);
        optionalBoolean(node, "trim_offsets", true, path);
        optionalBoolean(node, "use_regex", true, path);
      }
      case "Metaspace" -> {
        requireSingleScalar(optionalString(node, "replacement", "▁", path), path + ".replacement");
        validatePrependScheme(node, path);
        optionalBoolean(node, "split", true, path);
      }
      case "Split" -> {
        validatePattern(node.path("pattern"), path + ".pattern");
        String behavior = optionalString(node, "behavior", "Removed", path);
        if (!List.of("Removed", "Isolated", "MergedWithPrevious", "Contiguous")
            .contains(behavior)) {
          throw new TokenizerException(
              "TokenizerJsonLoader: " + path + ".behavior is unsupported: " + behavior);
        }
        optionalBoolean(node, "invert", false, path);
      }
      default -> {
        // Whitespace and BertPreTokenizer have no behavioral fields.
      }
    }
  }

  private static void validateDecoder(JsonNode node, String path) {
    String type = componentType(node, path);
    if ("Sequence".equals(type)) {
      if (!node.path("decoders").isArray()) {
        throw new TokenizerException("TokenizerJsonLoader: " + path + ".decoders must be an array");
      }
      int index = 0;
      for (JsonNode child : node.path("decoders")) {
        validateDecoder(child, path + ".decoders[" + index++ + "]");
      }
      return;
    }
    if (!List.of("ByteLevel", "Metaspace", "WordPiece", "Replace", "Strip", "ByteFallback", "Fuse")
        .contains(type)) {
      throw new TokenizerException(
          "TokenizerJsonLoader: unsupported " + path + ".type '" + type + "'");
    }
    switch (type) {
      case "Metaspace" -> {
        requireSingleScalar(optionalString(node, "replacement", "▁", path), path + ".replacement");
        validatePrependScheme(node, path);
      }
      case "WordPiece" -> {
        optionalString(node, "prefix", "##", path);
        optionalBoolean(node, "cleanup", true, path);
      }
      case "Replace" -> {
        validatePattern(node.path("pattern"), path + ".pattern");
        optionalString(node, "content", "", path);
      }
      case "Strip" -> {
        requireSingleScalar(optionalString(node, "content", " ", path), path + ".content");
        requireNonNegativeInt(node, "start", path);
        requireNonNegativeInt(node, "stop", path);
      }
      default -> {
        // ByteLevel, ByteFallback, and Fuse have no behavioral fields used by decoding.
      }
    }
  }

  private static void validatePattern(JsonNode pattern, String path) {
    if (!pattern.isObject()
        || (pattern.path("String").isMissingNode() == pattern.path("Regex").isMissingNode())
        || (!pattern.path("String").isMissingNode() && !pattern.path("String").isString())
        || (!pattern.path("Regex").isMissingNode() && !pattern.path("Regex").isString())) {
      throw new TokenizerException(
          "TokenizerJsonLoader: " + path + " must contain exactly one String or Regex value");
    }
  }

  private static void validatePrependScheme(JsonNode node, String path) {
    String scheme = optionalString(node, "prepend_scheme", "always", path).toLowerCase();
    if (!List.of("always", "first", "never").contains(scheme)) {
      throw new TokenizerException(
          "TokenizerJsonLoader: " + path + ".prepend_scheme is unsupported: " + scheme);
    }
  }

  private static void requireSingleScalar(String value, String path) {
    if (value.codePointCount(0, value.length()) != 1) {
      throw new TokenizerException(
          "TokenizerJsonLoader: " + path + " must contain one Unicode scalar");
    }
  }

  private static void requireNonNegativeInt(JsonNode node, String field, String path) {
    JsonNode value = node.path(field);
    if (!value.isIntegralNumber() || value.intValue() < 0) {
      throw new TokenizerException(
          "TokenizerJsonLoader: " + path + "." + field + " must be a non-negative integer");
    }
  }

  private static TokenizerDefinition.Model parseFlexibleModel(JsonNode node) {
    String type = componentType(node, "model");
    return switch (type) {
      case "BPE" -> parseFlexibleBpe(node);
      case "Unigram" -> parseUnigram(node);
      case "WordPiece" -> parseWordPiece(node);
      default ->
          throw new TokenizerException(
              "TokenizerJsonLoader: unsupported model.type '" + type + "'");
    };
  }

  private static TokenizerDefinition.Bpe parseFlexibleBpe(JsonNode node) {
    Map<String, Integer> vocab = parseObjectVocabulary(node.path("vocab"), "model.vocab");
    final Map<String, Integer> mergeRanks = parseMergeRanks(node.path("merges"));
    JsonNode dropout = node.path("dropout");
    if (!dropout.isMissingNode()
        && !dropout.isNull()
        && !(dropout.isNumber() && dropout.doubleValue() == 0.0)) {
      throw new TokenizerException(
          "TokenizerJsonLoader: model.dropout must be null, absent, or zero");
    }
    boolean byteFallback = optionalBoolean(node, "byte_fallback", false, "model");
    boolean fuseUnknown = optionalBoolean(node, "fuse_unk", false, "model");
    boolean ignoreMerges = optionalBoolean(node, "ignore_merges", false, "model");
    String unknown = optionalString(node, "unk_token", null, "model");
    String prefix = optionalString(node, "continuing_subword_prefix", "", "model");
    String suffix = optionalString(node, "end_of_word_suffix", "", "model");
    if (unknown != null && !vocab.containsKey(unknown)) {
      throw new TokenizerException("TokenizerJsonLoader: model.unk_token is not in model.vocab");
    }
    if (byteFallback) {
      validateByteFallbackTokens(vocab);
    }
    return new TokenizerDefinition.Bpe(
        vocab, mergeRanks, unknown, fuseUnknown, byteFallback, prefix, suffix, ignoreMerges);
  }

  private static TokenizerDefinition.Unigram parseUnigram(JsonNode node) {
    JsonNode values = node.path("vocab");
    if (!values.isArray() || values.isEmpty()) {
      throw new TokenizerException("TokenizerJsonLoader: model.vocab must be a non-empty array");
    }
    List<String> tokens = new ArrayList<>();
    List<Double> scores = new ArrayList<>();
    Map<String, Integer> vocab = new LinkedHashMap<>();
    for (int index = 0; index < values.size(); index++) {
      JsonNode value = values.get(index);
      if (!value.isArray()
          || value.size() != 2
          || !value.get(0).isString()
          || !value.get(1).isNumber()) {
        throw new TokenizerException(
            "TokenizerJsonLoader: model.vocab[" + index + "] must be [token, score]");
      }
      String token = value.get(0).asString();
      if (vocab.putIfAbsent(token, index) != null) {
        throw new TokenizerException(
            "TokenizerJsonLoader: duplicate Unigram token '" + token + "'");
      }
      tokens.add(token);
      scores.add(value.get(1).doubleValue());
    }
    int unknownId = requiredInt(node, "unk_id", "model");
    if (unknownId < 0 || unknownId >= tokens.size()) {
      throw new TokenizerException("TokenizerJsonLoader: model.unk_id is outside model.vocab");
    }
    boolean byteFallback = optionalBoolean(node, "byte_fallback", false, "model");
    if (byteFallback) {
      validateByteFallbackTokens(vocab);
    }
    return new TokenizerDefinition.Unigram(vocab, tokens, scores, unknownId, byteFallback);
  }

  private static void validateByteFallbackTokens(Map<String, Integer> vocab) {
    Pattern canonical = Pattern.compile("<0x[0-9A-F]{2}>");
    for (String token : vocab.keySet()) {
      if (token.startsWith("<0x") && !canonical.matcher(token).matches()) {
        throw new TokenizerException(
            "TokenizerJsonLoader: malformed byte fallback token '" + token + "'");
      }
    }
  }

  private static TokenizerDefinition.WordPiece parseWordPiece(JsonNode node) {
    Map<String, Integer> vocab = parseObjectVocabulary(node.path("vocab"), "model.vocab");
    String unknown = optionalString(node, "unk_token", null, "model");
    if (unknown == null || !vocab.containsKey(unknown)) {
      throw new TokenizerException(
          "TokenizerJsonLoader: model.unk_token must name a vocabulary token");
    }
    String prefix = optionalString(node, "continuing_subword_prefix", "##", "model");
    int maximum = node.path("max_input_chars_per_word").asInt(100);
    if (maximum <= 0) {
      throw new TokenizerException(
          "TokenizerJsonLoader: model.max_input_chars_per_word must be positive");
    }
    return new TokenizerDefinition.WordPiece(vocab, unknown, prefix, maximum);
  }

  private static Map<String, Integer> parseObjectVocabulary(JsonNode node, String path) {
    if (!node.isObject() || node.isEmpty()) {
      throw new TokenizerException("TokenizerJsonLoader: " + path + " must be a non-empty object");
    }
    Map<String, Integer> vocab = new LinkedHashMap<>();
    Map<Integer, String> byId = new HashMap<>();
    for (Map.Entry<String, JsonNode> entry : node.properties()) {
      if (!entry.getValue().isIntegralNumber()) {
        throw new TokenizerException(
            "TokenizerJsonLoader: " + path + "['" + entry.getKey() + "'] must be an integer");
      }
      int id = entry.getValue().intValue();
      String other = byId.putIfAbsent(id, entry.getKey());
      if (other != null) {
        throw new TokenizerException(
            "TokenizerJsonLoader: " + path + " id " + id + " is duplicated");
      }
      vocab.put(entry.getKey(), id);
    }
    return vocab;
  }

  private static Map<String, Integer> parseMergeRanks(JsonNode node) {
    if (!node.isArray()) {
      throw new TokenizerException("TokenizerJsonLoader: model.merges must be an array");
    }
    Map<String, Integer> ranks = new LinkedHashMap<>();
    for (int index = 0; index < node.size(); index++) {
      JsonNode merge = node.get(index);
      String pair;
      if (merge.isArray() && merge.size() == 2) {
        pair = merge.get(0).asString() + " " + merge.get(1).asString();
      } else if (merge.isString()) {
        pair = merge.asString();
      } else {
        throw new TokenizerException(
            "TokenizerJsonLoader: model.merges[" + index + "] is not a pair");
      }
      ranks.putIfAbsent(pair, index);
    }
    return ranks;
  }

  private static List<AddedToken> parseFlexibleAddedTokens(JsonNode node) {
    List<AddedToken> result = new ArrayList<>();
    if (node.isMissingNode() || node.isNull()) {
      return result;
    }
    if (!node.isArray()) {
      throw new TokenizerException("TokenizerJsonLoader: added_tokens must be an array");
    }
    for (int index = 0; index < node.size(); index++) {
      JsonNode entry = node.get(index);
      requireValidAddedTokenIdentity(entry);
      String path = "added_tokens[" + index + "]";
      boolean special = optionalBoolean(entry, "special", false, path);
      result.add(
          new AddedToken(
              entry.path("id").intValue(),
              entry.path("content").asString(),
              optionalBoolean(entry, "single_word", false, path),
              optionalBoolean(entry, "lstrip", false, path),
              optionalBoolean(entry, "rstrip", false, path),
              optionalBoolean(entry, "normalized", !special, path),
              special));
    }
    requireInternallyConsistentAddedTokens(result);
    return result;
  }

  private static List<PostProcessorStep> parseFlexiblePostProcessor(JsonNode node) {
    List<PostProcessorStep> result = new ArrayList<>();
    if (node.isMissingNode() || node.isNull()) {
      return result;
    }
    parseFlexiblePostProcessor(node, "post_processor", result);
    return result;
  }

  private static void parseFlexiblePostProcessor(
      JsonNode node, String path, List<PostProcessorStep> result) {
    String type = componentType(node, path);
    if ("Sequence".equals(type)) {
      int index = 0;
      for (JsonNode child : node.path("processors")) {
        parseFlexiblePostProcessor(child, path + ".processors[" + index++ + "]", result);
      }
      return;
    }
    if ("ByteLevel".equals(type)) {
      result.add(new ByteLevelStep(optionalBoolean(node, "trim_offsets", true, path)));
      return;
    }
    if ("TemplateProcessing".equals(type)) {
      result.add(parseTemplateProcessing(node));
      return;
    }
    if ("BertProcessing".equals(type)) {
      result.add(
          new BertProcessingStep(
              parseTokenPair(node.path("sep"), path + ".sep"),
              parseTokenPair(node.path("cls"), path + ".cls")));
      return;
    }
    throw new TokenizerException(
        "TokenizerJsonLoader: unsupported " + path + ".type '" + type + "'");
  }

  private static ResolvedToken parseTokenPair(JsonNode node, String path) {
    if (!node.isArray()
        || node.size() != 2
        || !node.get(0).isString()
        || !node.get(1).isIntegralNumber()) {
      throw new TokenizerException("TokenizerJsonLoader: " + path + " must be [token, id]");
    }
    return new ResolvedToken(node.get(0).asString(), node.get(1).intValue());
  }

  private static TemplateProcessingStep parseTemplateProcessing(JsonNode node) {
    List<TemplateItem> single = new ArrayList<>();
    for (JsonNode item : node.path("single")) {
      if (item.has("SpecialToken")) {
        single.add(new SpecialTokenItem(item.path("SpecialToken").path("id").asString()));
      } else if (item.has("Sequence")) {
        single.add(new SequenceItem());
      } else {
        throw new TokenizerException(
            "TokenizerJsonLoader: unrecognized TemplateProcessing item " + item);
      }
    }
    Map<String, SpecialTokenInfo> specialTokens = new LinkedHashMap<>();
    for (Map.Entry<String, JsonNode> entry : node.path("special_tokens").properties()) {
      JsonNode value = entry.getValue();
      List<Integer> ids = new ArrayList<>();
      List<String> tokens = new ArrayList<>();
      value.path("ids").forEach(id -> ids.add(id.intValue()));
      value.path("tokens").forEach(token -> tokens.add(token.asString()));
      specialTokens.put(
          entry.getKey(), new SpecialTokenInfo(value.path("id").asString(), ids, tokens));
    }
    return new TemplateProcessingStep(single, specialTokens);
  }

  private static EncodingOptions parseConfiguredDefaults(
      JsonNode root, Map<String, Integer> vocabulary) {
    Truncation truncation = Truncation.disabled();
    JsonNode truncationNode = root.path("truncation");
    if (!truncationNode.isMissingNode() && !truncationNode.isNull()) {
      String strategy = optionalString(truncationNode, "strategy", "LongestFirst", "truncation");
      if (!"LongestFirst".equals(strategy) && !"OnlyFirst".equals(strategy)) {
        throw new TokenizerException(
            "TokenizerJsonLoader: truncation.strategy is pair-only or unsupported: " + strategy);
      }
      int stride = truncationNode.path("stride").asInt(0);
      if (stride != 0) {
        throw new TokenizerException("TokenizerJsonLoader: truncation.stride must be zero");
      }
      truncation =
          new Truncation(
              requiredInt(truncationNode, "max_length", "truncation"),
              parseDirection(truncationNode, "direction", Direction.RIGHT, "truncation"));
    }
    Padding padding = Padding.disabled();
    JsonNode paddingNode = root.path("padding");
    if (!paddingNode.isMissingNode() && !paddingNode.isNull()) {
      JsonNode strategy = paddingNode.path("strategy");
      JsonNode fixed = strategy.path("Fixed");
      if (!strategy.isObject() || !fixed.isIntegralNumber() || fixed.intValue() <= 0) {
        throw new TokenizerException(
            "TokenizerJsonLoader: padding.strategy must be {\"Fixed\": positive length} "
                + "for single-sequence defaults");
      }
      JsonNode multiple = paddingNode.path("pad_to_multiple_of");
      if (!multiple.isMissingNode() && !multiple.isNull()) {
        throw new TokenizerException(
            "TokenizerJsonLoader: padding.pad_to_multiple_of is unsupported");
      }
      int id = requiredInt(paddingNode, "pad_id", "padding");
      String token = optionalString(paddingNode, "pad_token", null, "padding");
      if (token == null || !vocabulary.containsKey(token) || vocabulary.get(token) != id) {
        throw new TokenizerException(
            "TokenizerJsonLoader: padding token/id must resolve to model.vocab");
      }
      padding =
          new Padding(
              fixed.intValue(),
              parseDirection(paddingNode, "direction", Direction.RIGHT, "padding"),
              id,
              token,
              paddingNode.path("pad_type_id").asInt(0));
    }
    return new EncodingOptions(true, truncation, padding);
  }

  private static Map<String, Integer> configuredVocabulary(
      Map<String, Integer> modelVocabulary, List<AddedToken> addedTokens) {
    Map<String, Integer> result = new LinkedHashMap<>(modelVocabulary);
    for (AddedToken token : addedTokens) {
      result.put(token.content(), token.id());
    }
    return result;
  }

  private static Direction parseDirection(
      JsonNode node, String field, Direction defaultValue, String path) {
    String value = optionalString(node, field, defaultValue.name().toLowerCase(), path);
    return switch (value.toLowerCase()) {
      case "left" -> Direction.LEFT;
      case "right" -> Direction.RIGHT;
      default ->
          throw new TokenizerException(
              "TokenizerJsonLoader: " + path + "." + field + " must be left or right");
    };
  }

  private static boolean optionalBoolean(
      JsonNode node, String field, boolean defaultValue, String path) {
    JsonNode value = node.path(field);
    if (value.isMissingNode() || value.isNull()) {
      return defaultValue;
    }
    if (!value.isBoolean()) {
      throw new TokenizerException(
          "TokenizerJsonLoader: " + path + "." + field + " must be a boolean");
    }
    return value.booleanValue();
  }

  private static String optionalString(
      JsonNode node, String field, String defaultValue, String path) {
    JsonNode value = node.path(field);
    if (value.isMissingNode() || value.isNull()) {
      return defaultValue;
    }
    if (!value.isString()) {
      throw new TokenizerException(
          "TokenizerJsonLoader: " + path + "." + field + " must be a string");
    }
    return value.asString();
  }

  private static int requiredInt(JsonNode node, String field, String path) {
    JsonNode value = node.path(field);
    if (!value.isIntegralNumber()) {
      throw new TokenizerException(
          "TokenizerJsonLoader: " + path + "." + field + " must be an integer");
    }
    return value.intValue();
  }

  /**
   * Requires {@code node.path(field)} to be a genuine JSON boolean when present -- the same
   * principle round 8 finding 3 applied to {@code normalized}, generalized to every other {@code
   * asBoolean(default)} guard in this class that could otherwise fail open. Two distinct failure
   * modes, both verified directly against this port's pinned Jackson version: a "must be false"
   * guard (e.g. {@code byte_fallback}, {@code lstrip}) only rejects an explicit {@code true} --
   * {@code "byte_fallback": "yes"} and {@code "lstrip": "yes"} both loaded cleanly, since neither
   * string parses as the literal boolean {@code true} and {@code asBoolean(false)} silently returns
   * its {@code false} default for the unrecognized string, defeating the very guard checking it. A
   * real config value with no "must be" constraint (e.g. {@code model .ignore_merges}) has no guard
   * to defeat, but the same silent coercion means a malformed, non-boolean value is misinterpreted
   * as its default meaning instead of the file being rejected as malformed -- for {@code
   * ignore_merges} specifically, silently flipping tokenization rather than merely skipping a
   * validation (PR #14 review round 9, finding 3).
   */
  private static void requireBoolean(JsonNode node, String field, String description) {
    JsonNode value = node.path(field);
    if (!value.isMissingNode() && !value.isNull() && !value.isBoolean()) {
      throw new TokenizerException(
          "TokenizerJsonLoader: " + description + " must be a boolean, got " + value);
    }
  }

  private static NormalizerKind parseNormalizer(JsonNode node) {
    if (node.isNull() || node.isMissingNode()) {
      return NormalizerKind.NONE;
    }
    String type = node.path("type").asString("");
    if ("NFC".equals(type)) {
      return NormalizerKind.NFC;
    }
    throw new TokenizerException("TokenizerJsonLoader: unsupported normalizer type '" + type + "'");
  }

  /**
   * Requires {@code decoder.type} to be {@code "ByteLevel"} -- the only shape {@link
   * ByteLevelDecoder} implements. Unlike {@code normalizer}/{@code pre_tokenizer}/{@code
   * post_processor}/{@code model}, this field was previously never read at all: {@link
   * HfTokenizer#decode} unconditionally calls {@link ByteLevelDecoder#decode}, so a file declaring
   * a {@code Sequence}/{@code Replace}/{@code Strip}/{@code ByteFallback}/{@code Metaspace} decoder
   * loaded silently and produced wrong text with no diagnostic (PR #14 review round 3, finding 3).
   * A missing/{@code null} {@code decoder} (a shape HF's own serde really emits, unlike a required
   * field such as {@code pre_tokenizer}) is deliberately rejected too, not defaulted to {@code
   * ByteLevel}: silently guessing the decoder shape would be its own divergence, so this port
   * requires it declared explicitly, with a message that says so rather than reusing the
   * unsupported-type message's misleading {@code ''} (PR #14 review round 4, finding 4).
   *
   * <p>Only {@code type} is validated -- not {@code add_prefix_space}, {@code trim_offsets}, or
   * {@code use_regex}, all three of which a real {@code tokenizer.json} decoder object can also
   * declare. {@code ByteLevel}'s actual {@code decode_chain} (see the {@code
   * huggingface/tokenizers} Rust source) reads none of them: {@code add_prefix_space}/{@code
   * use_regex} only affect that same struct's separate {@code PreTokenizer} impl, and {@code
   * trim_offsets} only affects its {@code PostProcessor} impl's span offsets, which this port
   * doesn't track. Real Llama-3 files declare {@code add_prefix_space: true} and {@code use_regex:
   * true} on the decoder specifically (unlike the pre-tokenizer's own {@code ByteLevel} step, where
   * {@code use_regex} is validated above) -- rejecting those values here would reject real,
   * correctly-loadable files for a divergence that does not actually exist (PR #14 review round 4,
   * finding 5 -- investigated and found not to apply, unlike findings 1/3/4 above).
   *
   * <p>Only checks {@code isMissingNode()}/{@code isNull()}, not a Java {@code null} {@code node}
   * itself: the sole call site passes {@code root.path("decoder")}, and Jackson's {@code path()}
   * (unlike {@code get()}, see {@link #parsePreTokenizer}) never returns Java {@code null} -- only
   * ever a real node or a {@code MissingNode} -- so that branch was dead (PR #14 review round 5,
   * finding 9).
   */
  private static void requireByteLevelDecoder(JsonNode node) {
    if (node.isMissingNode() || node.isNull()) {
      throw new TokenizerException(
          "TokenizerJsonLoader: tokenizer.json has no decoder (only 'ByteLevel' is supported, and"
              + " HfTokenizer#decode always uses ByteLevelDecoder)");
    }
    String type = node.path("type").asString("");
    if (!"ByteLevel".equals(type)) {
      throw new TokenizerException(
          "TokenizerJsonLoader: unsupported decoder type '"
              + type
              + "' (only 'ByteLevel' is supported)");
    }
  }

  /**
   * Requires {@code pre_tokenizer.pretokenizers} to be exactly {@code [Split, ByteLevel]}, in that
   * order, with {@code Split.behavior == "Isolated"}, {@code Split.invert == false}, and {@code
   * ByteLevel.use_regex == false} -- the only shape this port's {@code ByteLevelPreTokenizer}
   * implements (see Global Constraints in {@code req/plans/phase5-m2-plan.md}). A file using any
   * other shape (a different step order, an extra/missing step, {@code invert: true}, a non-
   * "Isolated" behavior, or {@code use_regex: true}) throws here instead of silently tokenizing
   * differently from HF. {@code behavior} is required, not defaulted to {@code "Isolated"}: it
   * isn't optional in HF's own serde, unlike every other field this method checks (PR #14 review
   * round 4, finding 8).
   */
  private static PreTokenizerConfig parsePreTokenizer(JsonNode node) {
    if (node == null || !"Sequence".equals(node.path("type").asString(""))) {
      throw new TokenizerException(
          "TokenizerJsonLoader: expected pre_tokenizer.type == 'Sequence'");
    }
    List<JsonNode> steps = new ArrayList<>();
    node.path("pretokenizers").forEach(steps::add);
    if (steps.size() != 2
        || !"Split".equals(steps.get(0).path("type").asString(""))
        || !"ByteLevel".equals(steps.get(1).path("type").asString(""))) {
      throw new TokenizerException(
          "TokenizerJsonLoader: expected pre_tokenizer.pretokenizers == [Split, ByteLevel], got "
              + steps.stream().map(s -> s.path("type").asString("?")).toList());
    }
    JsonNode splitStep = steps.get(0);
    String behavior = splitStep.path("behavior").asString(null);
    if (!"Isolated".equals(behavior)) {
      throw new TokenizerException(
          "TokenizerJsonLoader: unsupported pre_tokenizer Split behavior '"
              + behavior
              + "' (only 'Isolated' is supported)");
    }
    requireBoolean(splitStep, "invert", "pre_tokenizer Split invert");
    if (splitStep.path("invert").asBoolean(false)) {
      throw new TokenizerException(
          "TokenizerJsonLoader: unsupported pre_tokenizer Split invert=true");
    }
    JsonNode byteLevelStep = steps.get(1);
    requireBoolean(byteLevelStep, "use_regex", "pre_tokenizer ByteLevel use_regex");
    if (byteLevelStep.path("use_regex").asBoolean(false)) {
      throw new TokenizerException(
          "TokenizerJsonLoader: unsupported pre_tokenizer ByteLevel use_regex=true");
    }
    String regex = splitStep.path("pattern").path("Regex").asString(null);
    if (regex == null) {
      throw new TokenizerException(
          "TokenizerJsonLoader: pre_tokenizer Split step has no pattern.Regex");
    }
    requireBoolean(byteLevelStep, "add_prefix_space", "pre_tokenizer ByteLevel add_prefix_space");
    boolean addPrefixSpace = byteLevelStep.path("add_prefix_space").asBoolean(false);
    try {
      // No Pattern.UNICODE_CHARACTER_CLASS: HF compiles this regex with onig (the default
      // "onig" cargo feature), whose \s/\S/\w/\d/\b are ASCII-only, matching plain Java Pattern's
      // own default. The flag would make \s Unicode-aware instead (e.g. matching U+00A0 NBSP),
      // diverging from HF on any input containing Unicode whitespace -- and buys nothing for the
      // Unicode-letter matching the flag was presumably added for, since \p{L} is already
      // Unicode-scoped by definition regardless of this flag (PR #14 review round 4, finding 1).
      return new PreTokenizerConfig(Pattern.compile(regex), addPrefixSpace);
    } catch (PatternSyntaxException e) {
      throw new TokenizerException(
          "TokenizerJsonLoader: invalid pre_tokenizer regex '" + regex + "'", e);
    }
  }

  private static List<PostProcessorStep> parsePostProcessor(JsonNode node) {
    List<PostProcessorStep> steps = new ArrayList<>();
    if (node.isNull() || node.isMissingNode()) {
      return steps;
    }
    String type = node.path("type").asString("");
    if ("Sequence".equals(type)) {
      for (JsonNode step : node.path("processors")) {
        steps.add(parsePostProcessorStep(step));
      }
    } else {
      steps.add(parsePostProcessorStep(node));
    }
    return steps;
  }

  private static PostProcessorStep parsePostProcessorStep(JsonNode node) {
    String type = node.path("type").asString("");
    if ("ByteLevel".equals(type)) {
      return new ByteLevelStep();
    }
    if ("TemplateProcessing".equals(type)) {
      // node.path("pair") is deliberately not parsed: this port has no sentence-pair encoding
      // API (see Global Constraints in req/plans/phase5-m2-plan.md), so there is nothing to
      // apply a pair template to.
      List<TemplateItem> single = new ArrayList<>();
      for (JsonNode item : node.path("single")) {
        if (item.has("SpecialToken")) {
          single.add(new SpecialTokenItem(item.path("SpecialToken").path("id").asString()));
        } else if (item.has("Sequence")) {
          single.add(new SequenceItem());
        } else {
          throw new TokenizerException(
              "TokenizerJsonLoader: unrecognized TemplateProcessing item " + item);
        }
      }
      Map<String, SpecialTokenInfo> specialTokens = new LinkedHashMap<>();
      for (Map.Entry<String, JsonNode> entry : node.path("special_tokens").properties()) {
        JsonNode v = entry.getValue();
        List<Integer> ids = new ArrayList<>();
        v.path("ids")
            .forEach(
                idNode -> {
                  if (idNode.isNull() || !idNode.isIntegralNumber()) {
                    throw new TokenizerException(
                        "TokenizerJsonLoader: TemplateProcessing special token '"
                            + entry.getKey()
                            + "' has a non-integral id: "
                            + idNode);
                  }
                  ids.add(idNode.asInt());
                });
        List<String> tokens = new ArrayList<>();
        v.path("tokens")
            .forEach(
                tokenNode -> {
                  if (tokenNode.isNull() || !tokenNode.isString()) {
                    throw new TokenizerException(
                        "TokenizerJsonLoader: TemplateProcessing special token '"
                            + entry.getKey()
                            + "' has a non-string token: "
                            + tokenNode);
                  }
                  tokens.add(tokenNode.asString());
                });
        // SpecialTokenInfo's own compact constructor enforces the non-empty, equal-length
        // invariant (PR #14 review round 3, finding 1) -- no need to duplicate that check here.
        specialTokens.put(
            entry.getKey(), new SpecialTokenInfo(v.path("id").asString(), ids, tokens));
      }
      return new TemplateProcessingStep(single, specialTokens);
    }
    throw new TokenizerException(
        "TokenizerJsonLoader: unsupported post_processor step type '" + type + "'");
  }

  /**
   * Requires the five {@code model} fields this port has no implementation for -- {@code
   * byte_fallback}, {@code dropout}, {@code unk_token}, {@code continuing_subword_prefix}, {@code
   * end_of_word_suffix} -- to be at their inert default for both target models (verified absent in
   * Qwen2.5's and Llama-3's real {@code tokenizer.json} files -- see this plan's Findings). Each
   * was previously read nowhere: {@code byte_fallback: true} would load silently and only surface
   * as a {@link BpeMerger#merge} exception whose message admits the assumption was never checked; a
   * {@code dropout} or {@code unk_token} would load and encode silently with different ids and no
   * diagnostic at all (PR #14 review round 4, finding 3).
   */
  private static void requireInertModelFields(JsonNode node) {
    requireBoolean(node, "byte_fallback", "model.byte_fallback");
    if (node.path("byte_fallback").asBoolean(false)) {
      throw new TokenizerException(
          "TokenizerJsonLoader: unsupported model.byte_fallback=true (this port assumes false --"
              + " see BpeMerger#merge)");
    }
    // "dropout": 0.0 is semantically identical to null/absent (no dropout applied) and must not
    // be rejected alongside a genuine positive value -- BpeMerger has no dropout behavior to
    // diverge either way at 0.0 (PR #14 review round 5, finding 6). Whitelisting only an exact
    // numeric zero, rather than gating on dropout.asDouble(0.0) > 0.0, matters because asDouble
    // silently returns its default for any node it can't coerce to a number at all -- an object,
    // a string, or an array all produced 0.0 and were wrongly accepted by that check, the same as
    // a genuine "dropout": 0.0 (PR #14 review round 6, finding 3, correcting round 5's own fix).
    JsonNode dropout = node.path("dropout");
    if (!dropout.isNull()
        && !dropout.isMissingNode()
        && !(dropout.isNumber() && dropout.doubleValue() == 0.0)) {
      throw new TokenizerException(
          "TokenizerJsonLoader: unsupported model.dropout " + dropout + " (must be 0/null/absent)");
    }
    JsonNode unkToken = node.path("unk_token");
    if (!unkToken.isNull() && !unkToken.isMissingNode()) {
      throw new TokenizerException(
          "TokenizerJsonLoader: unsupported model.unk_token "
              + unkToken
              + " (must be null/absent)");
    }
    String continuingSubwordPrefix = node.path("continuing_subword_prefix").asString("");
    if (!continuingSubwordPrefix.isEmpty()) {
      throw new TokenizerException(
          "TokenizerJsonLoader: unsupported model.continuing_subword_prefix '"
              + continuingSubwordPrefix
              + "' (must be empty/absent)");
    }
    String endOfWordSuffix = node.path("end_of_word_suffix").asString("");
    if (!endOfWordSuffix.isEmpty()) {
      throw new TokenizerException(
          "TokenizerJsonLoader: unsupported model.end_of_word_suffix '"
              + endOfWordSuffix
              + "' (must be empty/absent)");
    }
  }

  private static BpeModelConfig parseModel(JsonNode node) {
    if (!"BPE".equals(node.path("type").asString(""))) {
      throw new TokenizerException("TokenizerJsonLoader: expected model.type == 'BPE'");
    }
    requireInertModelFields(node);
    Map<String, Integer> vocab = new HashMap<>();
    // Rejects two different token strings sharing one id, not just the reverse (a token string
    // repeated with two different ids, which a plain Map already can't represent): Vocabulary's
    // own idToToken build (modelVocab.forEach((token, id) -> idToToken.put(id, token))) lets
    // whichever token HashMap iterates last silently win that id, while tokenToId keeps both --
    // so encode(firstToken) and decode(thatId) could disagree on which string it means, with
    // neither side aware anything is wrong (PR #14 review round 7, finding 2).
    Map<Integer, String> vocabTokenById = new HashMap<>();
    for (Map.Entry<String, JsonNode> entry : node.path("vocab").properties()) {
      String token = entry.getKey();
      JsonNode idNode = entry.getValue();
      if (idNode.isNull() || !idNode.isIntegralNumber()) {
        throw new TokenizerException(
            "TokenizerJsonLoader: model.vocab['" + token + "'] has no integral id: " + idNode);
      }
      int id = idNode.asInt();
      String existingToken = vocabTokenById.putIfAbsent(id, token);
      if (existingToken != null && !existingToken.equals(token)) {
        throw new TokenizerException(
            "TokenizerJsonLoader: model.vocab has id "
                + id
                + " for both '"
                + existingToken
                + "' and '"
                + token
                + "'");
      }
      vocab.put(token, id);
    }
    if (vocab.isEmpty()) {
      // Mirrors the mergeRank.isEmpty() guard below: an empty vocab makes Vocabulary#maxKnownId
      // -1, so HfTokenizer#decode's above-vocab branch (see its own javadoc) swallows every id and
      // silently returns "" instead of failing loudly (PR #14 review round 3, finding 4).
      throw new TokenizerException("TokenizerJsonLoader: model.vocab is empty");
    }
    Map<String, Integer> mergeRank = new HashMap<>();
    int rank = 0;
    for (JsonNode merge : node.path("merges")) {
      // tokenizers >= 0.20.0 (HF tokenizers PR #909) emits merges as ["l", "o"] pairs; the older
      // serialization emits "l o" as one space-separated string. Both encode the same
      // priority-rank-by-array-index semantics. Re-joining a pair with a plain " " cannot
      // reintroduce ambiguity: byte 0x20 (space) is never itself a printable-range byte (the
      // printable range starts at '!', see ByteLevelCoding), so a literal space can never occur
      // inside a byte-level symbol -- " " is guaranteed to be a safe, unambiguous separator.
      String pair;
      if (merge.isArray()) {
        if (merge.size() != 2) {
          throw new TokenizerException(
              "TokenizerJsonLoader: model.merges entry must be a 2-element array, got " + merge);
        }
        pair = merge.get(0).asString() + " " + merge.get(1).asString();
      } else {
        pair = merge.asString();
      }
      // HF keeps the first occurrence's rank on a duplicate pair; putIfAbsent matches that
      // instead of letting a later occurrence silently downgrade an earlier pair's priority.
      mergeRank.putIfAbsent(pair, rank);
      rank++;
    }
    if (mergeRank.isEmpty()) {
      throw new TokenizerException(
          "TokenizerJsonLoader: model.merges produced an empty merge table -- BpeMerger would"
              + " silently degrade to per-byte tokenization");
    }
    requireBoolean(node, "ignore_merges", "model.ignore_merges");
    boolean ignoreMerges = node.path("ignore_merges").asBoolean(false);
    return new BpeModelConfig(vocab, mergeRank, ignoreMerges);
  }

  /**
   * Requires an {@code added_tokens} entry to have a non-empty, string {@code content} and a
   * present, integral {@code id} -- each otherwise silently coerced or defaulted rather than
   * rejected: {@code entry.path("content").asString()} returns {@code ""} for a missing or {@code
   * null} node (verified directly against this port's pinned Jackson version), and {@link
   * AddedTokenSplitter} compiles every added token's content via {@code Pattern.quote} into one
   * alternation regex -- an empty pattern matches at every position, interleaving the token's id
   * between every character of the input and leaving every character unmerged, while decode still
   * round-trips the original text (a passing round-trip text assertion hiding garbage ids). {@code
   * entry.path("id").asInt()} likewise silently defaults to {@code 0} for a missing {@code id}, and
   * {@link Vocabulary}'s own added-token collision cleanup then vacates whatever real {@code
   * model.vocab} token currently owns id {@code 0}, making it permanently un-encodable with no
   * diagnostic at either site (PR #14 review round 8, finding 1).
   *
   * <p>{@code isString()}/{@code isIntegralNumber()} close a second gap left by the presence-only
   * checks above (PR #14 review round 9, finding 4): a present but non-string {@code content} (e.g.
   * {@code 123}) or non-integral {@code id} (e.g. {@code "3"}, coerced, or {@code 100.9},
   * truncated) both load without error otherwise -- and a truncated or type-coerced id landing on
   * one {@code model.vocab} already owns re-triggers the exact collision-vacating finding 1 above
   * was written to prevent. A genuinely non-coercible value ({@code {}}, {@code "abc"}) already
   * throws today, but as a raw Jackson error wrapped into {@link #load}'s generic "failed to parse"
   * message rather than this method's own specific diagnostic; checking the node type directly
   * gives every case the same clear failure mode.
   */
  private static void requireValidAddedTokenIdentity(JsonNode entry) {
    JsonNode contentNode = entry.path("content");
    if (contentNode.isMissingNode()
        || contentNode.isNull()
        || !contentNode.isString()
        || contentNode.asString().isEmpty()) {
      throw new TokenizerException(
          "TokenizerJsonLoader: added_tokens entry has missing, empty, or non-string content: "
              + entry);
    }
    JsonNode idNode = entry.path("id");
    if (idNode.isMissingNode() || idNode.isNull() || !idNode.isIntegralNumber()) {
      throw new TokenizerException(
          "TokenizerJsonLoader: added_tokens['"
              + contentNode.asString()
              + "'] has no integral id: "
              + idNode);
    }
  }

  /**
   * Requires every {@code added_tokens} entry's {@code (id, content)} pair to agree with every
   * *other* entry's, mirroring {@code model.vocab}'s own bijection check ({@link #parseModel}, PR
   * #14 review round 7, finding 2) and {@code HfTokenizer}'s {@code
   * requireInternallyConsistentTemplateTokens} for {@code TemplateProcessing} special tokens.
   * {@code added_tokens} was the one declaration source with no such check: verified directly --
   * two entries sharing one id under different content loaded cleanly, then {@link
   * AddedTokenSplitter} (built from every entry's content) still split input text at the first
   * entry's content while {@link Vocabulary}'s added-token collision cleanup had already vacated it
   * in favor of the second, so encoding perfectly valid input threw "no vocabulary entry" for a
   * token the file itself declared; the mirror case (one content under two different ids) loaded
   * cleanly and then made {@code decode} throw for an id the file itself declared, once the second
   * entry's collision vacated the first (PR #14 review round 9, finding 1).
   */
  private static void requireInternallyConsistentAddedTokens(List<AddedToken> tokens) {
    Map<Integer, String> contentById = new HashMap<>();
    Map<String, Integer> idByContent = new HashMap<>();
    for (AddedToken t : tokens) {
      String existingContent = contentById.putIfAbsent(t.id(), t.content());
      if (existingContent != null && !existingContent.equals(t.content())) {
        throw new TokenizerException(
            "TokenizerJsonLoader: added_tokens declares id "
                + t.id()
                + " for both '"
                + existingContent
                + "' and '"
                + t.content()
                + "'");
      }
      Integer existingId = idByContent.putIfAbsent(t.content(), t.id());
      if (existingId != null && !existingId.equals(t.id())) {
        throw new TokenizerException(
            "TokenizerJsonLoader: added_tokens declares '"
                + t.content()
                + "' for both id "
                + existingId
                + " and id "
                + t.id());
      }
    }
  }

  /**
   * Requires an {@code added_tokens} entry's {@code lstrip}/{@code rstrip}/{@code single_word}
   * fields to all be {@code false} -- {@link AddedTokenSplitter}'s own javadoc already documents
   * that it implements none of them, so a fine-tune's added token declaring {@code lstrip: true}
   * would otherwise load cleanly and tokenize adjacent whitespace differently from HF with no
   * diagnostic (PR #14 review round 4, finding 7).
   *
   * <p>{@code normalized} is checked separately, not folded into this loop, for two reasons (PR #14
   * review round 5, finding 5): first, it only matters when {@code normalizer} isn't {@link
   * NormalizerKind#NONE} -- HF routes an added token through a normalization trie only when a
   * normalizer actually exists, so with no normalizer (true of the entire Llama-3 family, including
   * this port's own {@code llama3-style} fixture), {@code normalized: true} vs {@code false} is a
   * no-op distinction, not a real divergence from what {@link AddedTokenSplitter} leaves
   * unimplemented. Second, HF's own serde defaults an absent {@code normalized} to {@code !special}
   * (normalized by default unless the token is special), not uniformly to {@code false} like the
   * three fields above -- so the check must default on {@code special} too, or a non-special token
   * that omits {@code normalized} (which HF would still normalize) silently bypasses this
   * validation.
   */
  private static void requireNoStrippingOrSingleWordFlags(JsonNode entry, String content) {
    for (String field : List.of("lstrip", "rstrip", "single_word")) {
      requireBoolean(entry, field, "added_tokens['" + content + "']." + field);
      if (entry.path(field).asBoolean(false)) {
        throw new TokenizerException(
            "TokenizerJsonLoader: added_tokens['"
                + content
                + "'] has "
                + field
                + "=true, which AddedTokenSplitter does not implement");
      }
    }
  }

  private static void requireNoNormalization(
      JsonNode entry, String content, boolean special, NormalizerKind normalizer) {
    if (normalizer == NormalizerKind.NONE) {
      return;
    }
    JsonNode normalizedNode = entry.path("normalized");
    // Rejects any present, non-boolean normalized value outright, before even considering what it
    // would coerce to: asBoolean(default) silently returns its default for ANY node it can't
    // coerce to a boolean at all (the same asDouble-style defect round 6 finding 3 fixed for
    // dropout, here for asBoolean instead), but unlike dropout's numeric 0.0, there is no
    // non-boolean normalized value this port can call semantically equivalent to a real boolean
    // -- HF's own serde requires this field to be a genuine JSON boolean when present, so any
    // other JSON type here is itself a file this port's target models never produce. Checking
    // this first, rather than folding it into the reason branches below, closes two problems
    // round 7 finding 4's three-branch version still had (PR #14 review round 8, finding 3):
    // a string like "true" or a number like 1 reached the old "normalized=true" branch even for
    // a *special* token, where that branch's "defaults to true for a non-special added token"
    // wording is false on both counts (it isn't a default, and the token isn't non-special); and
    // a string like "false" or a number like 0 coerced to boolean false regardless of the
    // special-token default, silently bypassing this validation entirely for a file HF's own
    // strictly-typed serde would itself reject.
    if (!normalizedNode.isMissingNode()
        && !normalizedNode.isNull()
        && !normalizedNode.isBoolean()) {
      throw new TokenizerException(
          "TokenizerJsonLoader: added_tokens['"
              + content
              + "'] has non-boolean normalized "
              + normalizedNode
              + " (HF's own serde requires a JSON boolean here)");
    }
    if (!normalizedNode.asBoolean(!special)) {
      return;
    }
    // Distinguishing an explicit "normalized": true from an absent field defaulting to true (via
    // !special, per HF's own serde) matters for the thrown message: naming a field the file
    // doesn't actually contain would be misleading (PR #14 review round 6, finding 5). Only two
    // branches remain now that the non-boolean case throws above instead of falling through here.
    String reason =
        normalizedNode.isMissingNode() || normalizedNode.isNull()
            ? "normalized absent, which defaults to true for a non-special added token"
            : "normalized=true";
    throw new TokenizerException(
        "TokenizerJsonLoader: added_tokens['"
            + content
            + "'] has "
            + reason
            + ", which AddedTokenSplitter does not implement");
  }

  private static List<AddedToken> parseAddedTokens(JsonNode node, NormalizerKind normalizer) {
    List<AddedToken> tokens = new ArrayList<>();
    for (JsonNode entry : node) {
      requireValidAddedTokenIdentity(entry);
      String content = entry.path("content").asString();
      requireBoolean(entry, "special", "added_tokens['" + content + "'].special");
      boolean special = entry.path("special").asBoolean(false);
      requireNoStrippingOrSingleWordFlags(entry, content);
      requireNoNormalization(entry, content, special, normalizer);
      tokens.add(new AddedToken(entry.path("id").asInt(), content, special));
    }
    requireInternallyConsistentAddedTokens(tokens);
    return tokens;
  }
}
