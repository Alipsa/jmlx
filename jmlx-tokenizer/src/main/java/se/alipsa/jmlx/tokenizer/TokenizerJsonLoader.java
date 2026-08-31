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

  /** Loads and parses {@code path} as a byte-level-BPE {@code tokenizer.json}. */
  public static TokenizerJson load(Path path) {
    Objects.requireNonNull(path, "TokenizerJsonLoader.load: path must not be null");
    try {
      JsonNode root = MAPPER.readTree(Files.newInputStream(path));
      requireByteLevelDecoder(root.path("decoder"));
      return new TokenizerJson(
          parseNormalizer(root.path("normalizer")),
          parsePreTokenizer(root.get("pre_tokenizer")),
          parsePostProcessor(root.path("post_processor")),
          parseModel(root.path("model")),
          parseAddedTokens(root.path("added_tokens")));
    } catch (IOException | JacksonException e) {
      throw new TokenizerException("TokenizerJsonLoader.load: failed to parse " + path, e);
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
   */
  private static void requireByteLevelDecoder(JsonNode node) {
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
   * differently from HF.
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
    JsonNode byteLevelStep = steps.get(1);
    String behavior = splitStep.path("behavior").asString("Isolated");
    if (!"Isolated".equals(behavior)) {
      throw new TokenizerException(
          "TokenizerJsonLoader: unsupported pre_tokenizer Split behavior '"
              + behavior
              + "' (only 'Isolated' is supported)");
    }
    if (splitStep.path("invert").asBoolean(false)) {
      throw new TokenizerException(
          "TokenizerJsonLoader: unsupported pre_tokenizer Split invert=true");
    }
    if (byteLevelStep.path("use_regex").asBoolean(false)) {
      throw new TokenizerException(
          "TokenizerJsonLoader: unsupported pre_tokenizer ByteLevel use_regex=true");
    }
    String regex = splitStep.path("pattern").path("Regex").asString(null);
    if (regex == null) {
      throw new TokenizerException(
          "TokenizerJsonLoader: pre_tokenizer Split step has no pattern.Regex");
    }
    boolean addPrefixSpace = byteLevelStep.path("add_prefix_space").asBoolean(false);
    try {
      return new PreTokenizerConfig(
          Pattern.compile(regex, Pattern.UNICODE_CHARACTER_CLASS), addPrefixSpace);
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
        v.path("ids").forEach(idNode -> ids.add(idNode.asInt()));
        List<String> tokens = new ArrayList<>();
        v.path("tokens").forEach(tokenNode -> tokens.add(tokenNode.asString()));
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

  private static BpeModelConfig parseModel(JsonNode node) {
    if (!"BPE".equals(node.path("type").asString(""))) {
      throw new TokenizerException("TokenizerJsonLoader: expected model.type == 'BPE'");
    }
    Map<String, Integer> vocab = new HashMap<>();
    for (Map.Entry<String, JsonNode> entry : node.path("vocab").properties()) {
      vocab.put(entry.getKey(), entry.getValue().asInt());
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
    boolean ignoreMerges = node.path("ignore_merges").asBoolean(false);
    return new BpeModelConfig(vocab, mergeRank, ignoreMerges);
  }

  private static List<AddedToken> parseAddedTokens(JsonNode node) {
    List<AddedToken> tokens = new ArrayList<>();
    for (JsonNode entry : node) {
      tokens.add(
          new AddedToken(
              entry.path("id").asInt(),
              entry.path("content").asString(),
              entry.path("special").asBoolean(false)));
    }
    return tokens;
  }
}
