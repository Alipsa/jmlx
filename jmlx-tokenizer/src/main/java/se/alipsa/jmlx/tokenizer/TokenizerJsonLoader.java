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
    JsonNode root;
    try {
      root = MAPPER.readTree(Files.newInputStream(path));
    } catch (IOException | JacksonException e) {
      throw new TokenizerException("TokenizerJsonLoader.load: failed to parse " + path, e);
    }
    return new TokenizerJson(
        parseNormalizer(root.path("normalizer")),
        parsePreTokenizer(root.get("pre_tokenizer")),
        parsePostProcessor(root.path("post_processor")),
        parseModel(root.get("model")),
        parseAddedTokens(root.path("added_tokens")));
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

  private static PreTokenizerConfig parsePreTokenizer(JsonNode node) {
    if (node == null || !"Sequence".equals(node.path("type").asString(""))) {
      throw new TokenizerException(
          "TokenizerJsonLoader: expected pre_tokenizer.type == 'Sequence'");
    }
    String regex = null;
    boolean addPrefixSpace = false;
    for (JsonNode step : node.get("pretokenizers")) {
      String stepType = step.path("type").asString("");
      if ("Split".equals(stepType)) {
        regex = step.path("pattern").path("Regex").asString(null);
      } else if ("ByteLevel".equals(stepType)) {
        addPrefixSpace = step.path("add_prefix_space").asBoolean(false);
      }
    }
    if (regex == null) {
      throw new TokenizerException(
          "TokenizerJsonLoader: pre_tokenizer.pretokenizers has no Split step");
    }
    return new PreTokenizerConfig(
        Pattern.compile(regex, Pattern.UNICODE_CHARACTER_CLASS), addPrefixSpace);
  }

  private static List<PostProcessorStep> parsePostProcessor(JsonNode node) {
    List<PostProcessorStep> steps = new ArrayList<>();
    if (node.isNull() || node.isMissingNode()) {
      return steps;
    }
    String type = node.path("type").asString("");
    if ("Sequence".equals(type)) {
      for (JsonNode step : node.get("processors")) {
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
      List<TemplateItem> single = new ArrayList<>();
      for (JsonNode item : node.get("single")) {
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
    Map<String, Integer> mergeRank = new HashMap<>();
    int rank = 0;
    for (JsonNode merge : node.path("merges")) {
      mergeRank.put(merge.asString(), rank++);
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
