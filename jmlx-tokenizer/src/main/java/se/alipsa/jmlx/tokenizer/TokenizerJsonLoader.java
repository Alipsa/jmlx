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
    JsonNode byteLevelStep = steps.get(1);
    String behavior = splitStep.path("behavior").asString(null);
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
    if (!normalizedNode.asBoolean(!special)) {
      return;
    }
    // Distinguishing an explicit "normalized": true from an absent field defaulting to true (via
    // !special, per HF's own serde) matters for the thrown message: naming a field the file
    // doesn't actually contain would be misleading (PR #14 review round 6, finding 5).
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
      String content = entry.path("content").asString();
      boolean special = entry.path("special").asBoolean(false);
      requireNoStrippingOrSingleWordFlags(entry, content);
      requireNoNormalization(entry, content, special, normalizer);
      tokens.add(new AddedToken(entry.path("id").asInt(), content, special));
    }
    return tokens;
  }
}
