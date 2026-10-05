package se.alipsa.jmlx.tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import tools.jackson.databind.JsonNode;

/** Immutable component runtime shared by all calls to one tokenizer. */
final class TokenizerRuntime {

  private final TokenizerDefinition definition;
  private final TokenizerModels.Encoder modelEncoder;
  private final java.util.function.UnaryOperator<AlignedText> normalizer;
  private final AddedTokenMatcher rawMatcher;
  private final AddedTokenMatcher normalizedMatcher;
  private final Vocabulary vocabulary;
  private final int baseVocabularyMaxKnownId;

  TokenizerRuntime(TokenizerDefinition definition) {
    this.definition = Objects.requireNonNull(definition, "definition");
    this.modelEncoder = TokenizerModels.prepare(definition.model());
    this.normalizer = NormalizerPipeline.prepare(definition.normalizer());
    this.rawMatcher =
        new AddedTokenMatcher(definition.addedTokens(), false, definition.normalizer());
    this.normalizedMatcher =
        new AddedTokenMatcher(definition.addedTokens(), true, definition.normalizer());
    List<AddedToken> templateTokens = collectTemplateTokens(definition.postProcessor());
    Vocabulary base = new Vocabulary(definition.model().vocab(), definition.addedTokens());
    requireCompatible(templateTokens, base);
    List<AddedToken> merged = new ArrayList<>(definition.addedTokens());
    for (AddedToken token : templateTokens) {
      if (!base.hasToken(token.content())) {
        merged.add(token);
      }
    }
    this.vocabulary = new Vocabulary(definition.model().vocab(), merged);
    this.baseVocabularyMaxKnownId = base.maxKnownId();
  }

  int vocabSize() {
    return baseVocabularyMaxKnownId + 1;
  }

  Vocabulary vocabulary() {
    return vocabulary;
  }

  IncrementalTokenDecoder newIncrementalDecoder(boolean skipSpecialTokens) {
    return new RuntimeIncrementalDecoder(this, skipSpecialTokens, definition.decoder());
  }

  DecodableToken decodableToken(int id, boolean skipSpecialTokens) {
    if (skipSpecialTokens && vocabulary.isSpecial(id)) {
      return null;
    }
    if (vocabulary.hasId(id)) {
      return new DecodableToken(vocabulary.tokenOf(id));
    }
    if (id > baseVocabularyMaxKnownId) {
      return null;
    }
    throw new TokenizerException(
        "HfTokenizer.decode: no vocabulary entry for id "
            + id
            + ", within the known vocabulary range (max "
            + baseVocabularyMaxKnownId
            + ")");
  }

  EncodingOptions configuredDefaults(boolean addSpecialTokens) {
    EncodingOptions defaults = definition.configuredDefaults();
    return new EncodingOptions(addSpecialTokens, defaults.truncation(), defaults.padding());
  }

  TokenizerEncoding encode(String text, EncodingOptions options) {
    Objects.requireNonNull(text, "text");
    Objects.requireNonNull(options, "options");
    if (options.padding().enabled()) {
      validatePadding(options.padding());
    }
    List<TokenPiece> pieces = encodeInput(text);
    int specialBudget = options.addSpecialTokens() ? applyPostProcessor(List.of(), true).size() : 0;
    if (options.truncation().enabled()) {
      int available = options.truncation().maxLength() - specialBudget;
      if (available < 0) {
        throw new TokenizerException(
            "TokenizerRuntime: truncation maxLength cannot contain required special tokens");
      }
      // A zero remaining budget is a successful truncation to empty under every strategy:
      // the encoding keeps only the post-processor special tokens (HF does not fail here).
      if (available > 0
          && pieces.size() > available
          && definition.configuredStrategy() == PairTruncationStrategy.ONLY_SECOND) {
        throw new TokenizerException("Truncation error: Second sequence not provided");
      }
      pieces = truncate(pieces, available, options.truncation().direction());
    }
    pieces = applyPostProcessor(pieces, options.addSpecialTokens());
    if (options.padding().enabled()) {
      pieces = pad(pieces, options.padding());
    }
    return columns(pieces);
  }

  TokenizerEncoding encodePair(String text, String textPair, PairEncodingOptions pairOptions) {
    Objects.requireNonNull(text, "text");
    Objects.requireNonNull(textPair, "textPair");
    Objects.requireNonNull(pairOptions, "options");
    EncodingOptions options = pairOptions.options();
    if (options.padding().enabled()) {
      validatePadding(options.padding());
    }
    List<TokenPiece> first = encodeInput(text);
    List<TokenPiece> second = encodeInput(textPair);
    int budget = pairProcess(List.of(), List.of(), options.addSpecialTokens()).size();
    if (options.truncation().enabled()) {
      int available = options.truncation().maxLength() - budget;
      if (available < 0) {
        throw new TokenizerException("truncation maxLength cannot contain pair special tokens");
      }
      int remove = Math.max(0, first.size() + second.size() - available);
      int firstLength = first.size();
      int secondLength = second.size();
      // A zero remaining budget is a successful truncation to empty under every strategy:
      // the encoding keeps only the pair post-processor special tokens (HF does not fail here).
      if (available == 0) {
        firstLength = 0;
        secondLength = 0;
      } else if (remove > 0) {
        switch (pairOptions.strategy()) {
          case ONLY_FIRST -> {
            if (firstLength <= remove) {
              throw new TokenizerException("Truncation error: Sequence to truncate too short");
            }
            firstLength -= remove;
          }
          case ONLY_SECOND -> {
            if (secondLength <= remove) {
              throw new TokenizerException("Truncation error: Sequence to truncate too short");
            }
            secondLength -= remove;
          }
          case LONGEST_FIRST -> {
            // Mirrors tokenizers/src/utils/truncation.rs: each sequence is first truncated to
            // the raw max_length during model tokenization, then the remaining budget
            // (available, i.e. max_length minus the special tokens) is split, giving the extra
            // token to the originally longer input.
            firstLength = Math.min(firstLength, options.truncation().maxLength());
            secondLength = Math.min(secondLength, options.truncation().maxLength());
            int shorter = Math.min(firstLength, secondLength);
            int longer = shorter > available ? shorter : Math.max(shorter, available - shorter);
            if (shorter + longer > available) {
              shorter = available / 2;
              longer = shorter + available % 2;
            }
            boolean firstLonger = firstLength > secondLength;
            firstLength = firstLonger ? longer : shorter;
            secondLength = firstLonger ? shorter : longer;
          }
          default -> throw new TokenizerException("unsupported pair truncation strategy");
        }
      }
      first = truncate(first, firstLength, options.truncation().direction());
      second = truncate(second, secondLength, options.truncation().direction());
    }
    List<TokenPiece> output = pairProcess(first, second, options.addSpecialTokens());
    if (options.padding().enabled()) {
      output = pad(output, options.padding());
    }
    return columns(output);
  }

  private List<TokenPiece> pairProcess(
      List<TokenPiece> first, List<TokenPiece> second, boolean special) {
    JsonNode processor = definition.pairPostProcessor();
    if (processor == null) {
      throw new TokenizerException("pair encoding requires a BERT pair post-processor");
    }
    if ("Sequence".equals(processor.path("type").asString())) {
      JsonNode template = null;
      for (JsonNode step : processor.path("processors")) {
        if ("BertProcessing".equals(step.path("type").asString())
            || "TemplateProcessing".equals(step.path("type").asString())) {
          if (template != null) {
            throw new TokenizerException("multiple pair templates are unsupported");
          }
          template = step;
        } else {
          throw new TokenizerException("unsupported pair post-processor sequence");
        }
      }
      processor = template;
    }
    if (processor == null) {
      throw new TokenizerException("missing pair template");
    }
    List<TokenPiece> output = new ArrayList<>();
    if ("BertProcessing".equals(processor.path("type").asString())) {
      if (special) {
        pairSpecial(
            output,
            processor.path("cls").get(0).asString(),
            processor.path("cls").get(1).intValue(),
            0);
      }
      typed(output, first, 0);
      if (special) {
        pairSpecial(
            output,
            processor.path("sep").get(0).asString(),
            processor.path("sep").get(1).intValue(),
            0);
      }
      typed(output, second, 1);
      if (special) {
        pairSpecial(
            output,
            processor.path("sep").get(0).asString(),
            processor.path("sep").get(1).intValue(),
            1);
      }
    } else if ("TemplateProcessing".equals(processor.path("type").asString())) {
      boolean seenFirst = false;
      boolean seenSecond = false;
      for (JsonNode item : processor.path("pair")) {
        JsonNode sequence = item.get("Sequence");
        if (sequence != null) {
          String id = sequence.path("id").asString();
          int type = sequence.path("type_id").asInt(0);
          if ("A".equals(id) && type == 0 && !seenFirst) {
            typed(output, first, 0);
            seenFirst = true;
          } else if ("B".equals(id) && type == 1 && !seenSecond) {
            typed(output, second, 1);
            seenSecond = true;
          } else {
            throw new TokenizerException("pair template must distinguish A:0 and B:1 exactly once");
          }
        } else if (item.has("SpecialToken")) {
          JsonNode token = item.path("SpecialToken");
          int type = token.path("type_id").asInt(0);
          JsonNode info = processor.path("special_tokens").path(token.path("id").asString());
          if (!info.path("ids").isArray()
              || info.path("ids").isEmpty()
              || info.path("ids").size() != info.path("tokens").size()) {
            throw new TokenizerException("invalid pair special token definition");
          }
          if (special) {
            for (int i = 0; i < info.path("ids").size(); i++) {
              pairSpecial(
                  output,
                  info.path("tokens").get(i).asString(),
                  info.path("ids").get(i).intValue(),
                  type);
            }
          }
        } else {
          throw new TokenizerException("unsupported pair template item");
        }
      }
      if (!seenFirst || !seenSecond) {
        throw new TokenizerException("pair template must contain A:0 and B:1");
      }
    } else {
      throw new TokenizerException("unsupported pair post-processor");
    }
    return output;
  }

  private static void pairSpecial(List<TokenPiece> output, String text, int id, int type) {
    output.add(new TokenPiece(text, TokenOffset.NONE, id, type, true));
  }

  private static void typed(List<TokenPiece> output, List<TokenPiece> input, int type) {
    for (TokenPiece piece : input) {
      output.add(new TokenPiece(piece.text(), piece.offset(), piece.id(), type, piece.special()));
    }
  }

  private List<TokenPiece> encodeInput(String text) {
    AlignedText original = AlignedText.original(text);
    List<TokenPiece> result = new ArrayList<>();
    for (AddedTokenMatcher.Segment raw : rawMatcher.split(original)) {
      if (raw.token() != null) {
        result.add(added(raw));
        continue;
      }
      AlignedText normalized = normalizer.apply(raw.text());
      for (AddedTokenMatcher.Segment segment : normalizedMatcher.split(normalized)) {
        if (segment.token() != null) {
          result.add(added(segment));
          continue;
        }
        for (AlignedText pretoken :
            PreTokenizerPipeline.apply(definition.preTokenizer(), segment.text())) {
          result.addAll(modelEncoder.encode(pretoken));
        }
      }
    }
    return result;
  }

  private TokenPiece added(AddedTokenMatcher.Segment segment) {
    AddedToken token = segment.token();
    return new TokenPiece(segment.text().text(), segment.text().offset(), token.id(), 0, false);
  }

  String decode(List<Integer> ids, boolean skipSpecialTokens) {
    Objects.requireNonNull(ids, "ids");
    List<String> tokens = new ArrayList<>();
    for (int id : ids) {
      if (skipSpecialTokens && vocabulary.isSpecial(id)) {
        continue;
      }
      if (vocabulary.hasId(id)) {
        tokens.add(vocabulary.tokenOf(id));
      } else if (id <= baseVocabularyMaxKnownId) {
        throw new TokenizerException(
            "HfTokenizer.decode: no vocabulary entry for id "
                + id
                + ", within the known vocabulary range (max "
                + baseVocabularyMaxKnownId
                + ")");
      }
    }
    return DecoderPipeline.decode(definition.decoder(), tokens);
  }

  record DecodableToken(String text) {}

  private List<TokenPiece> applyPostProcessor(List<TokenPiece> input, boolean addSpecialTokens) {
    List<TokenPiece> result = new ArrayList<>(input);
    for (PostProcessorStep step : definition.postProcessor()) {
      if (step instanceof TemplateProcessingStep template) {
        result = applyTemplate(template, result, addSpecialTokens);
      }
    }
    if (definition.trimByteLevelOffsets()) {
      result = trimSyntheticSpaces(result);
    }
    return result;
  }

  private List<TokenPiece> applyTemplate(
      TemplateProcessingStep template, List<TokenPiece> input, boolean addSpecialTokens) {
    List<TokenPiece> result = new ArrayList<>();
    for (TemplateItem item : template.single()) {
      if (item instanceof SequenceItem) {
        result.addAll(input);
      } else if (item instanceof SpecialTokenItem token && addSpecialTokens) {
        SpecialTokenInfo info = template.specialTokens().get(token.id());
        if (info == null) {
          throw new TokenizerException(
              "TokenizerRuntime: template references unknown special token '" + token.id() + "'");
        }
        for (int index = 0; index < info.ids().size(); index++) {
          result.add(
              new TokenPiece(
                  info.tokens().get(index), TokenOffset.NONE, info.ids().get(index), 0, true));
        }
      }
    }
    return result;
  }

  /**
   * Replicates HF's {@code ByteLevel} post-processor {@code process_offsets} (see {@code
   * huggingface/tokenizers}'s {@code byte_level.rs}) exactly: for each token, count leading and
   * trailing synthetic-space ({@code 'Ġ'}) characters in its own text and trim that many bytes off
   * each side of its offset, clamped so start never passes end. The one exemption is the very first
   * token in the list: when {@code add_prefix_space} is set and it has exactly one leading space,
   * that space is the one the pre-tokenizer itself synthesized, so it is left untrimmed instead of
   * being treated as if it existed in the original input.
   */
  private List<TokenPiece> trimSyntheticSpaces(List<TokenPiece> input) {
    List<TokenPiece> result = new ArrayList<>(input.size());
    for (int index = 0; index < input.size(); index++) {
      TokenPiece piece = input.get(index);
      int start = piece.offset().startByte();
      int end = piece.offset().endByte();
      String text = piece.text();
      int leading = 0;
      while (leading < text.length() && text.charAt(leading) == 'Ġ') {
        leading++;
      }
      int trailing = 0;
      while (trailing < text.length() && text.charAt(text.length() - 1 - trailing) == 'Ġ') {
        trailing++;
      }
      if (leading > 0 || trailing > 0) {
        if (leading > 0) {
          boolean isFirst = index == 0 || start == 0;
          if (isFirst && definition.byteLevelAddPrefixSpace() && leading == 1) {
            leading = 0;
          }
          start = Math.min(start + leading, end);
        }
        if (trailing > 0 && end >= trailing) {
          end = Math.max(end - trailing, start);
        }
      }
      result.add(
          new TokenPiece(
              text, new TokenOffset(start, end), piece.id(), piece.typeId(), piece.special()));
    }
    return result;
  }

  private static List<TokenPiece> truncate(
      List<TokenPiece> input, int maximum, Direction direction) {
    if (input.size() <= maximum) {
      return input;
    }
    return direction == Direction.RIGHT
        ? new ArrayList<>(input.subList(0, maximum))
        : new ArrayList<>(input.subList(input.size() - maximum, input.size()));
  }

  private List<TokenPiece> pad(List<TokenPiece> input, Padding padding) {
    if (input.size() >= padding.length()) {
      return input;
    }
    int count = padding.length() - input.size();
    TokenPiece pad =
        new TokenPiece(
            padding.padToken(), TokenOffset.NONE, padding.padId(), padding.padTypeId(), true, true);
    List<TokenPiece> result = new ArrayList<>(padding.length());
    if (padding.direction() == Direction.LEFT) {
      for (int index = 0; index < count; index++) {
        result.add(pad);
      }
    }
    result.addAll(input);
    if (padding.direction() == Direction.RIGHT) {
      for (int index = 0; index < count; index++) {
        result.add(pad);
      }
    }
    return result;
  }

  private void validatePadding(Padding padding) {
    if (!vocabulary.hasId(padding.padId())
        || !vocabulary.hasToken(padding.padToken())
        || vocabulary.idOf(padding.padToken()) != padding.padId()) {
      throw new TokenizerException(
          "TokenizerRuntime: padding token and id must resolve to the loaded vocabulary");
    }
  }

  private TokenizerEncoding columns(List<TokenPiece> pieces) {
    List<Integer> ids = new ArrayList<>(pieces.size());
    List<Integer> types = new ArrayList<>(pieces.size());
    List<Integer> attention = new ArrayList<>(pieces.size());
    List<Integer> special = new ArrayList<>(pieces.size());
    List<TokenOffset> offsets = new ArrayList<>(pieces.size());
    List<String> tokens = new ArrayList<>(pieces.size());
    for (TokenPiece piece : pieces) {
      int id = piece.id() == null ? vocabulary.idOf(piece.text()) : piece.id();
      ids.add(id);
      types.add(piece.typeId());
      attention.add(piece.padding() ? 0 : 1);
      special.add(piece.special() ? 1 : 0);
      offsets.add(piece.offset());
      tokens.add(piece.text());
    }
    return new TokenizerEncoding(ids, types, attention, special, offsets, tokens);
  }

  private static List<AddedToken> collectTemplateTokens(List<PostProcessorStep> steps) {
    List<AddedToken> result = new ArrayList<>();
    for (PostProcessorStep step : steps) {
      if (step instanceof TemplateProcessingStep template) {
        for (SpecialTokenInfo info : template.specialTokens().values()) {
          for (int index = 0; index < info.ids().size(); index++) {
            result.add(new AddedToken(info.ids().get(index), info.tokens().get(index), true));
          }
        }
      }
    }
    return result;
  }

  private static void requireCompatible(List<AddedToken> tokens, Vocabulary base) {
    Map<Integer, String> textById = new java.util.HashMap<>();
    Map<String, Integer> idByText = new java.util.HashMap<>();
    for (AddedToken token : tokens) {
      String oldText = textById.putIfAbsent(token.id(), token.content());
      Integer oldId = idByText.putIfAbsent(token.content(), token.id());
      if ((oldText != null && !oldText.equals(token.content()))
          || (oldId != null && oldId != token.id())
          || (base.hasId(token.id()) && !base.tokenOf(token.id()).equals(token.content()))
          || (base.hasToken(token.content()) && base.idOf(token.content()) != token.id())) {
        throw new TokenizerException(
            "TokenizerRuntime: conflicting post-processor token '" + token.content() + "'");
      }
    }
  }
}
