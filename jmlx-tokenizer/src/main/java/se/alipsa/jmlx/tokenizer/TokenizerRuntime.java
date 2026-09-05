package se.alipsa.jmlx.tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Immutable component runtime shared by all calls to one tokenizer. */
final class TokenizerRuntime {

  private final TokenizerDefinition definition;
  private final Vocabulary vocabulary;
  private final int baseVocabularyMaxKnownId;

  TokenizerRuntime(TokenizerDefinition definition) {
    this.definition = Objects.requireNonNull(definition, "definition");
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
      pieces = truncate(pieces, available, options.truncation().direction());
    }
    pieces = applyPostProcessor(pieces, options.addSpecialTokens());
    if (options.padding().enabled()) {
      pieces = pad(pieces, options.padding());
    }
    return columns(pieces);
  }

  private List<TokenPiece> encodeInput(String text) {
    AlignedText original = AlignedText.original(text);
    List<TokenPiece> result = new ArrayList<>();
    for (AddedTokenMatcher.Segment raw :
        AddedTokenMatcher.split(
            original, definition.addedTokens(), false, definition.normalizer())) {
      if (raw.token() != null) {
        result.add(added(raw));
        continue;
      }
      AlignedText normalized = NormalizerPipeline.apply(definition.normalizer(), raw.text());
      for (AddedTokenMatcher.Segment segment :
          AddedTokenMatcher.split(
              normalized, definition.addedTokens(), true, definition.normalizer())) {
        if (segment.token() != null) {
          result.add(added(segment));
          continue;
        }
        for (AlignedText pretoken :
            PreTokenizerPipeline.apply(definition.preTokenizer(), segment.text())) {
          result.addAll(TokenizerModels.encode(definition.model(), pretoken));
        }
      }
    }
    return result;
  }

  private TokenPiece added(AddedTokenMatcher.Segment segment) {
    AddedToken token = segment.token();
    return new TokenPiece(token.content(), segment.text().offset(), token.id(), 0, token.special());
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
      } else if (step instanceof BertProcessingStep bert && addSpecialTokens) {
        List<TokenPiece> processed = new ArrayList<>();
        processed.add(special(bert.classification()));
        processed.addAll(result);
        processed.add(special(bert.separator()));
        result = processed;
      } else if (step instanceof ByteLevelStep byteLevel && byteLevel.trimOffsets()) {
        result = trimSyntheticSpaces(result);
      }
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

  private static TokenPiece special(ResolvedToken token) {
    return new TokenPiece(token.text(), TokenOffset.NONE, token.id(), 0, true);
  }

  private static List<TokenPiece> trimSyntheticSpaces(List<TokenPiece> input) {
    List<TokenPiece> result = new ArrayList<>(input.size());
    for (TokenPiece piece : input) {
      result.add(
          piece.offset().startByte() == piece.offset().endByte()
              ? new TokenPiece(
                  piece.text(), TokenOffset.NONE, piece.id(), piece.typeId(), piece.special())
              : piece);
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
    if (input.size() > padding.length()) {
      throw new TokenizerException(
          "TokenizerRuntime: encoded sequence exceeds fixed padding length " + padding.length());
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
    for (TokenPiece piece : pieces) {
      int id = piece.id() == null ? vocabulary.idOf(piece.text()) : piece.id();
      ids.add(id);
      types.add(piece.typeId());
      attention.add(piece.padding() ? 0 : 1);
      special.add(piece.special() ? 1 : 0);
      offsets.add(piece.offset());
    }
    return new TokenizerEncoding(ids, types, attention, special, offsets);
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
      } else if (step instanceof BertProcessingStep bert) {
        result.add(new AddedToken(bert.separator().id(), bert.separator().text(), true));
        result.add(new AddedToken(bert.classification().id(), bert.classification().text(), true));
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
