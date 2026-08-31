package se.alipsa.jmlx.tokenizer;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/** A byte-level BPE tokenizer loaded from a {@code tokenizer.json} file. */
public final class HfTokenizer {

  private final TokenizerJson json;
  private final Vocabulary vocabulary;
  private final BpeMerger merger;
  private final ByteLevelPreTokenizer pretokenizer;
  private final AddedTokenSplitter addedTokenSplitter;
  private final Set<String> addedTokenContents;

  private HfTokenizer(TokenizerJson json) {
    this.json = json;
    Vocabulary baseVocabulary = new Vocabulary(json.model().vocab(), json.addedTokens());
    List<AddedToken> templateTokens = collectTemplateSpecialTokens(json.postProcessor());
    requireNoTemplateVocabularyConflicts(templateTokens, baseVocabulary);
    // Registering validated template tokens as if they were addedTokens, rather than leaving
    // them merely "trusted but unregistered" in encode() below, is what makes decode() agree with
    // encode(): otherwise a template id with no other vocabulary entry either vanishes silently
    // from decode's above-maxKnownId skip, or throws as an in-range hole if it happens to be
    // numerically below some unrelated, larger id -- both observed with the pre-round-4 encode
    // check (PR #14 review round 4, finding 2).
    List<AddedToken> mergedAddedTokens = new ArrayList<>(json.addedTokens());
    mergedAddedTokens.addAll(templateTokens);
    this.vocabulary = new Vocabulary(json.model().vocab(), mergedAddedTokens);
    this.merger = new BpeMerger(json.model());
    this.pretokenizer = new ByteLevelPreTokenizer(json.preTokenizer());
    // addedTokenSplitter is deliberately scoped to json.addedTokens() only, not
    // mergedAddedTokens: it splits raw input *text* at encode time, and HF's own AddedVocabulary
    // component never consults TemplateProcessing's special_tokens for that -- those are two
    // independent mechanisms. addedTokenContents, by contrast, must include template-registered
    // tokens too: it decides whether ByteLevelDecoder passes a resolved token through literally
    // or byte-decodes it, and a template-only special token (see mergedAddedTokens above) needs
    // the same literal pass-through decode() applies to a "real" added_tokens entry.
    this.addedTokenSplitter = new AddedTokenSplitter(json.addedTokens());
    this.addedTokenContents = new HashSet<>();
    mergedAddedTokens.forEach(t -> addedTokenContents.add(t.content()));
  }

  /**
   * Every {@code (id, text)} pair any {@code TemplateProcessing} step's special_tokens declares.
   */
  private static List<AddedToken> collectTemplateSpecialTokens(List<PostProcessorStep> steps) {
    List<AddedToken> tokens = new ArrayList<>();
    for (PostProcessorStep step : steps) {
      if (step instanceof TemplateProcessingStep template) {
        for (SpecialTokenInfo info : template.specialTokens().values()) {
          for (int i = 0; i < info.ids().size(); i++) {
            tokens.add(new AddedToken(info.ids().get(i), info.tokens().get(i), true));
          }
        }
      }
    }
    return tokens;
  }

  /**
   * Requires every template-declared {@code (id, text)} pair to not contradict {@code
   * baseVocabulary} (built from just {@code model.vocab} + {@code added_tokens}) in either
   * direction: {@code id} already meaning a *different* token, or {@code text} already meaning a
   * *different* id. Either is a genuine internal contradiction within the same file -- not merely
   * "unknown," which is the legitimate case {@link ResolvedToken}'s own javadoc documents and this
   * check does not reject (PR #14 review, finding 2, correcting round 3's one-directional check,
   * which only tested the first direction).
   */
  private static void requireNoTemplateVocabularyConflicts(
      List<AddedToken> templateTokens, Vocabulary baseVocabulary) {
    for (AddedToken t : templateTokens) {
      if (baseVocabulary.hasId(t.id()) && !baseVocabulary.tokenOf(t.id()).equals(t.content())) {
        throw new TokenizerException(
            "HfTokenizer: TemplateProcessing special token '"
                + t.content()
                + "' has id "
                + t.id()
                + ", which the vocabulary maps to a different token '"
                + baseVocabulary.tokenOf(t.id())
                + "'");
      }
      if (baseVocabulary.hasToken(t.content()) && baseVocabulary.idOf(t.content()) != t.id()) {
        throw new TokenizerException(
            "HfTokenizer: TemplateProcessing special token '"
                + t.content()
                + "' is declared with id "
                + t.id()
                + ", but the vocabulary already maps it to a different id "
                + baseVocabulary.idOf(t.content()));
      }
    }
  }

  /** Loads a tokenizer from a {@code tokenizer.json} file. */
  public static HfTokenizer fromFile(Path tokenizerJsonPath) {
    Objects.requireNonNull(
        tokenizerJsonPath, "HfTokenizer.fromFile: tokenizerJsonPath must not be null");
    return new HfTokenizer(TokenizerJsonLoader.load(tokenizerJsonPath));
  }

  /** Encodes {@code text} into token ids. */
  public List<Integer> encode(String text, boolean addSpecialTokens) {
    Objects.requireNonNull(text, "HfTokenizer.encode: text must not be null");
    List<String> tokens = new ArrayList<>();
    for (AddedTokenSplitter.Segment segment : addedTokenSplitter.split(text)) {
      if (segment.isAddedToken()) {
        tokens.add(segment.text());
        continue;
      }
      String normalized = TextNormalizer.normalize(json.normalizer(), segment.text());
      for (String chunk : pretokenizer.split(normalized)) {
        tokens.addAll(merger.merge(chunk));
      }
    }
    List<ResolvedToken> processed =
        PostProcessorApplier.apply(json.postProcessor(), tokens, addSpecialTokens);
    List<Integer> ids = new ArrayList<>(processed.size());
    for (ResolvedToken token : processed) {
      // A ResolvedToken's pre-resolved id (from a TemplateProcessing special token) is trusted
      // directly, not re-validated here: every such id was already checked against, and merged
      // into, this.vocabulary in the constructor (requireNoTemplateVocabularyConflicts), so
      // hasId/tokenOf/idOf all already agree with it by construction.
      Integer id = token.id();
      ids.add(id != null ? id : vocabulary.idOf(token.text()));
    }
    return ids;
  }

  /**
   * Decodes token ids back into text. An id above {@link Vocabulary#maxKnownId} is skipped rather
   * than aborting the whole decode -- e.g. Qwen2.5's {@code config.json} {@code vocab_size}
   * (152064) exceeds its tokenizer vocab (151665), so a sampled logit can legitimately fall in that
   * gap. An id within the known vocabulary range that still has no entry is a real hole (the wrong
   * tokenizer for the checkpoint, a mis-parsed {@code added_tokens}, a {@link Vocabulary} bug) and
   * still throws, so it can't be mistaken for the above-vocab case.
   */
  public String decode(List<Integer> ids, boolean skipSpecialTokens) {
    Objects.requireNonNull(ids, "HfTokenizer.decode: ids must not be null");
    List<String> tokens = new ArrayList<>();
    for (int id : ids) {
      if (skipSpecialTokens && vocabulary.isSpecial(id)) {
        continue;
      }
      if (!vocabulary.hasId(id)) {
        if (id > vocabulary.maxKnownId()) {
          continue;
        }
        throw new TokenizerException(
            "HfTokenizer.decode: no vocabulary entry for id "
                + id
                + ", within the known vocabulary range (max "
                + vocabulary.maxKnownId()
                + ")");
      }
      tokens.add(vocabulary.tokenOf(id));
    }
    return ByteLevelDecoder.decode(tokens, addedTokenContents);
  }
}
