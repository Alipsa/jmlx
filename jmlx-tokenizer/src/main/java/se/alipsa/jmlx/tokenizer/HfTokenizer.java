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
    this.vocabulary = new Vocabulary(json.model().vocab(), json.addedTokens());
    this.merger = new BpeMerger(json.model());
    this.pretokenizer = new ByteLevelPreTokenizer(json.preTokenizer());
    this.addedTokenSplitter = new AddedTokenSplitter(json.addedTokens());
    this.addedTokenContents = new HashSet<>();
    json.addedTokens().forEach(t -> addedTokenContents.add(t.content()));
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
      Integer id = token.id();
      if (id != null) {
        // A template id with no vocabulary entry at all is trusted as-is: HF's own
        // TemplateProcessing performs no vocabulary lookup on special_tokens ids, and
        // ResolvedToken's own javadoc documents exactly this case (a template can reference a
        // special token whose id isn't duplicated anywhere in model.vocab/added_tokens). What
        // must still be rejected is an id that collides with a *different*, already-known
        // vocabulary token: that is a genuine internal contradiction within the same file (not
        // merely "unknown"), and baking it in would silently swap in the wrong token wherever
        // this id is later decoded (PR #14 review, finding 2, correcting round 2's finding 7,
        // which rejected every unknown id and so also rejected this legitimate case).
        if (vocabulary.hasId(id) && !vocabulary.tokenOf(id).equals(token.text())) {
          throw new TokenizerException(
              "HfTokenizer.encode: TemplateProcessing special token '"
                  + token.text()
                  + "' has id "
                  + id
                  + ", which the vocabulary maps to a different token '"
                  + vocabulary.tokenOf(id)
                  + "'");
        }
        ids.add(id);
      } else {
        ids.add(vocabulary.idOf(token.text()));
      }
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
