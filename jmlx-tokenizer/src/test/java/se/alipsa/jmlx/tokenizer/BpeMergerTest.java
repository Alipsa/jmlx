package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class BpeMergerTest {

  private static final Map<String, Integer> BASE_VOCAB =
      Map.ofEntries(
          Map.entry("l", 0),
          Map.entry("o", 1),
          Map.entry("w", 2),
          Map.entry("t", 3),
          Map.entry("h", 4),
          Map.entry("e", 5),
          Map.entry("Ġ", 11),
          Map.entry("lo", 12),
          Map.entry("low", 13),
          Map.entry("th", 14),
          Map.entry("the", 15),
          Map.entry("Ġthe", 16));

  private static final Map<String, Integer> MERGE_RANK =
      Map.of("l o", 0, "lo w", 1, "t h", 2, "th e", 3, "Ġ the", 4);

  @Test
  void mergesLowestRankPairsInOrderUntilASingleSymbolRemains() {
    BpeMerger merger = new BpeMerger(new BpeModelConfig(BASE_VOCAB, MERGE_RANK, false));
    assertEquals(List.of("low"), merger.merge("low"));
  }

  @Test
  void mergesAcrossFourSymbolsIncludingTheByteLevelSpaceMarker() {
    BpeMerger merger = new BpeMerger(new BpeModelConfig(BASE_VOCAB, MERGE_RANK, false));
    assertEquals(List.of("Ġthe"), merger.merge("Ġthe"));
  }

  @Test
  void ignoreMergesShortCircuitsToAWholeVocabHitWithoutRunningTheMergeLoop() {
    // Empty merge-rank table: without the ignore_merges shortcut, "low" could never merge past
    // its three individual byte symbols. With it, the whole-word vocab hit wins directly.
    BpeMerger merger = new BpeMerger(new BpeModelConfig(BASE_VOCAB, Map.of(), true));
    assertEquals(List.of("low"), merger.merge("low"));
  }

  @Test
  void withoutIgnoreMergesAndNoMergeRulesEachByteStaysItsOwnSymbol() {
    BpeMerger merger = new BpeMerger(new BpeModelConfig(BASE_VOCAB, Map.of(), false));
    assertEquals(List.of("l", "o", "w"), merger.merge("low"));
  }

  @Test
  void mutatingTheVocabOrMergeRankMapAfterConstructionDoesNotAffectAnAlreadyBuiltBpeModelConfig() {
    // BpeModelConfig's compact constructor defensively copies vocab/mergeRank (PR #14 review
    // round 7, finding 3); without it, a caller mutating the map it passed in after construction
    // would silently change BpeMerger's behavior mid-flight, since BpeMerger re-reads both maps
    // directly via model.vocab()/model.mergeRank() on every merge() call rather than caching a
    // snapshot (PR #14 review round 8, finding 4, adding the coverage that fix itself lacked).
    Map<String, Integer> mutableVocab = new HashMap<>(BASE_VOCAB);
    Map<String, Integer> mutableMergeRank = new HashMap<>(MERGE_RANK);
    BpeModelConfig model = new BpeModelConfig(mutableVocab, mutableMergeRank, false);
    mutableVocab.clear();
    mutableMergeRank.clear();
    BpeMerger merger = new BpeMerger(model);
    assertEquals(List.of("low"), merger.merge("low"));
  }

  @Test
  void equalRankCandidatePairsMergeTheLeftmostOccurrenceFirst() {
    // "aaa" has two candidate "a a" pairs at the same rank (index 0-1 and 1-2). Merging the
    // leftmost first yields "aa"+"a"; merging the rightmost first would instead yield "a"+"aa".
    Map<String, Integer> vocab = Map.of("a", 0, "aa", 1);
    BpeMerger merger = new BpeMerger(new BpeModelConfig(vocab, Map.of("a a", 0), false));
    assertEquals(List.of("aa", "a"), merger.merge("aaa"));
  }

  @Test
  void mergesALongUnboundedPreTokenChunkWithoutRescanningAllPairsAfterEachMerge() {
    String word = "a".repeat(16_384);
    Map<String, Integer> ranks = new HashMap<>();
    for (String symbol = "a"; symbol.length() < word.length(); symbol += symbol) {
      ranks.put(symbol + " " + symbol, ranks.size());
    }
    BpeMerger merger = new BpeMerger(new BpeModelConfig(Map.of("a", 0, word, 1), ranks, false));
    assertEquals(List.of(word), merger.merge(word));
  }

  @Test
  void bpeMergeDropsContinuingSubwordPrefixOfRightSymbol() {
    TokenizerDefinition.Bpe model =
        new TokenizerDefinition.Bpe(
            Map.of("[UNK]", 0, "a", 1, "##b", 2, "ab", 3, "##c", 4, "abc", 5),
            Map.of("a ##b", 0, "ab ##c", 1),
            "[UNK]",
            false,
            false,
            "##",
            "",
            false);
    assertEquals(List.of("abc"), bpe(model, "abc"));
    assertEquals(List.of("ab"), bpe(model, "ab"));
  }

  @Test
  void bpeByteFallbackKeepsSubwordPrefixLikeHuggingFace() {
    TokenizerDefinition.Bpe model =
        new TokenizerDefinition.Bpe(
            Map.of("[UNK]", 0, "a", 1, "<0x62>", 2, "<0x23>", 3),
            Map.of(),
            "[UNK]",
            false,
            true,
            "##",
            "",
            false);
    assertEquals(List.of("a", "<0x23>", "<0x23>", "<0x62>"), bpe(model, "ab"));
  }

  @Test
  void bpeDropsUnknownWithoutUnkAndClampsFallbackOffsets() {
    TokenizerDefinition.Bpe noUnknown =
        new TokenizerDefinition.Bpe(
            Map.of("a", 0, "b", 1), Map.of(), null, false, false, "", "", false);
    List<TokenPiece> pieces = TokenizerModels.encode(noUnknown, AlignedText.original("azb"));
    assertEquals(List.of("a", "b"), pieces.stream().map(TokenPiece::text).toList());
    assertEquals(
        List.of(new TokenOffset(0, 1), new TokenOffset(2, 3)),
        pieces.stream().map(TokenPiece::offset).toList());

    TokenizerDefinition.Bpe fallback =
        new TokenizerDefinition.Bpe(
            Map.of("a", 0, "<0x58>", 1, "<0x62>", 2, "<0x59>", 3),
            Map.of(),
            null,
            false,
            true,
            "X",
            "Y",
            false);
    List<TokenPiece> encoded = TokenizerModels.encode(fallback, AlignedText.original("ab"));
    assertEquals(
        List.of("a", "<0x58>", "<0x62>", "<0x59>"),
        encoded.stream().map(TokenPiece::text).toList());
    assertEquals(4, encoded.size());
  }

  private static List<String> bpe(TokenizerDefinition.Bpe model, String text) {
    return TokenizerModels.encode(model, AlignedText.original(text)).stream()
        .map(TokenPiece::text)
        .toList();
  }
}
