package se.alipsa.jmlx.tokenizer;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

/** Runtime implementations of the supported tokenizer model families. */
final class TokenizerModels {

  private TokenizerModels() {}

  interface Encoder {
    List<TokenPiece> encode(AlignedText input);
  }

  static Encoder prepare(TokenizerDefinition.Model model) {
    if (model instanceof TokenizerDefinition.Unigram unigram) {
      TrieNode trie = unigramTrie(unigram);
      double unknownScore = unigramUnknownScore(unigram);
      return input -> unigram(unigram, input, trie, unknownScore);
    }
    return input -> encode(model, input);
  }

  static List<TokenPiece> encode(TokenizerDefinition.Model model, AlignedText input) {
    return switch (model) {
      case TokenizerDefinition.Bpe bpe -> bpe(bpe, input);
      case TokenizerDefinition.Unigram unigram -> unigram(unigram, input);
      case TokenizerDefinition.WordPiece wordPiece -> wordPiece(wordPiece, input);
    };
  }

  private static List<TokenPiece> bpe(TokenizerDefinition.Bpe model, AlignedText input) {
    if (input.units().isEmpty()) {
      return List.of();
    }
    String complete = input.text();
    if (model.ignoreMerges() && model.vocab().containsKey(complete)) {
      return List.of(new TokenPiece(complete, input.offset()));
    }
    List<BpeNode> nodes = new ArrayList<>();
    for (int index = 0; index < input.units().size(); index++) {
      AlignedText.Unit unit = input.units().get(index);
      int trueLength = unit.value().getBytes(StandardCharsets.UTF_8).length;
      String symbol = unit.value();
      if (index > 0) {
        symbol = model.continuingSubwordPrefix() + symbol;
      }
      if (index + 1 == input.units().size()) {
        symbol += model.endOfWordSuffix();
      }
      nodes.add(new BpeNode(symbol, trueLength));
    }
    for (int index = 0; index < nodes.size(); index++) {
      nodes.get(index).previous = index - 1;
      nodes.get(index).next = index + 1 < nodes.size() ? index + 1 : -1;
    }
    PriorityQueue<BpeCandidate> candidates =
        new PriorityQueue<>(
            Comparator.comparingInt(BpeCandidate::rank).thenComparingInt(BpeCandidate::left));
    if (!model.ignoreMerges()) {
      for (int index = 0; index + 1 < nodes.size(); index++) {
        addCandidate(model, nodes, candidates, index, index + 1);
      }
    }
    while (!candidates.isEmpty()) {
      BpeCandidate candidate = candidates.remove();
      BpeNode left = nodes.get(candidate.left());
      BpeNode right = nodes.get(candidate.right());
      if (!left.live
          || !right.live
          || left.next != candidate.right()
          || left.version != candidate.leftVersion()
          || right.version != candidate.rightVersion()) {
        continue;
      }
      left.symbol += right.symbol;
      left.trueLength += right.trueLength;
      left.version++;
      left.next = right.next;
      right.live = false;
      if (right.next >= 0) {
        nodes.get(right.next).previous = candidate.left();
      }
      if (left.previous >= 0) {
        addCandidate(model, nodes, candidates, left.previous, candidate.left());
      }
      if (left.next >= 0) {
        addCandidate(model, nodes, candidates, candidate.left(), left.next);
      }
    }
    return emitBpeTokens(model, input, nodes);
  }

  /**
   * Mirrors HF's {@code BPE::merge_word} per-character loop plus {@code Word::get_offsets_iter} and
   * {@code PreTokenizedString::into_encoding}'s offset conversion. HF tracks each emitted symbol's
   * length in the *normalized* string only (a vocab hit or merged span uses the true summed byte
   * length of the characters it represents; a byte-fallback token always uses length 1, regardless
   * of its source character's true byte length; a deferred/pending unk uses the summed true length
   * of only the characters that actually became unk, captured at the moment each fails, and is
   * emitted -- i.e. consumes its position -- only when flushed) and only at the very end converts a
   * cumulative normalized-byte position range back to an original offset, via a table mapping each
   * normalized byte to the original character it came from. Because a byte-fallback success does
   * not flush a still-pending unk, the unk's cumulative position (and therefore its final offset)
   * reflects wherever its *flush* lands in this consumption order, not its own character's true
   * position -- producing surprising but oracle-verified offset swaps whenever byte-fallback and a
   * deferred unk interleave in the same word (PR #24 review round 2, finding 2's offset half, on
   * top of its token-order half already fixed above).
   */
  private static List<TokenPiece> emitBpeTokens(
      TokenizerDefinition.Bpe model, AlignedText input, List<BpeNode> nodes) {
    int totalBytes = 0;
    for (AlignedText.Unit unit : input.units()) {
      totalBytes += unit.value().getBytes(StandardCharsets.UTF_8).length;
    }
    int[] alignStart = new int[totalBytes];
    int[] alignEnd = new int[totalBytes];
    int position = 0;
    for (AlignedText.Unit unit : input.units()) {
      int length = unit.value().getBytes(StandardCharsets.UTF_8).length;
      for (int i = 0; i < length; i++) {
        alignStart[position] = unit.startByte();
        alignEnd[position] = unit.endByte();
        position++;
      }
    }

    List<TokenPiece> output = new ArrayList<>();
    int fictionalPos = 0;
    int pendingLength = 0;
    for (int index = 0; index >= 0; index = nodes.get(index).next) {
      BpeNode node = nodes.get(index);
      if (model.vocab().containsKey(node.symbol)) {
        if (pendingLength > 0) {
          output.add(
              new TokenPiece(
                  model.unknownToken(),
                  bpeOffset(alignStart, alignEnd, fictionalPos, pendingLength)));
          fictionalPos += pendingLength;
          pendingLength = 0;
        }
        output.add(
            new TokenPiece(
                node.symbol, bpeOffset(alignStart, alignEnd, fictionalPos, node.trueLength)));
        fictionalPos += node.trueLength;
        continue;
      }
      if (model.byteFallback()) {
        List<String> fallback = new ArrayList<>();
        for (byte value : node.symbol.getBytes(StandardCharsets.UTF_8)) {
          String token = String.format("<0x%02X>", value & 0xff);
          if (!model.vocab().containsKey(token)) {
            fallback.clear();
            break;
          }
          fallback.add(token);
        }
        if (!fallback.isEmpty()) {
          for (String token : fallback) {
            output.add(new TokenPiece(token, bpeOffset(alignStart, alignEnd, fictionalPos, 1)));
            fictionalPos += 1;
          }
          continue;
        }
      }
      if (model.unknownToken() == null) {
        fictionalPos += node.trueLength;
        continue;
      }
      if (pendingLength > 0 && !model.fuseUnknown()) {
        output.add(
            new TokenPiece(
                model.unknownToken(),
                bpeOffset(alignStart, alignEnd, fictionalPos, pendingLength)));
        fictionalPos += pendingLength;
        pendingLength = 0;
      }
      pendingLength += node.trueLength;
    }
    if (pendingLength > 0) {
      output.add(
          new TokenPiece(
              model.unknownToken(), bpeOffset(alignStart, alignEnd, fictionalPos, pendingLength)));
    }
    return output;
  }

  private static TokenOffset bpeOffset(int[] alignStart, int[] alignEnd, int start, int length) {
    int last = alignStart.length - 1;
    if (start > last) {
      return new TokenOffset(alignEnd[last], alignEnd[last]);
    }
    return new TokenOffset(alignStart[start], alignEnd[Math.min(last, start + length - 1)]);
  }

  private static void addCandidate(
      TokenizerDefinition.Bpe model,
      List<BpeNode> nodes,
      PriorityQueue<BpeCandidate> candidates,
      int left,
      int right) {
    BpeNode a = nodes.get(left);
    BpeNode b = nodes.get(right);
    Integer rank = model.mergeRanks().get(a.symbol + " " + b.symbol);
    if (rank != null) {
      candidates.add(new BpeCandidate(rank, left, right, a.version, b.version));
    }
  }

  private static List<TokenPiece> unigram(TokenizerDefinition.Unigram model, AlignedText input) {
    return unigram(model, input, unigramTrie(model), unigramUnknownScore(model));
  }

  private static List<TokenPiece> unigram(
      TokenizerDefinition.Unigram model, AlignedText input, TrieNode trie, double unknownScore) {
    int size = input.units().size();
    if (size == 0) {
      return List.of();
    }
    double[] best = new double[size + 1];
    Arrays.fill(best, Double.NEGATIVE_INFINITY);
    best[0] = 0.0;
    int[] previous = new int[size + 1];
    int[] tokenIds = new int[size + 1];
    Arrays.fill(tokenIds, -1);
    for (int start = 0; start < size; start++) {
      if (!Double.isFinite(best[start])) {
        continue;
      }
      TrieNode node = trie;
      boolean hasSingleCodePointMatch = false;
      for (int end = start + 1; end <= size; end++) {
        int codePoint = input.units().get(end - 1).value().codePointAt(0);
        node = node.children.get(codePoint);
        if (node == null) {
          break;
        }
        if (node.tokenId >= 0) {
          if (end == start + 1) {
            hasSingleCodePointMatch = true;
          }
          int id = node.tokenId;
          double score = best[start] + model.scores().get(id);
          // Start positions are visited left-to-right, so retaining the existing path on an exact
          // score tie prefers the longer earlier piece, matching the reference Unigram lattice.
          if (score > best[end]) {
            best[end] = score;
            previous[end] = start;
            tokenIds[end] = id;
          }
        }
      }
      // Mirrors HF's Unigram::populate_nodes: the unknown-token fallback competes at every
      // position lacking a single-code-point vocabulary match, even when a longer match exists
      // there too, and always scores at minScore - 10.0 (K_UNK_PENALTY) rather than whatever score
      // the vocabulary happens to declare for <unk> -- both keep the lattice fully connected while
      // still yielding to any real match with the same or better score (PR #24 review, finding 3).
      if (!hasSingleCodePointMatch) {
        double score = best[start] + unknownScore;
        if (score > best[start + 1]) {
          best[start + 1] = score;
          previous[start + 1] = start;
          tokenIds[start + 1] = model.unknownId();
        }
      }
    }
    // HF's encode_optimized backtracks the lattice into a Vec<String>, fusing every consecutive
    // run of unk nodes into ONE string first (fuse_unk), and only then does tokenize() attempt
    // byte-fallback -- on that whole fused string as a single all-or-nothing unit, never on the
    // individual lattice nodes that composed it. A run where any single byte lacks a <0xXX> vocab
    // entry falls back to one plain <unk> token covering the entire run, not a byte/unk split
    // (PR #24 review round 2, finding 1, correcting PR #24 review round 1, finding 3's fix, which
    // byte-fell-back each unk lattice node independently).
    List<int[]> spans = new ArrayList<>();
    for (int end = size; end > 0; end = previous[end]) {
      int start = previous[end];
      spans.add(new int[] {start, end, tokenIds[end]});
    }
    List<TokenPiece> result = new ArrayList<>();
    for (int index = spans.size() - 1; index >= 0; ) {
      int[] span = spans.get(index);
      if (span[2] != model.unknownId()) {
        result.add(
            new TokenPiece(
                model.tokens().get(span[2]),
                new TokenOffset(
                    input.units().get(span[0]).startByte(),
                    input.units().get(span[1] - 1).endByte())));
        index--;
        continue;
      }
      int runStart = index;
      while (runStart - 1 >= 0 && spans.get(runStart - 1)[2] == model.unknownId()) {
        runStart--;
      }
      int fusedStart = spans.get(index)[0];
      int fusedEnd = spans.get(runStart)[1];
      TokenOffset offset =
          new TokenOffset(
              input.units().get(fusedStart).startByte(), input.units().get(fusedEnd - 1).endByte());
      String value = join(input.units().subList(fusedStart, fusedEnd));
      List<TokenPiece> bytes =
          model.byteFallback() ? byteFallback(model.vocab(), value, offset) : null;
      if (bytes != null) {
        result.addAll(bytes);
      } else {
        result.add(new TokenPiece(value, offset, model.unknownId(), 0, false));
      }
      index = runStart - 1;
    }
    return result;
  }

  private static final double UNIGRAM_UNKNOWN_PENALTY = 10.0;

  /**
   * The unknown-token lattice score HF's Unigram model actually uses: the vocabulary's own lowest
   * declared score, minus {@code K_UNK_PENALTY} (10.0) -- not whatever score the file declares for
   * the {@code <unk>} entry itself, which HF's own {@code populate_nodes}/{@code encode_optimized}
   * never reads for this purpose (PR #24 review, finding 3).
   */
  private static double unigramUnknownScore(TokenizerDefinition.Unigram model) {
    double minScore = Double.POSITIVE_INFINITY;
    for (double score : model.scores()) {
      minScore = Math.min(minScore, score);
    }
    return minScore - UNIGRAM_UNKNOWN_PENALTY;
  }

  private static TrieNode unigramTrie(TokenizerDefinition.Unigram model) {
    TrieNode root = new TrieNode();
    for (int id = 0; id < model.tokens().size(); id++) {
      TrieNode node = root;
      for (int codePoint : model.tokens().get(id).codePoints().toArray()) {
        node = node.children.computeIfAbsent(codePoint, ignored -> new TrieNode());
      }
      node.tokenId = id;
    }
    return root;
  }

  private static List<TokenPiece> byteFallback(
      Map<String, Integer> vocabulary, String value, TokenOffset offset) {
    List<TokenPiece> result = new ArrayList<>();
    for (byte b : value.getBytes(StandardCharsets.UTF_8)) {
      String token = String.format("<0x%02X>", b & 0xff);
      if (!vocabulary.containsKey(token)) {
        return null;
      }
      result.add(new TokenPiece(token, offset));
    }
    return result;
  }

  private static List<TokenPiece> wordPiece(
      TokenizerDefinition.WordPiece model, AlignedText input) {
    List<AlignedText.Unit> units = input.units();
    if (units.size() > model.maxInputCharsPerWord()) {
      return List.of(new TokenPiece(model.unknownToken(), input.offset()));
    }
    List<TokenPiece> result = new ArrayList<>();
    int start = 0;
    while (start < units.size()) {
      int end = units.size();
      String matched = null;
      while (end > start) {
        String candidate = join(units.subList(start, end));
        if (start > 0) {
          candidate = model.continuingSubwordPrefix() + candidate;
        }
        if (model.vocab().containsKey(candidate)) {
          matched = candidate;
          break;
        }
        end--;
      }
      if (matched == null) {
        return List.of(new TokenPiece(model.unknownToken(), input.offset()));
      }
      result.add(
          new TokenPiece(
              matched,
              new TokenOffset(units.get(start).startByte(), units.get(end - 1).endByte())));
      start = end;
    }
    return result;
  }

  private static String join(List<AlignedText.Unit> units) {
    StringBuilder result = new StringBuilder();
    units.forEach(unit -> result.append(unit.value()));
    return result.toString();
  }

  private static final class BpeNode {
    private String symbol;
    private int trueLength;
    private int previous;
    private int next;
    private int version;
    private boolean live = true;

    private BpeNode(String symbol, int trueLength) {
      this.symbol = symbol;
      this.trueLength = trueLength;
    }
  }

  private static final class TrieNode {
    private final Map<Integer, TrieNode> children = new HashMap<>();
    private int tokenId = -1;
  }

  private record BpeCandidate(int rank, int left, int right, int leftVersion, int rightVersion) {}
}
