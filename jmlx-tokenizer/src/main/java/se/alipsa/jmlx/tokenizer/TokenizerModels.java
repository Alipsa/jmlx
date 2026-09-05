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
      String symbol = unit.value();
      if (index > 0) {
        symbol = model.continuingSubwordPrefix() + symbol;
      }
      if (index + 1 == input.units().size()) {
        symbol += model.endOfWordSuffix();
      }
      nodes.add(new BpeNode(symbol, new TokenOffset(unit.startByte(), unit.endByte())));
    }
    for (int index = 0; index < nodes.size(); index++) {
      nodes.get(index).previous = index - 1;
      nodes.get(index).next = index + 1 < nodes.size() ? index + 1 : -1;
    }
    PriorityQueue<BpeCandidate> candidates =
        new PriorityQueue<>(
            Comparator.comparingInt(BpeCandidate::rank).thenComparingInt(BpeCandidate::left));
    for (int index = 0; index + 1 < nodes.size(); index++) {
      addCandidate(model, nodes, candidates, index, index + 1);
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
      left.offset = new TokenOffset(left.offset.startByte(), right.offset.endByte());
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
    List<TokenPiece> output = new ArrayList<>();
    for (int index = 0; index >= 0; index = nodes.get(index).next) {
      BpeNode node = nodes.get(index);
      appendBpeSymbol(model, node.symbol, node.offset, output);
    }
    return output;
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

  private static void appendBpeSymbol(
      TokenizerDefinition.Bpe model, String symbol, TokenOffset offset, List<TokenPiece> output) {
    if (model.vocab().containsKey(symbol)) {
      output.add(new TokenPiece(symbol, offset));
      return;
    }
    if (model.byteFallback()) {
      List<String> fallback = new ArrayList<>();
      for (byte value : symbol.getBytes(StandardCharsets.UTF_8)) {
        String token = String.format("<0x%02X>", value & 0xff);
        if (!model.vocab().containsKey(token)) {
          fallback.clear();
          break;
        }
        fallback.add(token);
      }
      if (!fallback.isEmpty()) {
        fallback.forEach(token -> output.add(new TokenPiece(token, offset)));
        return;
      }
    }
    if (model.unknownToken() != null) {
      if (model.fuseUnknown()
          && !output.isEmpty()
          && output.getLast().text().equals(model.unknownToken())) {
        TokenPiece previous = output.removeLast();
        output.add(
            new TokenPiece(
                previous.text(), new TokenOffset(previous.offset().startByte(), offset.endByte())));
      } else {
        output.add(new TokenPiece(model.unknownToken(), offset));
      }
      return;
    }
    throw new TokenizerException(
        "TokenizerModels: BPE symbol '" + symbol + "' has no vocabulary entry");
  }

  private static List<TokenPiece> unigram(TokenizerDefinition.Unigram model, AlignedText input) {
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
    TrieNode trie = unigramTrie(model);
    for (int start = 0; start < size; start++) {
      if (!Double.isFinite(best[start])) {
        continue;
      }
      TrieNode node = trie;
      boolean found = false;
      for (int end = start + 1; end <= size; end++) {
        int codePoint = input.units().get(end - 1).value().codePointAt(0);
        node = node.children.get(codePoint);
        if (node == null) {
          break;
        }
        if (node.tokenId >= 0) {
          found = true;
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
      if (!found) {
        double score = best[start] + model.scores().get(model.unknownId());
        if (score > best[start + 1]) {
          best[start + 1] = score;
          previous[start + 1] = start;
          tokenIds[start + 1] = model.unknownId();
        }
      }
    }
    List<TokenPiece> reversed = new ArrayList<>();
    for (int end = size; end > 0; end = previous[end]) {
      int start = previous[end];
      int id = tokenIds[end];
      String token = model.tokens().get(id);
      TokenOffset offset =
          new TokenOffset(
              input.units().get(start).startByte(), input.units().get(end - 1).endByte());
      if (id == model.unknownId() && model.byteFallback()) {
        String value = join(input.units().subList(start, end));
        List<TokenPiece> bytes = byteFallback(model.vocab(), value, offset);
        if (bytes != null) {
          for (int index = bytes.size() - 1; index >= 0; index--) {
            reversed.add(bytes.get(index));
          }
        } else {
          reversed.add(new TokenPiece(token, offset));
        }
      } else {
        reversed.add(new TokenPiece(token, offset));
      }
    }
    List<TokenPiece> result = new ArrayList<>(reversed.size());
    for (int index = reversed.size() - 1; index >= 0; index--) {
      result.add(reversed.get(index));
    }
    return result;
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
    private TokenOffset offset;
    private int previous;
    private int next;
    private int version;
    private boolean live = true;

    private BpeNode(String symbol, TokenOffset offset) {
      this.symbol = symbol;
      this.offset = offset;
    }
  }

  private static final class TrieNode {
    private final Map<Integer, TrieNode> children = new HashMap<>();
    private int tokenId = -1;
  }

  private record BpeCandidate(int rank, int left, int right, int leftVersion, int rightVersion) {}
}
