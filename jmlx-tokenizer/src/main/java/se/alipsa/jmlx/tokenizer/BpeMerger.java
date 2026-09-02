package se.alipsa.jmlx.tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.PriorityQueue;

/** The byte-level BPE merge algorithm: repeatedly applies the lowest-rank adjacent-pair merge. */
public final class BpeMerger {

  private final BpeModelConfig model;

  /**
   * Prepares a BPE model's vocabulary and merge-rank tables for token merging.
   *
   * @param model BPE configuration
   */
  public BpeMerger(BpeModelConfig model) {
    this.model = Objects.requireNonNull(model, "BpeMerger: model must not be null");
  }

  /**
   * Merges one byte-level-encoded pre-token chunk into its final BPE symbol sequence.
   *
   * @param byteLevelWord byte-level-encoded input chunk
   * @return merged BPE symbols
   */
  public List<String> merge(String byteLevelWord) {
    Objects.requireNonNull(byteLevelWord, "BpeMerger.merge: byteLevelWord must not be null");
    if (model.ignoreMerges() && model.vocab().containsKey(byteLevelWord)) {
      return List.of(byteLevelWord);
    }
    List<Node> nodes = new ArrayList<>();
    byteLevelWord
        .codePoints()
        .forEach(cp -> nodes.add(new Node(new String(Character.toChars(cp)))));
    for (int i = 0; i < nodes.size(); i++) {
      Node node = nodes.get(i);
      node.previous = i - 1;
      node.next = i + 1 < nodes.size() ? i + 1 : -1;
    }
    PriorityQueue<Candidate> candidates = new PriorityQueue<>();
    for (int i = 0; i + 1 < nodes.size(); i++) {
      addCandidate(candidates, nodes, i, i + 1);
    }
    while (!candidates.isEmpty()) {
      Candidate candidate = candidates.remove();
      Node left = nodes.get(candidate.left);
      Node right = nodes.get(candidate.right);
      if (!candidate.isCurrent(left, right)) {
        continue;
      }
      left.symbol += right.symbol;
      left.version++;
      left.next = right.next;
      right.live = false;
      if (right.next != -1) {
        nodes.get(right.next).previous = candidate.left;
      }
      if (left.previous != -1) {
        addCandidate(candidates, nodes, left.previous, candidate.left);
      }
      if (left.next != -1) {
        addCandidate(candidates, nodes, candidate.left, left.next);
      }
    }
    List<String> symbols = new ArrayList<>();
    for (int index = nodes.isEmpty() ? -1 : 0; index != -1; index = nodes.get(index).next) {
      symbols.add(nodes.get(index).symbol);
    }
    for (String symbol : symbols) {
      if (!model.vocab().containsKey(symbol)) {
        throw new TokenizerException(
            "BpeMerger.merge: merged symbol '"
                + symbol
                + "' has no vocabulary entry (byte_fallback is assumed false for this port's target"
                + " models — see req/plans/phase5-m2-plan.md)");
      }
    }
    return symbols;
  }

  private void addCandidate(
      PriorityQueue<Candidate> candidates, List<Node> nodes, int left, int right) {
    Node leftNode = nodes.get(left);
    Node rightNode = nodes.get(right);
    Integer rank = model.mergeRank().get(leftNode.symbol + " " + rightNode.symbol);
    if (rank != null) {
      candidates.add(new Candidate(rank, left, right, leftNode.version, rightNode.version));
    }
  }

  private static final class Node {
    private String symbol;
    private int previous;
    private int next;
    private int version;
    private boolean live = true;

    private Node(String symbol) {
      this.symbol = symbol;
    }
  }

  private record Candidate(int rank, int left, int right, int leftVersion, int rightVersion)
      implements Comparable<Candidate> {

    @Override
    public int compareTo(Candidate other) {
      int rankComparison = Integer.compare(rank, other.rank);
      return rankComparison != 0 ? rankComparison : Integer.compare(left, other.left);
    }

    private boolean isCurrent(Node leftNode, Node rightNode) {
      return leftNode.live
          && rightNode.live
          && leftNode.next == right
          && leftNode.version == leftVersion
          && rightNode.version == rightVersion;
    }
  }
}
