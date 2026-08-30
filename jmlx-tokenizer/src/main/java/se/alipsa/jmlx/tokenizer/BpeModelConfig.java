package se.alipsa.jmlx.tokenizer;

import java.util.Map;

/**
 * {@code tokenizer.json}'s {@code model} object, scoped to the byte-level-BPE fields this port
 * uses.
 */
public record BpeModelConfig(
    Map<String, Integer> vocab, Map<String, Integer> mergeRank, boolean ignoreMerges) {}
