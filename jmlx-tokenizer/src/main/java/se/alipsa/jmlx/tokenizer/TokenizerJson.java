package se.alipsa.jmlx.tokenizer;

import java.util.List;

/** The parsed, byte-level-BPE-scoped contents of a {@code tokenizer.json} file. */
public record TokenizerJson(
    NormalizerKind normalizer,
    PreTokenizerConfig preTokenizer,
    List<PostProcessorStep> postProcessor,
    BpeModelConfig model,
    List<AddedToken> addedTokens) {}
