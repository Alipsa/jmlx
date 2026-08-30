package se.alipsa.jmlx.tokenizer;

import java.util.List;

/** One entry of a {@code TemplateProcessing} step's {@code special_tokens} map. */
public record SpecialTokenInfo(String id, List<Integer> ids, List<String> tokens) {}
