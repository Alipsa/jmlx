package se.alipsa.jmlx.tokenizer;

import java.util.List;
import java.util.Map;

/** The {@code TemplateProcessing} post-processor step's {@code single} template. */
public record TemplateProcessingStep(
    List<TemplateItem> single, Map<String, SpecialTokenInfo> specialTokens)
    implements PostProcessorStep {}
