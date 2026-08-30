package se.alipsa.jmlx.tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Applies a {@code tokenizer.json} post-processor's steps to an already-encoded token-string list.
 */
public final class PostProcessorApplier {

  private PostProcessorApplier() {}

  /** Applies every step in {@code steps} to {@code tokens} in order. */
  public static List<String> apply(
      List<PostProcessorStep> steps, List<String> tokens, boolean addSpecialTokens) {
    Objects.requireNonNull(steps, "PostProcessorApplier.apply: steps must not be null");
    Objects.requireNonNull(tokens, "PostProcessorApplier.apply: tokens must not be null");
    List<String> result = tokens;
    for (PostProcessorStep step : steps) {
      if (step instanceof TemplateProcessingStep template) {
        result = applyTemplate(template, result, addSpecialTokens);
      }
      // ByteLevelStep is a no-op on the token list (see Findings: it only fixes up offsets, which
      // this port does not track).
    }
    return result;
  }

  private static List<String> applyTemplate(
      TemplateProcessingStep step, List<String> tokens, boolean addSpecialTokens) {
    List<String> out = new ArrayList<>();
    for (TemplateItem item : step.single()) {
      if (item instanceof SequenceItem) {
        out.addAll(tokens);
      } else if (item instanceof SpecialTokenItem special) {
        if (addSpecialTokens) {
          SpecialTokenInfo info = step.specialTokens().get(special.id());
          if (info == null) {
            throw new TokenizerException(
                "PostProcessorApplier: template references unknown special token '"
                    + special.id()
                    + "'");
          }
          out.addAll(info.tokens());
        }
      }
    }
    return out;
  }
}
