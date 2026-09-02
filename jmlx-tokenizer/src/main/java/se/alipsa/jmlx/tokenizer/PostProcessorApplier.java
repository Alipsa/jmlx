package se.alipsa.jmlx.tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Applies a {@code tokenizer.json} post-processor's steps to an already-encoded token-string list.
 */
public final class PostProcessorApplier {

  private PostProcessorApplier() {}

  /**
   * Applies every step in {@code steps} to {@code tokens} in order.
   *
   * @param steps post-processing steps
   * @param tokens encoded token strings
   * @param addSpecialTokens whether special tokens should be emitted
   * @return processed tokens
   */
  public static List<ResolvedToken> apply(
      List<PostProcessorStep> steps, List<String> tokens, boolean addSpecialTokens) {
    Objects.requireNonNull(steps, "PostProcessorApplier.apply: steps must not be null");
    Objects.requireNonNull(tokens, "PostProcessorApplier.apply: tokens must not be null");
    List<ResolvedToken> result = new ArrayList<>();
    for (String token : tokens) {
      result.add(new ResolvedToken(token, null));
    }
    for (PostProcessorStep step : steps) {
      if (step instanceof TemplateProcessingStep template) {
        result = applyTemplate(template, result, addSpecialTokens);
      }
      // ByteLevelStep is a no-op on the token list (see Findings: it only fixes up offsets, which
      // this port does not track).
    }
    return result;
  }

  private static List<ResolvedToken> applyTemplate(
      TemplateProcessingStep step, List<ResolvedToken> tokens, boolean addSpecialTokens) {
    List<ResolvedToken> out = new ArrayList<>();
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
          List<String> specialTokens = info.tokens();
          List<Integer> specialIds = info.ids();
          // SpecialTokenInfo's own compact constructor guarantees specialIds.size() ==
          // specialTokens.size(), so specialIds.get(i) is always in range here -- no null
          // fallback needed (PR #14 review round 4, finding 9).
          for (int i = 0; i < specialTokens.size(); i++) {
            out.add(new ResolvedToken(specialTokens.get(i), specialIds.get(i)));
          }
        }
      }
    }
    return out;
  }
}
