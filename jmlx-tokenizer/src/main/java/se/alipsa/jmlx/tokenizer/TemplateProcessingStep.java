package se.alipsa.jmlx.tokenizer;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The {@code TemplateProcessing} post-processor step's {@code single} template.
 *
 * @param single ordered template items
 * @param specialTokens special-token definitions by id
 */
public record TemplateProcessingStep(
    List<TemplateItem> single, Map<String, SpecialTokenInfo> specialTokens)
    implements PostProcessorStep {

  /**
   * Defensively copies {@code single}/{@code specialTokens} -- the last of this loader's mutable-
   * collection-holding types to get this treatment. {@code SpecialTokenInfo}, {@code
   * BpeModelConfig}, and {@code TokenizerJson} all gained one already, but {@code TokenizerJson}'s
   * own {@code List.copyOf(postProcessor)} is shallow: it freezes the outer list, not the {@code
   * TemplateProcessingStep} instances inside it, so a caller clearing {@code single} or removing a
   * {@code specialTokens} entry from an already-loaded tokenizer still corrupted {@link
   * PostProcessorApplier#apply} for every subsequent {@code encode} call -- verified through public
   * API only: clearing {@code single} after construction made {@code apply} return an empty
   * sequence instead of the real one, and removing a {@code specialTokens} entry turned a valid
   * template into {@code PostProcessorApplier}'s "references unknown special token" throw (PR #14
   * review round 8, finding 2, closing the gap {@code HfTokenizer}'s own class javadoc and this
   * plan's round-7 amendment both overclaimed as already closed).
   *
   * <p>{@code specialTokens} is copied via {@code Collections.unmodifiableMap(new
   * LinkedHashMap<>(...))}, not {@code Map.copyOf}: the loader builds this map as a {@code
   * LinkedHashMap} specifically to preserve {@code tokenizer.json}'s own {@code special_tokens}
   * declaration order, but {@code Map.copyOf}'s own immutable-map implementation does not preserve
   * insertion order and, worse, randomizes its iteration order per JVM run (a JDK-documented
   * property, not a bug) -- verified directly: the same {@code LinkedHashMap} insertion order
   * produced three different {@code Map.copyOf} iteration orders across four runs. Nothing
   * observable breaks today, since {@code HfTokenizer#requireInternallyConsistentTemplateTokens}
   * rejects template contradictions before iteration order matters to anything, but a future
   * message-content assertion over a multi-entry contradiction (the exact direction {@code
   * TokenizerJsonLoaderTest}'s and {@code HfTokenizerTest}'s round-8 additions push this suite in)
   * would flake on which entry's name lands in the thrown message (PR #14 review round 9, finding
   * 5).
   */
  public TemplateProcessingStep {
    Objects.requireNonNull(single, "TemplateProcessingStep: single must not be null");
    Objects.requireNonNull(specialTokens, "TemplateProcessingStep: specialTokens must not be null");
    single = List.copyOf(single);
    specialTokens = Collections.unmodifiableMap(new LinkedHashMap<>(specialTokens));
    // A single with no Sequence item can never emit the real encoded token sequence under any
    // addSpecialTokens setting -- PostProcessorApplier#applyTemplate rebuilds its whole output
    // purely from single's items, so a missing or SpecialToken-only single (the shape a malformed
    // or truncated tokenizer.json's post_processor.single can take, e.g. an absent/empty "single"
    // array parsing to zero items) silently discards the entire input text with no diagnostic --
    // verified directly: an empty single made encode("low", true) return []. SpecialTokenInfo's
    // own compact constructor is the precedent for validating this kind of vanishing-content
    // invariant in the type itself rather than only at the loader (PR #14 review round 9, finding
    // 2).
    if (single.stream().noneMatch(item -> item instanceof SequenceItem)) {
      throw new TokenizerException(
          "TemplateProcessingStep: single has no Sequence item, so applyTemplate would silently"
              + " discard the entire encoded sequence: "
              + single);
    }
  }
}
