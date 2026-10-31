package se.alipsa.jmlx.tokenizer;

import java.util.Map;
import java.util.Objects;
import java.util.Set;
import se.alipsa.jmlx.jinja.RenderOptions;

/**
 * Named chat-template rendering options.
 *
 * @param templateName configured template name, or empty for default selection
 * @param addGenerationPrompt whether the template should open an assistant turn
 * @param extraContext additional non-reserved Jinja context such as tools
 * @param renderOptions explicit render settings, for example a fixed clock and zone for
 *     reproducible prompts, or null to follow the system clock and default zone on each render
 */
public record ChatTemplateOptions(
    String templateName,
    boolean addGenerationPrompt,
    Map<String, Object> extraContext,
    RenderOptions renderOptions) {

  private static final Set<String> RESERVED =
      Set.of(
          "messages",
          "add_generation_prompt",
          "bos_token",
          "eos_token",
          "pad_token",
          "unk_token",
          "sep_token",
          "cls_token",
          "mask_token");

  /**
   * Options without explicit render settings; each render follows the system clock and default
   * zone.
   *
   * @param templateName configured template name, or empty for default selection
   * @param addGenerationPrompt whether the template should open an assistant turn
   * @param extraContext additional non-reserved Jinja context such as tools
   */
  public ChatTemplateOptions(
      String templateName, boolean addGenerationPrompt, Map<String, Object> extraContext) {
    this(templateName, addGenerationPrompt, extraContext, null);
  }

  /**
   * Validates and defensively copies the caller context (see {@link
   * ChatContextCopies#copyContext(Map)}): a later caller mutation of a nested value (a tools list,
   * a nested array) cannot change what a render, including a model's later re-render of the same
   * options, sees. The nested maps and lists {@link #extraContext()} returns are therefore
   * unmodifiable copies, and a cyclic structure is rejected here with a clear error. Sets and other
   * collections are not accepted by the renderer as context values and fail at the render boundary.
   * Empty template name means default selection.
   */
  public ChatTemplateOptions {
    templateName = Objects.requireNonNull(templateName, "templateName");
    extraContext =
        ChatContextCopies.copyContext(Objects.requireNonNull(extraContext, "extraContext"));
    for (String key : extraContext.keySet()) {
      if (RESERVED.contains(key)) {
        throw new IllegalArgumentException(
            "ChatTemplateOptions.extraContext contains reserved key '" + key + "'");
      }
    }
  }

  /**
   * Default-template options without extra context or explicit render settings.
   *
   * @param addGenerationPrompt whether the template should open an assistant turn
   * @return default-template options
   */
  public static ChatTemplateOptions defaults(boolean addGenerationPrompt) {
    return new ChatTemplateOptions("", addGenerationPrompt, Map.of());
  }
}
