package se.alipsa.jmlx.tokenizer;

import java.time.Clock;
import java.time.ZoneId;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import se.alipsa.jmlx.jinja.JinjaException;
import se.alipsa.jmlx.jinja.RenderOptions;
import se.alipsa.jmlx.jinja.Template;

/** Renders a Hugging Face {@code chat_template} Jinja string via {@code jmlx-jinja}. */
public final class ChatTemplateRenderer {
  /** Resolves the current default zone for each render, including changes to TimeZone defaults. */
  private static RenderOptions defaultRenderOptions() {
    return withSystemClock(RenderOptions.DEFAULT);
  }

  /**
   * Fills in only the missing pieces of {@code renderOptions} — the clock, the zone, or both — so
   * an options object that changes only another setting (e.g. {@code maxSteps}) keeps the
   * no-options overloads' {@code strftime_now} behavior instead of failing at its first use:
   * jinja's {@code strftime_now} requires both a clock and a zone at first use, and jinja's own
   * option defaults carry neither. Options that already supply both pass through unchanged; a
   * supplied zone with a missing clock derives the clock from that zone, so an explicit zone is
   * still honored; a supplied clock with a missing zone keeps the clock's own zone, so a pinned
   * clock renders the same prompt on every host regardless of the default zone; and the system
   * clock and zone apply only when both are missing. The top-up copies through {@link
   * RenderOptions#toBuilder()} so a field added to that class later is carried over here
   * automatically.
   */
  private static RenderOptions withSystemClock(RenderOptions renderOptions) {
    if (renderOptions.clock().isPresent() && renderOptions.zoneId().isPresent()) {
      return renderOptions;
    }
    ZoneId zone =
        renderOptions
            .zoneId()
            .orElseGet(
                () -> renderOptions.clock().map(Clock::getZone).orElseGet(ZoneId::systemDefault));
    return renderOptions.toBuilder()
        .clock(renderOptions.clock().orElseGet(() -> Clock.system(zone)))
        .zoneId(zone)
        .build();
  }

  private ChatTemplateRenderer() {}

  /**
   * Renders {@code chatTemplate} against the standard HF chat-template context variables, plus any
   * caller-supplied {@code extraContext} (e.g. {@code tools} for tool-calling templates) merged in
   * underneath the fixed keys below. This overload parses on each call; use {@link #parse(String)}
   * and the {@link #render(Template, List, boolean, String, String, Map)} overload in a serving
   * loop.
   *
   * @param chatTemplate template source
   * @param messages chat messages
   * @param addGenerationPrompt whether to add the generation prompt
   * @param bosToken beginning-of-sequence token
   * @param eosToken end-of-sequence token
   * @param extraContext additional template values
   * @return rendered prompt
   */
  public static String render(
      String chatTemplate,
      List<Map<String, Object>> messages,
      boolean addGenerationPrompt,
      String bosToken,
      String eosToken,
      Map<String, Object> extraContext) {
    return render(
        chatTemplate,
        messages,
        addGenerationPrompt,
        bosToken,
        eosToken,
        extraContext,
        defaultRenderOptions());
  }

  /**
   * Renders with explicit options, allowing a fixed clock and zone for reproducible prompts. A
   * render options object that leaves the clock and/or zone unset gets a top-up, so options that
   * change only another setting keep the no-options overloads' {@code strftime_now} behavior: the
   * system clock and zone apply when both are missing, a supplied zone with a missing clock derives
   * the clock from that zone, and a supplied clock without a zone keeps the clock's own zone, so a
   * pinned clock renders the same prompt on every host.
   *
   * @param chatTemplate template source
   * @param messages chat messages
   * @param addGenerationPrompt whether to add the generation prompt
   * @param bosToken beginning-of-sequence token
   * @param eosToken end-of-sequence token
   * @param extraContext additional template values
   * @param renderOptions explicit render settings; a missing clock or zone is topped up with the
   *     system clock and zone, except that a missing zone next to a supplied clock uses that
   *     clock's own zone
   * @return rendered prompt
   */
  public static String render(
      String chatTemplate,
      List<Map<String, Object>> messages,
      boolean addGenerationPrompt,
      String bosToken,
      String eosToken,
      Map<String, Object> extraContext,
      RenderOptions renderOptions) {
    return render(
        parse(chatTemplate),
        messages,
        addGenerationPrompt,
        bosToken,
        eosToken,
        extraContext,
        renderOptions);
  }

  /**
   * Renders a parsed chat template against the standard HF chat-template context variables.
   *
   * @param chatTemplate parsed template
   * @param messages chat messages
   * @param addGenerationPrompt whether to add the generation prompt
   * @param bosToken beginning-of-sequence token
   * @param eosToken end-of-sequence token
   * @param extraContext additional template values
   * @return rendered prompt
   */
  public static String render(
      Template chatTemplate,
      List<Map<String, Object>> messages,
      boolean addGenerationPrompt,
      String bosToken,
      String eosToken,
      Map<String, Object> extraContext) {
    return render(
        chatTemplate,
        messages,
        addGenerationPrompt,
        bosToken,
        eosToken,
        extraContext,
        defaultRenderOptions());
  }

  /**
   * Renders with explicit options, allowing a fixed clock and zone for reproducible prompts. A
   * render options object that leaves the clock and/or zone unset gets a top-up, so options that
   * change only another setting keep the no-options overloads' {@code strftime_now} behavior: the
   * system clock and zone apply when both are missing, a supplied zone with a missing clock derives
   * the clock from that zone, and a supplied clock without a zone keeps the clock's own zone, so a
   * pinned clock renders the same prompt on every host.
   *
   * @param chatTemplate parsed template
   * @param messages chat messages
   * @param addGenerationPrompt whether to add the generation prompt
   * @param bosToken beginning-of-sequence token
   * @param eosToken end-of-sequence token
   * @param extraContext additional template values
   * @param renderOptions explicit render settings; a missing clock or zone is topped up with the
   *     system clock and zone, except that a missing zone next to a supplied clock uses that
   *     clock's own zone
   * @return rendered prompt
   */
  public static String render(
      Template chatTemplate,
      List<Map<String, Object>> messages,
      boolean addGenerationPrompt,
      String bosToken,
      String eosToken,
      Map<String, Object> extraContext,
      RenderOptions renderOptions) {
    Objects.requireNonNull(
        chatTemplate, "ChatTemplateRenderer.render: chatTemplate must not be null");
    Objects.requireNonNull(messages, "ChatTemplateRenderer.render: messages must not be null");
    Objects.requireNonNull(
        extraContext, "ChatTemplateRenderer.render: extraContext must not be null");
    Map<String, Object> context = new HashMap<>(extraContext);
    context.put("messages", messages);
    context.put("add_generation_prompt", addGenerationPrompt);
    context.put("bos_token", bosToken);
    context.put("eos_token", eosToken);
    try {
      return chatTemplate.render(
          context, withSystemClock(Objects.requireNonNull(renderOptions, "renderOptions")));
    } catch (JinjaException e) {
      throw new TokenizerException(
          "ChatTemplateRenderer.render: failed to render chat template", e);
    }
  }

  /**
   * Renders source against a fully assembled Hugging Face template context.
   *
   * @param chatTemplate template source
   * @param context complete immutable render context
   * @return rendered prompt
   */
  public static String render(String chatTemplate, Map<String, Object> context) {
    return render(chatTemplate, context, defaultRenderOptions());
  }

  /**
   * Renders with explicit options, allowing a fixed clock and zone for reproducible prompts. A
   * render options object that leaves the clock and/or zone unset gets a top-up, so options that
   * change only another setting keep the no-options overloads' {@code strftime_now} behavior: the
   * system clock and zone apply when both are missing, a supplied zone with a missing clock derives
   * the clock from that zone, and a supplied clock without a zone keeps the clock's own zone, so a
   * pinned clock renders the same prompt on every host.
   *
   * @param chatTemplate template source
   * @param context complete render context
   * @param renderOptions explicit render settings; a missing clock or zone is topped up with the
   *     system clock and zone, except that a missing zone next to a supplied clock uses that
   *     clock's own zone
   * @return rendered prompt
   */
  public static String render(
      String chatTemplate, Map<String, Object> context, RenderOptions renderOptions) {
    return render(parse(chatTemplate), context, renderOptions);
  }

  /**
   * Renders a parsed template against a fully assembled Hugging Face template context.
   *
   * @param chatTemplate parsed template
   * @param context complete immutable render context
   * @return rendered prompt
   */
  public static String render(Template chatTemplate, Map<String, Object> context) {
    return render(chatTemplate, context, defaultRenderOptions());
  }

  /**
   * Renders with explicit options, allowing a fixed clock and zone for reproducible prompts. A
   * render options object that leaves the clock and/or zone unset gets a top-up, so options that
   * change only another setting keep the no-options overloads' {@code strftime_now} behavior: the
   * system clock and zone apply when both are missing, a supplied zone with a missing clock derives
   * the clock from that zone, and a supplied clock without a zone keeps the clock's own zone, so a
   * pinned clock renders the same prompt on every host.
   *
   * @param chatTemplate parsed template
   * @param context complete render context
   * @param renderOptions explicit render settings; a missing clock or zone is topped up with the
   *     system clock and zone, except that a missing zone next to a supplied clock uses that
   *     clock's own zone
   * @return rendered prompt
   */
  public static String render(
      Template chatTemplate, Map<String, Object> context, RenderOptions renderOptions) {
    Objects.requireNonNull(
        chatTemplate, "ChatTemplateRenderer.render: chatTemplate must not be null");
    Objects.requireNonNull(context, "ChatTemplateRenderer.render: context must not be null");
    try {
      return chatTemplate.render(
          Collections.unmodifiableMap(new HashMap<>(context)),
          withSystemClock(Objects.requireNonNull(renderOptions, "renderOptions")));
    } catch (JinjaException e) {
      throw new TokenizerException(
          "ChatTemplateRenderer.render: failed to render chat template", e);
    }
  }

  /**
   * Parses a chat template for callers to retain and render repeatedly.
   *
   * @param chatTemplate template source
   * @return parsed template
   */
  public static Template parse(String chatTemplate) {
    Objects.requireNonNull(
        chatTemplate, "ChatTemplateRenderer.parse: chatTemplate must not be null");
    try {
      return Template.parse(chatTemplate);
    } catch (JinjaException e) {
      throw new TokenizerException("ChatTemplateRenderer.parse: failed to parse chat template", e);
    }
  }
}
