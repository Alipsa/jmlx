package se.alipsa.jmlx.tokenizer;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import se.alipsa.hfjinja.HfJinjaException;
import se.alipsa.hfjinja.Template;

/** Renders a Hugging Face {@code chat_template} Jinja string via {@code hfjinja}. */
public final class ChatTemplateRenderer {

  private ChatTemplateRenderer() {}

  /**
   * Renders {@code chatTemplate} against the standard HF chat-template context variables, plus any
   * caller-supplied {@code extraContext} (e.g. {@code tools} for tool-calling templates) merged in
   * underneath the fixed keys below.
   */
  public static String render(
      String chatTemplate,
      List<Map<String, Object>> messages,
      boolean addGenerationPrompt,
      String bosToken,
      String eosToken,
      Map<String, Object> extraContext) {
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
      return Template.parse(chatTemplate).render(context);
    } catch (HfJinjaException e) {
      throw new TokenizerException(
          "ChatTemplateRenderer.render: failed to render chat template", e);
    }
  }
}
