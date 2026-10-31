package se.alipsa.jmlx.tokenizer;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
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
   * Validates and defensively copies the caller context (see {@link #copyContext(Map)}): a later
   * caller mutation of a nested value (a tools list, a nested array) cannot change what a render,
   * including a model's later re-render of the same options, sees. The nested maps and lists {@link
   * #extraContext()} returns are therefore unmodifiable copies, and a cyclic structure is rejected
   * here with a clear error. Sets and other collections are not accepted by the renderer as context
   * values and fail at the render boundary. Empty template name means default selection.
   */
  public ChatTemplateOptions {
    templateName = Objects.requireNonNull(templateName, "templateName");
    extraContext = copyContext(Objects.requireNonNull(extraContext, "extraContext"));
    for (String key : extraContext.keySet()) {
      if (RESERVED.contains(key)) {
        throw new IllegalArgumentException(
            "ChatTemplateOptions.extraContext contains reserved key '" + key + "'");
      }
    }
  }

  /**
   * Deep-copies a chat-template context so a later caller mutation cannot change what a render —
   * including a model's later re-render of the same options — sees. Shared by this record's compact
   * constructor and (through {@link #copyMessages(List)}) by {@code GenerationRequest.chat}'s
   * message snapshot (jmlx-models), so the two snapshots cannot drift apart.
   *
   * <p>Maps and lists are copied into insertion-order unmodifiable wrappers (nested map keys are
   * preserved as they are — the renderer rejects non-String keys at the render boundary, not here).
   * Arrays, object and primitive alike, are copied into unmodifiable lists: the renderer converts
   * both to the same array value, and an element-wise copy into a concretely typed object array (an
   * {@code ArrayList[]}) would throw {@link ArrayStoreException}, since the elements come back as
   * unmodifiable wrappers. The remaining values (strings, booleans, numbers — {@code Number} also
   * covers mutable types such as {@link AtomicInteger}, which the renderer accepts) are shared by
   * reference. A value referenced several times is copied once and shared, matching the renderer's
   * per-render conversion of each host object; a cycle is rejected with an {@link
   * IllegalArgumentException} rather than a {@link StackOverflowError}.
   *
   * @param context the context to copy
   * @return the copy; maps and lists are unmodifiable, and any array value has become a list
   */
  public static Map<String, Object> copyContext(Map<String, ?> context) {
    return copyStringKeyedMap(context, new CopyState());
  }

  /**
   * Deep-copies chat messages with the semantics of {@link #copyContext(Map)}: the message maps and
   * their values become unmodifiable (an array value within them becomes a list), a value
   * referenced several times is copied once and shared, and a cycle is rejected with a clear error.
   *
   * @param messages the messages to copy
   * @return the copy; unmodifiable, with unmodifiable message maps
   */
  public static List<Map<String, Object>> copyMessages(List<? extends Map<String, ?>> messages) {
    CopyState state = new CopyState();
    List<Map<String, Object>> copy = new ArrayList<>(messages.size());
    for (Map<String, ?> message : messages) {
      copy.add(copyStringKeyedMap(message, state));
    }
    return Collections.unmodifiableList(copy);
  }

  /** The per-call state of one {@link #copyContext(Map)}/{@link #copyMessages(List)} copy. */
  private static final class CopyState {
    final IdentityHashMap<Object, Object> converted = new IdentityHashMap<>();

    final Set<Object> visiting = Collections.newSetFromMap(new IdentityHashMap<>());
  }

  private static Map<String, Object> copyStringKeyedMap(Map<String, ?> map, CopyState state) {
    Map<String, Object> copy = new LinkedHashMap<>(map.size());
    for (Map.Entry<String, ?> entry : map.entrySet()) {
      copy.put(entry.getKey(), copyValue(entry.getValue(), state));
    }
    return Collections.unmodifiableMap(copy);
  }

  private static Object copyValue(Object value, CopyState state) {
    if (value == null
        || !(value instanceof Map<?, ?>
            || value instanceof List<?>
            || value.getClass().isArray())) {
      return value;
    }
    if (state.visiting.contains(value)) {
      throw new IllegalArgumentException("chat context value contains a cycle");
    }
    Object existing = state.converted.get(value);
    if (existing != null) {
      return existing;
    }
    state.visiting.add(value);
    try {
      Object copy;
      if (value instanceof Map<?, ?> map) {
        Map<Object, Object> typed = new LinkedHashMap<>(map.size());
        for (Map.Entry<?, ?> entry : map.entrySet()) {
          typed.put(entry.getKey(), copyValue(entry.getValue(), state));
        }
        copy = Collections.unmodifiableMap(typed);
      } else {
        int length = value instanceof List<?> list ? list.size() : Array.getLength(value);
        List<Object> elements = new ArrayList<>(length);
        if (value instanceof List<?> list) {
          for (Object element : list) {
            elements.add(copyValue(element, state));
          }
        } else {
          for (int i = 0; i < length; i++) {
            elements.add(copyValue(Array.get(value, i), state));
          }
        }
        copy = Collections.unmodifiableList(elements);
      }
      state.converted.put(value, copy);
      return copy;
    } finally {
      state.visiting.remove(value);
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
