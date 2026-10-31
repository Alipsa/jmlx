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

/**
 * Deep-copies chat-template context values and chat messages into the build-time snapshot that a
 * later render — including a model's re-render of the same retained values — must see unchanged.
 */
public final class ChatContextCopies {

  private ChatContextCopies() {}

  /**
   * Deep-copies a chat-template context so a later caller mutation cannot change what a render —
   * including a later re-render of the same retained values — sees.
   *
   * <p>Maps and lists are copied into insertion-order unmodifiable wrappers (nested map keys are
   * preserved as they are — the renderer rejects non-String keys at the render boundary, not here).
   * Arrays, object and primitive alike, are copied into unmodifiable lists: the renderer converts
   * both to the same array value, and an element-wise copy into a concretely typed object array (an
   * {@code ArrayList[]}) would throw {@link ArrayStoreException}, since the elements come back as
   * unmodifiable wrappers. The remaining values (strings, booleans, numbers — {@code Number} also
   * covers mutable types such as {@link AtomicInteger}, which the renderer accepts) are shared by
   * reference. A nested value referenced several times is copied once and shared, matching the
   * renderer's per-render conversion of each host object (the top-level map itself is not part of
   * that sharing); a cycle is rejected with an {@link IllegalArgumentException} rather than a
   * {@link StackOverflowError}.
   *
   * @param context the context to copy
   * @return the copy; maps and lists are unmodifiable, and any array value has become a list
   * @throws NullPointerException if {@code context} is null
   */
  public static Map<String, Object> copyContext(Map<String, ?> context) {
    return copyStringKeyedMap(Objects.requireNonNull(context, "context"), new CopyState());
  }

  /**
   * Deep-copies chat messages with the semantics of {@link #copyContext(Map)}: the message maps and
   * their values become unmodifiable (an array value within them becomes a list), a nested value
   * referenced several times is copied once and shared, and a cycle is rejected with a clear error.
   *
   * @param messages the messages to copy
   * @return the copy; unmodifiable, with unmodifiable message maps
   * @throws NullPointerException if {@code messages} or any of its elements is null
   */
  public static List<Map<String, Object>> copyMessages(List<? extends Map<String, ?>> messages) {
    Objects.requireNonNull(messages, "messages");
    CopyState state = new CopyState();
    List<Map<String, Object>> copy = new ArrayList<>(messages.size());
    int index = 0;
    for (Map<String, ?> message : messages) {
      copy.add(
          copyStringKeyedMap(Objects.requireNonNull(message, "messages[" + index + "]"), state));
      index++;
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
}
