package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * The public entry points of {@link ChatContextCopies}: named null rejection, the array-to-list
 * conversion on the message path, and the copy-once sharing being a property of nested values (not
 * of the top-level container's elements).
 */
class ChatContextCopiesTest {

  @Test
  void nullArgumentsAreRejectedWithANamedParameter() {
    assertEquals(
        "context",
        assertThrows(NullPointerException.class, () -> ChatContextCopies.copyContext(null))
            .getMessage());
    assertEquals(
        "messages",
        assertThrows(NullPointerException.class, () -> ChatContextCopies.copyMessages(null))
            .getMessage());
    List<Map<String, Object>> withNullElement = new ArrayList<>();
    withNullElement.add(null);
    assertEquals(
        "messages[0]",
        assertThrows(
                NullPointerException.class, () -> ChatContextCopies.copyMessages(withNullElement))
            .getMessage());
  }

  @Test
  void copyMessagesTurnsNestedArraysIntoUnmodifiableLists() {
    Map<String, Object> message = new LinkedHashMap<>();
    message.put("role", "user");
    String[] tags = {"a"};
    message.put("content", tags);
    List<Map<String, Object>> copy = ChatContextCopies.copyMessages(List.of(message));
    tags[0] = "sabotage";
    assertEquals(List.of("a"), copy.get(0).get("content"));
    assertThrows(
        UnsupportedOperationException.class,
        () -> ((List<String>) copy.get(0).get("content")).add("b"));
  }

  @Test
  void nestedReferencesAreSharedAndTopLevelMessagesAreCopiedSeparately() {
    // A nested value referenced several times is copied once and shared ...
    Map<String, Object> nested = new LinkedHashMap<>();
    nested.put("k", "v");
    Map<String, Object> message = new LinkedHashMap<>();
    message.put("a", nested);
    message.put("b", nested);
    List<Map<String, Object>> one = ChatContextCopies.copyMessages(List.of(message));
    // Identity, not just equality: the two references hold the one shared copy.
    assertSame(one.get(0).get("a"), one.get(0).get("b"));

    // ... but a message map that is also referenced inside another message is a top-level element
    // of the messages list, which the Javadoc does not promise to share. The assertion below
    // records today's behavior (top-level elements get separate copies), not the contract: if a
    // future change records top-level maps in the copy state and shares them, that is a harmless
    // improvement and this line should be updated to assertSame, not treated as a regression.
    Map<String, Object> a = new LinkedHashMap<>();
    a.put("role", "user");
    Map<String, Object> b = new LinkedHashMap<>();
    b.put("role", "user");
    b.put("ref", a);
    List<Map<String, Object>> two = ChatContextCopies.copyMessages(List.of(a, b));
    assertNotSame(two.get(0), two.get(1).get("ref"));
  }
}
