package se.alipsa.jmlx.models;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import se.alipsa.jmlx.tokenizer.ChatTemplateOptions;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.vision.Phi3VGeometryPlan;

/**
 * Pure Phi-3.5-Vision prompt expansion: replaces each image marker in a rendered chat prompt with
 * the image's contiguous sentinel run, in prompt order.
 *
 * <p>The pinned reference (transformers 4.57.6 {@code Phi3VProcessor}) expands at the string level
 * (findings §7.1, {@code req/plans/phase7-3c-reference-findings.md}): it splits the rendered prompt
 * on {@code re.split(r"<\|image_\d+\|>", text)}, tokenizes each chunk separately (a BPE merge never
 * crosses a marker boundary, and every chunk — the first and every chunk after an image block —
 * starts with the per-encoding {@code <s>}), and expands marker {@code i} to {@code
 * num_img_tokens[i-1]} sentinel rows of {@code -i}. The prompt builder strips nothing ({@code
 * offset = 0}), so the ID-level result this class computes is exactly the reference's: {@code
 * chunk0, sentinels1, chunk1, ..., sentinelsN, chunkN}.
 *
 * <p>Where the reference is permissive, this expander is strict (plan WP3): marker indices must be
 * 1-based, distinct, in prompt order and continuous from 1 (the reference accepts duplicates and
 * silently misbinds out-of-order markers, failing later in the vision path or not at all); the
 * marker count must equal the geometry count; and a rendered prompt whose expansion overflows a
 * signed 32-bit integer is rejected before any native work. The literal added token {@code
 * <|image|>} (id 32044 in the pinned artifact) is not a marker ({@link Phi3VPromptTokens}) and is
 * passed through as an ordinary token.
 *
 * <p>This class never touches native tensors or pixels: it is pure text- and ID-space arithmetic
 * with checked length math. The plan it returns is what the model's row-replacement seam and its
 * prompt-position accounting consume.
 */
final class Phi3VPromptExpander {

  private Phi3VPromptExpander() {}

  /**
   * Expands one chat message list against the given images, in prompt order.
   *
   * <p>Each message's content is flattened to the plain string the artifact's chat template
   * string-concatenates: a text part keeps its text, and an image part becomes its 1-based marker,
   * in order of appearance across all messages. The flattened messages are rendered with the
   * tokenizer's configured chat template, then delegated to {@link #expandRendered(HfTokenizer,
   * String, List)}.
   *
   * @param tokenizer the tokenizer bundle; its configured chat template must be the artifact's
   * @param messages ordered role/content messages; content is text or a list of text/image parts;
   *     null entries are rejected
   * @param options the chat template options (generation-prompt flag, extra context); null is
   *     rejected
   * @param geometries one validated geometry per image, in prompt order; null entries are rejected
   * @return the expanded plan for the rendered prompt
   * @throws IllegalArgumentException when a content part is malformed, a text part or plain string
   *     content contains an image marker, the image part/geometry counts differ, or the rendered
   *     prompt violates {@link #expandRendered(HfTokenizer, String, List)}'s marker contract
   */
  static Phi3VPromptPlan expand(
      HfTokenizer tokenizer,
      List<Map<String, Object>> messages,
      ChatTemplateOptions options,
      List<Phi3VGeometryPlan> geometries) {
    Objects.requireNonNull(tokenizer, "tokenizer");
    Objects.requireNonNull(messages, "messages");
    Objects.requireNonNull(options, "options");
    Objects.requireNonNull(geometries, "geometries");
    for (Phi3VGeometryPlan geometry : geometries) {
      Objects.requireNonNull(geometry, "geometries entries");
    }

    List<Map<String, Object>> flatMessages = new ArrayList<>(messages.size());
    int imageParts = 0;
    for (Map<String, Object> message : messages) {
      Objects.requireNonNull(message, "messages entries");
      StringBuilder text = new StringBuilder();
      imageParts = flattenedMessage(text, message, imageParts);
      // Pass every field through (role, name, ...): a template that reads another field must see
      // the same message the request rendered, or the re-rendered prompt silently differs.
      Map<String, Object> flatMessage = new LinkedHashMap<>(message);
      flatMessage.put("content", text.toString());
      flatMessages.add(flatMessage);
    }
    if (imageParts != geometries.size()) {
      throw new IllegalArgumentException(
          imageParts > geometries.size()
              ? "prompt contains "
                  + imageParts
                  + " image parts but only "
                  + geometries.size()
                  + " image(s) were provided"
              : "prompt contains "
                  + imageParts
                  + " image part(s) but "
                  + geometries.size()
                  + " image(s) were provided");
    }
    String rendered = tokenizer.renderChat(flatMessages, options);
    return expandRendered(tokenizer, rendered, geometries);
  }

  /**
   * Expands one rendered Phi-3.5-Vision prompt against the given images, in prompt order.
   *
   * @param tokenizer the tokenizer whose per-chunk encoding (per-encoding {@code <s>} included)
   *     must be byte-exact against the pinned artifact; null is rejected
   * @param renderedText the rendered prompt text, image markers in place; null is rejected
   * @param geometries one validated geometry per image, in prompt order; null entries are rejected
   * @return the expanded plan: the rendered text, the expanded IDs and each image's sentinel
   *     destination
   * @throws IllegalArgumentException when a marker index is not 1-based, duplicates a previous
   *     marker, is out of prompt order, is not continuous from 1, has more than 9 digits, the
   *     marker count differs from the geometry count, or the expanded length overflows a signed
   *     32-bit integer
   */
  static Phi3VPromptPlan expandRendered(
      HfTokenizer tokenizer, String renderedText, List<Phi3VGeometryPlan> geometries) {
    Objects.requireNonNull(tokenizer, "tokenizer");
    Objects.requireNonNull(renderedText, "renderedText");
    Objects.requireNonNull(geometries, "geometries");
    for (Phi3VGeometryPlan geometry : geometries) {
      Objects.requireNonNull(geometry, "geometries entries");
    }

    List<String> chunks = new ArrayList<>();
    List<Integer> markerIndices = new ArrayList<>();
    splitMarkers(renderedText, chunks, markerIndices);
    if (markerIndices.size() != geometries.size()) {
      throw new IllegalArgumentException(
          "prompt contains "
              + markerIndices.size()
              + " image marker(s) but "
              + geometries.size()
              + " image(s) were provided");
    }
    verifyMarkerOrder(markerIndices);

    long length = 0;
    List<List<Integer>> chunkIds = new ArrayList<>(chunks.size());
    for (String chunk : chunks) {
      // The reference tokenizes each chunk separately; every encoding — including an empty
      // chunk's — starts with the per-encoding <s> (findings §7.1).
      List<Integer> ids = tokenizer.encode(chunk, true);
      chunkIds.add(ids);
      length = Math.addExact(length, ids.size());
    }
    for (Phi3VGeometryPlan geometry : geometries) {
      length = Math.addExact(length, geometry.numImgTokens());
    }
    if (length > Integer.MAX_VALUE) {
      throw new IllegalArgumentException(
          "expanded prompt length " + length + " overflows a signed 32-bit integer");
    }

    int[] expanded = new int[(int) length];
    List<Phi3VImageDestination> destinations = new ArrayList<>(geometries.size());
    int position = 0;
    for (int i = 0; i < chunks.size(); i++) {
      List<Integer> ids = chunkIds.get(i);
      for (int id : ids) {
        expanded[position++] = id;
      }
      if (i < markerIndices.size()) {
        int imageIndex = i + 1; // 1-based prompt order
        int count = geometries.get(i).numImgTokens();
        destinations.add(new Phi3VImageDestination(position, count));
        Arrays.fill(expanded, position, position + count, Phi3VPromptTokens.sentinel(imageIndex));
        position += count;
      }
    }
    Phi3VPromptPlan plan = new Phi3VPromptPlan(renderedText, expanded, List.copyOf(destinations));
    verifySentinelDestinations(plan);
    return plan;
  }

  /**
   * Splits {@code renderedText} on the marker pattern, appending the text chunks to {@code chunks}
   * and each marker's index to {@code markerIndices}, mirroring the reference's {@code
   * re.split(r"<\|image_\d+\|>", text)}.
   */
  private static void splitMarkers(
      String renderedText, List<String> chunks, List<Integer> markerIndices) {
    Matcher matcher = Phi3VPromptTokens.MARKER_PATTERN.matcher(renderedText);
    int lastEnd = 0;
    while (matcher.find()) {
      chunks.add(renderedText.substring(lastEnd, matcher.start()));
      String digits = matcher.group(1);
      if (digits.length() > 9) {
        throw new IllegalArgumentException(
            "marker index has more than 9 digits: " + digits.length());
      }
      markerIndices.add(Integer.parseInt(digits));
      lastEnd = matcher.end();
    }
    chunks.add(renderedText.substring(lastEnd));
  }

  /**
   * Verifies the marker-index sequence is exactly {@code 1, 2, ..., n}: 1-based, no duplicates, in
   * prompt order and continuous. The reference accepts duplicates and out-of-order markers
   * (misbinding them silently); the expander rejects both.
   */
  private static void verifyMarkerOrder(List<Integer> markerIndices) {
    int expected = 1;
    int previous = -1;
    for (int i = 0; i < markerIndices.size(); i++) {
      int actual = markerIndices.get(i);
      if (actual < 1) {
        throw new IllegalArgumentException("marker index " + actual + " is not 1-based");
      }
      if (actual != expected) {
        String reason;
        if (actual == previous) {
          reason = "duplicates the previous marker";
        } else if (actual < expected) {
          reason = "is out of prompt order (expected " + expected + ")";
        } else {
          reason = "skips " + expected + " (marker indices must be continuous from 1)";
        }
        throw new IllegalArgumentException(
            "marker index " + actual + " at position " + i + " " + reason);
      }
      expected++;
      previous = actual;
    }
  }

  /**
   * The sentinel invariant the plan cannot check on its own: every destination range holds its
   * image's sentinel, and no sentinel row is left unplanned.
   *
   * @param plan the plan to verify
   * @throws IllegalStateException when a destination holds another ID or the expanded prompt holds
   *     more sentinel rows than planned
   */
  static void verifySentinelDestinations(Phi3VPromptPlan plan) {
    int[] expanded = plan.expandedIds();
    int sentinelRows = 0;
    for (int id : expanded) {
      if (id < 0) {
        sentinelRows++;
      }
    }
    if (sentinelRows != plan.totalSentinelRows()) {
      throw new IllegalStateException(
          "expanded prompt holds "
              + sentinelRows
              + " sentinel rows but "
              + plan.totalSentinelRows()
              + " sentinel rows are planned");
    }
    List<Phi3VImageDestination> images = plan.images();
    for (int i = 0; i < images.size(); i++) {
      Phi3VImageDestination destination = images.get(i);
      int sentinel = Phi3VPromptTokens.sentinel(i + 1);
      int end = destination.sentinelEnd();
      for (int position = destination.sentinelStart(); position < end; position++) {
        if (expanded[position] != sentinel) {
          throw new IllegalStateException(
              "sentinel destination "
                  + position
                  + " does not hold the image "
                  + (i + 1)
                  + " sentinel");
        }
      }
    }
  }

  /**
   * Flattens one message's content to {@code text} and returns the running image-part count.
   * Mirrors the tokenizer's part-key validation, then adds the marker contract: a marker literal
   * may only be produced by an image part, never written in text.
   */
  private static int flattenedMessage(
      StringBuilder text, Map<String, Object> message, int imageParts) {
    Object content = message.get("content");
    if (content instanceof String contentText) {
      rejectMarkerInText(contentText, "plain string content");
      text.append(contentText);
    } else if (content instanceof List<?> parts) {
      if (parts.isEmpty()) {
        throw new IllegalArgumentException("content list must not be empty");
      }
      for (Object part : parts) {
        imageParts = flattenedContentPart(text, part, imageParts);
      }
    } else {
      throw new IllegalArgumentException(
          "message content must be text or a list of text/image parts");
    }
    return imageParts;
  }

  private static int flattenedContentPart(StringBuilder text, Object part, int imageParts) {
    if (!(part instanceof Map<?, ?> raw)) {
      throw new IllegalArgumentException("content parts must be objects");
    }
    Object type = raw.get("type");
    if (!"text".equals(type) && !"image".equals(type)) {
      throw new IllegalArgumentException(
          "unsupported content part type " + type + " (only text and image)");
    }
    for (Object key : raw.keySet()) {
      if (!"type".equals(key)
          && !("text".equals(type) && "text".equals(key))
          && !("image".equals(type) && "image".equals(key))) {
        throw new IllegalArgumentException("content part contains unsupported key '" + key + "'");
      }
    }
    if ("image".equals(type)) {
      imageParts++;
      text.append(Phi3VPromptTokens.marker(imageParts));
      return imageParts;
    }
    Object value = raw.get("text");
    if (!(value instanceof String valueText)) {
      throw new IllegalArgumentException("text content parts require a 'text' string value");
    }
    rejectMarkerInText(valueText, "text content part");
    text.append(valueText);
    return imageParts;
  }

  private static void rejectMarkerInText(String text, String what) {
    if (Phi3VPromptTokens.MARKER_PATTERN.matcher(text).find()) {
      throw new IllegalArgumentException(
          what + " must not contain image markers; image markers are produced by image parts only");
    }
  }
}
