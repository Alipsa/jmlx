package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTimeout;

import java.time.Duration;
import java.util.List;
import org.junit.jupiter.api.Test;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/**
 * {@link NormalizerPipeline} behavior: NFC cost bounds, Bert normalizer clean/accent/control rules,
 * and source-offset preservation for inserted and composed text.
 */
class NormalizerPipelineTest {
  private static final String UNASSIGNED = "\u0378"; // unassigned code point
  private static final String COMBINING_ENCLOSING = "\u0488"; // combining mark
  private static final String COMPAT_IDEOGRAPH = "\uf900"; // compatibility ideograph

  private static final ObjectMapper MAPPER = new ObjectMapper();

  @Test
  void unicodeNormalizationOfLongDecomposedTextStaysBounded() throws Exception {
    JsonNode nfc = json("{\"type\":\"NFC\"}");
    String input = "Cafe\u0301 ".repeat(10_000); // e + combining acute
    assertTimeout(
        Duration.ofSeconds(5),
        () ->
            assertEquals(
                "Café ".repeat(10_000),
                NormalizerPipeline.apply(nfc, AlignedText.original(input)).text()));
  }

  @Test
  void bertCleanAndAccentRulesMatchHf() throws Exception {
    JsonNode unclean =
        json(
            "{\"type\":\"BertNormalizer\",\"clean_text\":false,"
                + "\"handle_chinese_chars\":false,\"lowercase\":false,\"strip_accents\":false}");
    assertEquals(
        "a\tb\nc\u00a0d",
        NormalizerPipeline.apply(unclean, AlignedText.original("a\tb\nc\u00a0d")).text());
    JsonNode clean =
        json(
            "{\"type\":\"BertNormalizer\",\"handle_chinese_chars\":false,"
                + "\"lowercase\":false,\"strip_accents\":false}");
    assertEquals(
        UNASSIGNED + "z",
        NormalizerPipeline.apply(clean, AlignedText.original(UNASSIGNED + "z")).text());
    JsonNode accents =
        json(
            "{\"type\":\"BertNormalizer\",\"handle_chinese_chars\":false,"
                + "\"lowercase\":false,\"strip_accents\":true}");
    assertEquals("कार", NormalizerPipeline.apply(accents, AlignedText.original("कार")).text());
    assertEquals("अः", NormalizerPipeline.apply(accents, AlignedText.original("अः")).text());
    assertEquals(
        "a" + COMBINING_ENCLOSING,
        NormalizerPipeline.apply(accents, AlignedText.original("a" + COMBINING_ENCLOSING)).text());
  }

  @Test
  void insertedAndComposedTextKeepsSourceOffsets() throws Exception {
    AlignedText prepended =
        NormalizerPipeline.apply(
            json("{\"type\":\"Prepend\",\"prepend\":\"_\"}"), AlignedText.original("hi"));
    assertEquals(0, prepended.units().getFirst().startByte());
    assertEquals(1, prepended.units().getFirst().endByte());
    AlignedText byteLevel =
        PreTokenizerPipeline.apply(
                json("{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\"use_regex\":false}"),
                AlignedText.original("hi"))
            .getFirst();
    assertEquals(0, byteLevel.units().getFirst().startByte());
    assertEquals(1, byteLevel.units().getFirst().endByte());
    AlignedText composed =
        NormalizerPipeline.apply(
            json("{\"type\":\"NFC\"}"), AlignedText.original("e\u0301")); // NFD e-acute
    assertEquals(composed.offset(), new TokenOffset(0, 1));
    AlignedText replaced =
        NormalizerPipeline.apply(
            json("{\"type\":\"Replace\",\"pattern\":{\"Regex\":\"\\\\b\"},\"content\":\"|\"}"),
            AlignedText.original("ab cd"));
    AlignedText.Unit inserted =
        replaced.units().stream().filter(u -> u.value().equals("|")).toList().get(2);
    assertEquals(new TokenOffset(2, 3), new TokenOffset(inserted.startByte(), inserted.endByte()));
  }

  @Test
  void emptyReplaceMatchAtSegmentStartKeepsTheSegmentBoundary() throws Exception {
    // The "abc" segment of an original "[T]abc" where an added token owns bytes 0-3: the units
    // carry absolute byte offsets, so an empty match at the segment's char 0 must map to the
    // segment's start boundary, not absolute byte 0 (PR #39 review round 2, finding 1).
    AlignedText segment =
        new AlignedText(
            List.of(
                new AlignedText.Unit("a", 3, 4),
                new AlignedText.Unit("b", 4, 5),
                new AlignedText.Unit("c", 5, 6)));
    AlignedText replaced =
        NormalizerPipeline.apply(
            json("{\"type\":\"Replace\",\"pattern\":{\"Regex\":\"^\"},\"content\":\"X\"}"),
            segment);
    AlignedText.Unit inserted = replaced.units().getFirst();
    assertEquals("X", inserted.value());
    assertEquals(3, inserted.startByte());
    assertEquals(3, inserted.endByte());
    assertEquals("Xabc", replaced.text());
  }

  @Test
  void bertNormalizerRemovesControlsAndSpacesCompatibilityIdeographs() throws Exception {
    JsonNode bert =
        json("{\"type\":\"BertNormalizer\",\"lowercase\":false,\"strip_accents\":false}");
    assertEquals("ab", NormalizerPipeline.apply(bert, AlignedText.original("a\u000bb")).text());
    assertEquals(
        "ab",
        NormalizerPipeline.apply(bert, AlignedText.original("a\ue000b")).text()); // private use
    assertEquals(
        "a " + COMPAT_IDEOGRAPH + " b",
        NormalizerPipeline.apply(bert, AlignedText.original("a" + COMPAT_IDEOGRAPH + "b")).text());
  }

  private static JsonNode json(String value) throws Exception {
    return MAPPER.readTree(value);
  }
}
