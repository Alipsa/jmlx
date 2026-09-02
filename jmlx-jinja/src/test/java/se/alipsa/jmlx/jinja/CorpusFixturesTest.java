package se.alipsa.jmlx.jinja;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.ZoneId;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** Focused schema and JSON coverage for the test-only corpus reader. */
class CorpusFixturesTest {
  @Test
  void parsesNestedJsonEscapesNumbersAndDuplicateKeys() {
    var records =
        CorpusFixtures.readContent(
            "{\"id\":\"first\",\"source\":\"test\",\"template\":\"\\uD83D\\uDE00\\n"
                + "\",\"context\":{\"n\":1e2,\"list\":[true,null]},\"expected\":{\"text\":\"x\"}}\n"
                + "\n"
                + "{\"id\":\"second\",\"id\":\"second-final\",\"source\":\"test\","
                + "\"template\":\"x\",\"context\":{},\"expected\":{\"text\":\"x\"}}\n",
            "synthetic.jsonl");
    assertEquals(1, records.get(0).line());
    assertEquals(3, records.get(1).line());
    assertEquals("second-final", records.get(1).id());
    assertEquals("😀\n", records.getFirst().template());
    assertEquals(100d, ((Number) records.get(0).context().get("n")).doubleValue());
  }

  @Test
  void parsesExactErrorMessageExpectations() {
    var record =
        CorpusFixtures.readContent(
                "{\"id\":\"exact-error\",\"source\":\"test\",\"template\":\"x\",\"context\":{},"
                    + "\"expected\":{\"errorCategory\":\"EXPLICIT_RAISE\",\"errorMessage\":\"x\"}}",
                "synthetic.jsonl")
            .getFirst();
    assertEquals(ErrorCategory.EXPLICIT_RAISE, record.expected().errorCategory());
    assertEquals("x", record.expected().errorMessage());
    var invalid =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                CorpusFixtures.readContent(
                    "{\"id\":\"invalid-error\",\"source\":\"test\",\"template\":\"x\","
                        + "\"context\":{},\"expected\":{\"errorCategory\":\"EXPLICIT_RAISE\","
                        + "\"errorMessage\":1}}",
                    "synthetic.jsonl"));
    assertTrue(invalid.getMessage().contains("errorMessage must be a string"));
  }

  @Test
  void rejectsMalformedSchemaAndUnsupportedGlobalsWithPhysicalLine() {
    var malformed =
        assertThrows(
            IllegalArgumentException.class,
            () -> CorpusFixtures.readContent("\n {" + "}", "bad.jsonl"));
    assertTrue(malformed.getMessage().contains("bad.jsonl:2"));
    var globals =
        assertThrows(
            IllegalArgumentException.class,
            () -> CorpusFixtures.readContent(record("\"globals\":{}"), "globals.jsonl"));
    assertTrue(globals.getMessage().contains("not supported"));
    String invalidUnicode = new String(new char[] {'\\', 'u', '+', '0', '4', '1'});
    assertThrows(
        IllegalArgumentException.class,
        () ->
            CorpusFixtures.readContent(
                "{\"id\":\"unicode\",\"source\":\"test\",\"template\":\""
                    + invalidUnicode
                    + "\",\"context\":{},\"expected\":{\"text\":\"x\"}}",
                "unicode.jsonl"));
    String nonAsciiUnicode = new String(new char[] {'\\', 'u', '٠', '٠', '٤', '1'});
    assertThrows(
        IllegalArgumentException.class,
        () ->
            CorpusFixtures.readContent(
                "{\"id\":\"unicode-non-ascii\",\"source\":\"test\",\"template\":\""
                    + nonAsciiUnicode
                    + "\",\"context\":{},\"expected\":{\"text\":\"x\"}}",
                "unicode.jsonl"));
  }

  @Test
  void validatesTemplateOptions() {
    var record =
        CorpusFixtures.readContent(
                record("\"templateOptions\":{\"trimBlocks\":false,\"lstripBlocks\":false}"),
                "options.jsonl")
            .getFirst();
    assertEquals(Map.of("trimBlocks", false, "lstripBlocks", false), record.templateOptions());
    assertThrows(
        IllegalArgumentException.class,
        () ->
            CorpusFixtures.readContent(
                record("\"templateOptions\":{\"unknown\":true}"), "options.jsonl"));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            CorpusFixtures.readContent(
                record("\"templateOptions\":{\"trimBlocks\":1}"), "options.jsonl"));
  }

  @Test
  void rejectsUnrepresentableTimesAndNonCanonicalZones() {
    for (String instant :
        new String[] {"2000-01-02T03:04Z", "2000-02-30T00:00:00Z", "2000-01-01T24:00:00Z"}) {
      assertThrows(
          IllegalArgumentException.class,
          () ->
              CorpusFixtures.readContent(
                  record("\"instant\":\"" + instant + "\",\"zone\":\"UTC\""), "time.jsonl"));
    }
    assertThrows(
        IllegalArgumentException.class,
        () ->
            CorpusFixtures.readContent(
                record("\"instant\":\"2000-01-02T03:04:05Z\",\"zone\":\"utc\""), "zone.jsonl"));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            CorpusFixtures.readContent(
                record("\"instant\":\"2000-01-02T03:04:05Z\",\"zone\":\"GMT+5\""), "zone.jsonl"));
    assertEquals(
        ZoneId.of("Africa/Abidjan"),
        CorpusFixtures.readContent(
                record("\"instant\":\"2000-01-02T03:04:05Z\",\"zone\":\"Africa/Abidjan\""),
                "zone.jsonl")
            .getFirst()
            .zone());
    assertEquals(
        "2000-01-02T03:04:05.120Z",
        CorpusFixtures.readContent(
                record("\"instant\":\"2000-01-02T03:04:05.12Z\",\"zone\":\"UTC\""), "time.jsonl")
            .getFirst()
            .instant()
            .toString());
  }

  @Test
  void skipsHashOnlyFixturesAndRejectsAllHashOnlyDynamicRuns() {
    var records =
        CorpusFixtures.readContent(
            "{\"id\":\"hash\",\"source\":\"test\",\"templateSha256\":\""
                + "a".repeat(64)
                + "\",\"modelRepo\":\"example/model\",\"modelRevision\":\""
                + "b".repeat(40)
                + "\",\"templatePath\":\"template.jinja\",\"context\":{},"
                + "\"expected\":{\"sha256\":\""
                + "c".repeat(64)
                + "\"}}",
            "hash.jsonl");
    assertFalse(records.getFirst().templateBearing());
    assertThrows(AssertionError.class, () -> CorpusDifferentialTest.dynamicTests(records));
  }

  private static String record(String extra) {
    return "{\"id\":\"id\",\"source\":\"test\",\"template\":\"x\",\"context\":{},"
        + extra
        + ",\"expected\":{\"text\":\"x\"}}";
  }
}
