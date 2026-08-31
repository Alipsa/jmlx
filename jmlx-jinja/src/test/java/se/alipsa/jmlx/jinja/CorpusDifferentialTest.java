package se.alipsa.jmlx.jinja;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

/** Runs every checked-in template-bearing corpus record through hfjinja's public API. */
class CorpusDifferentialTest {
  @TestFactory
  Stream<DynamicTest> matchesPinnedNodeCorpus() {
    return dynamicTests(CorpusFixtures.readResource("/corpus/v1.jsonl")).stream();
  }

  static List<DynamicTest> dynamicTests(List<CorpusFixtures.Record> records) {
    var executable = records.stream().filter(CorpusFixtures.Record::templateBearing).toList();
    long expectedExecutions = records.stream().filter(record -> !record.hashOnly()).count();
    if (executable.isEmpty())
      throw new AssertionError("no template-bearing corpus records were executed");
    var tests =
        executable.stream()
            .map(
                record ->
                    dynamicTest(
                        record.id() + " (line " + record.line() + ")", () -> render(record)))
            .toList();
    assertEquals(
        expectedExecutions,
        tests.size(),
        "every non-hash-only record must become one dynamic test");
    return tests;
  }

  static void render(CorpusFixtures.Record record) {
    assertTimeoutPreemptively(Duration.ofSeconds(30), () -> assertOutcome(record));
  }

  static void assertOutcome(CorpusFixtures.Record record) {
    assertOutcome(
        record, Clock.fixed(record.instantOrDefault(), ZoneOffset.UTC), record.zoneOrDefault());
  }

  static void assertOutcome(CorpusFixtures.Record record, Clock clock, ZoneId zone) {
    String label = record.id() + " (corpus line " + record.line() + ")";
    var options = RenderOptions.builder().clock(clock).zoneId(zone).build();
    if (record.expected().errorCategory() == null) {
      try {
        assertEquals(
            record.expected().text(),
            Template.parse(record.template(), templateOptions(record))
                .render(record.context(), options),
            label);
      } catch (HfJinjaException error) {
        throw new AssertionError(
            label + " expected text but got " + error.category() + ": " + error.getMessage(),
            error);
      }
      return;
    }
    HfJinjaException error =
        assertThrows(
            HfJinjaException.class,
            () ->
                Template.parse(record.template(), templateOptions(record))
                    .render(record.context(), options),
            label);
    assertEquals(
        record.expected().errorCategory(),
        error.category(),
        label
            + " expected "
            + record.expected().errorCategory()
            + " but got "
            + error.category()
            + ": "
            + error.getMessage());
    if (record.expected().errorMessage() != null)
      assertEquals(record.expected().errorMessage(), error.getMessage(), label + " error message");
  }

  private static TemplateOptions templateOptions(CorpusFixtures.Record record) {
    return TemplateOptions.builder()
        .trimBlocks(
            record.templateOptions().getOrDefault("trimBlocks", !record.hasTemplateOptions()))
        .lstripBlocks(
            record.templateOptions().getOrDefault("lstripBlocks", !record.hasTemplateOptions()))
        .build();
  }

  @Test
  void rejectsMismatchedOutcomesAndIncorrectDefaultTimeBindings() {
    var text =
        new CorpusFixtures.Record(
            "wrong-text",
            "synthetic",
            "x",
            java.util.Map.of(),
            false,
            java.util.Map.of(),
            null,
            null,
            new CorpusFixtures.Expected("y", null, null),
            false,
            7);
    var textFailure = assertThrows(AssertionError.class, () -> render(text));
    assertTrue(textFailure.getMessage().contains("wrong-text"));

    var successWhenErrorExpected =
        new CorpusFixtures.Record(
            "expected-error",
            "synthetic",
            "x",
            java.util.Map.of(),
            false,
            java.util.Map.of(),
            null,
            null,
            new CorpusFixtures.Expected(null, ErrorCategory.SYNTAX, null),
            false,
            8);
    var errorFailure = assertThrows(AssertionError.class, () -> render(successWhenErrorExpected));
    assertTrue(errorFailure.getMessage().contains("expected-error"));

    var wrongErrorMessage =
        new CorpusFixtures.Record(
            "wrong-error-message",
            "synthetic",
            "{{ raise_exception('actual') }}",
            java.util.Map.of(),
            false,
            java.util.Map.of(),
            null,
            null,
            new CorpusFixtures.Expected(null, ErrorCategory.EXPLICIT_RAISE, "expected"),
            false,
            9);
    var messageFailure = assertThrows(AssertionError.class, () -> render(wrongErrorMessage));
    assertTrue(messageFailure.getMessage().contains("wrong-error-message"));

    var defaultTime =
        new CorpusFixtures.Record(
            "default-time",
            "synthetic",
            "{{ strftime_now('%Y-%m-%d %H:%M') }}",
            java.util.Map.of(),
            false,
            java.util.Map.of(),
            null,
            null,
            new CorpusFixtures.Expected("2000-01-02 03:04", null, null),
            false,
            10);
    assertThrows(
        AssertionError.class,
        () ->
            assertOutcome(
                defaultTime,
                Clock.fixed(Instant.parse("2000-01-02T04:04:05Z"), ZoneOffset.UTC),
                ZoneId.of("UTC")));
    assertThrows(
        AssertionError.class,
        () ->
            assertOutcome(
                defaultTime,
                Clock.fixed(CorpusFixtures.DEFAULT_INSTANT, ZoneOffset.UTC),
                ZoneId.of("Europe/Stockholm")));
  }
}
