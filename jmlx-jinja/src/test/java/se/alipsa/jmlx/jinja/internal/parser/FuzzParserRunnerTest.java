package se.alipsa.jmlx.jinja.internal.parser;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class FuzzParserRunnerTest {
  @Test
  void preservesUnpairedSurrogateUtf16CodeUnits() throws Exception {
    var candidate =
        FuzzParserRunner.Candidate.read(
            "{\"id\":\"x\",\"family\":\"hostile\",\"source\":\"QQAA2EIA\",\"trimBlocks\":true,\"lstripBlocks\":true,\"sourceCodeUnits\":3}");
    String value = candidate.source();
    assertEquals(3, value.length());
    assertEquals('A', value.charAt(0));
    assertEquals('\uD800', value.charAt(1));
    assertEquals('B', value.charAt(2));
    assertEquals("QQAA2EIA", candidate.encodedSource());
  }

  @Test
  void rejectsOddBytesAndIncorrectLength() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            FuzzParserRunner.Candidate.read(
                "{\"id\":\"x\",\"family\":\"hostile\",\"source\":\"QQ==\",\"trimBlocks\":true,\"lstripBlocks\":true,\"sourceCodeUnits\":1}"));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            FuzzParserRunner.Candidate.read(
                "{\"id\":\"x\",\"family\":\"hostile\",\"source\":\"QQA=\",\"trimBlocks\":true,\"lstripBlocks\":true,\"sourceCodeUnits\":2}"));
  }
}
