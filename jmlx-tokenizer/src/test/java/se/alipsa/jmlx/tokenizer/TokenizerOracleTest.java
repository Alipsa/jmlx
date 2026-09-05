package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

class TokenizerOracleTest {

  private static final ObjectMapper MAPPER = new ObjectMapper();

  @Test
  void componentFamiliesMatchPinnedOracle() throws Exception {
    String rootProperty =
        java.util.Objects.requireNonNull(
            System.getProperty("jmlx.repository.root"),
            "jmlx.repository.root must be set by build.gradle");
    Path fixtures = Path.of(rootProperty, "tools", "tokenizer-oracle", "fixtures");
    JsonNode input = MAPPER.readTree(fixtures.resolve("phase6-2-components.input.json").toFile());
    JsonNode expected =
        MAPPER.readTree(fixtures.resolve("phase6-2-components.expected.json").toFile());
    assertFalse(input.required("fixtures").isEmpty());
    assertEquals(input.required("fixtures").size(), expected.required("fixtures").size());
    for (int fixtureIndex = 0; fixtureIndex < input.required("fixtures").size(); fixtureIndex++) {
      JsonNode fixture = input.required("fixtures").get(fixtureIndex);
      JsonNode expectedFixture = expected.required("fixtures").get(fixtureIndex);
      assertEquals(
          fixture.required("name").asString(), expectedFixture.required("name").asString());
      HfTokenizer tokenizer =
          HfTokenizer.fromFile(fixtures.resolve(fixture.required("tokenizer").asString()));
      assertEquals(fixture.required("cases").size(), expectedFixture.required("cases").size());
      for (int caseIndex = 0; caseIndex < fixture.required("cases").size(); caseIndex++) {
        JsonNode testCase = fixture.required("cases").get(caseIndex);
        JsonNode expectedCase = expectedFixture.required("cases").get(caseIndex);
        String name = testCase.required("name").asString();
        TokenizerEncoding encoding =
            tokenizer.encode(testCase.required("text").asString(), options(testCase));
        assertEquals(ints(expectedCase.required("ids")), encoding.ids(), name);
        assertEquals(ints(expectedCase.required("typeIds")), encoding.typeIds(), name);
        assertEquals(ints(expectedCase.required("attentionMask")), encoding.attentionMask(), name);
        assertEquals(
            ints(expectedCase.required("specialTokensMask")), encoding.specialTokensMask(), name);
        assertEquals(offsets(expectedCase.required("offsets")), encoding.offsets(), name);
        assertEquals(
            expectedCase.required("decoded").asString(),
            tokenizer.decode(encoding.ids(), true),
            name);
        IncrementalTokenDecoder decoder = tokenizer.newIncrementalDecoder(true);
        StringBuilder incremental = new StringBuilder();
        encoding.ids().forEach(id -> incremental.append(decoder.append(id)));
        incremental.append(decoder.finish());
        assertEquals(expectedCase.required("decoded").asString(), incremental.toString(), name);
      }
    }
  }

  private static EncodingOptions options(JsonNode testCase) {
    boolean special = testCase.required("addSpecialTokens").booleanValue();
    Truncation truncation = Truncation.disabled();
    if (testCase.has("truncation")) {
      JsonNode value = testCase.required("truncation");
      truncation =
          new Truncation(
              value.required("max_length").intValue(),
              Direction.valueOf(value.required("direction").asString().toUpperCase()));
    }
    Padding padding = Padding.disabled();
    if (testCase.has("padding")) {
      JsonNode value = testCase.required("padding");
      padding =
          new Padding(
              value.required("length").intValue(),
              Direction.valueOf(value.required("direction").asString().toUpperCase()),
              value.required("pad_id").intValue(),
              value.required("pad_token").asString(),
              value.required("pad_type_id").intValue());
    }
    return new EncodingOptions(special, truncation, padding);
  }

  private static List<Integer> ints(JsonNode values) {
    List<Integer> result = new ArrayList<>();
    values.forEach(value -> result.add(value.intValue()));
    return result;
  }

  private static List<TokenOffset> offsets(JsonNode values) {
    List<TokenOffset> result = new ArrayList<>();
    values.forEach(
        value -> result.add(new TokenOffset(value.get(0).intValue(), value.get(1).intValue())));
    return result;
  }
}
