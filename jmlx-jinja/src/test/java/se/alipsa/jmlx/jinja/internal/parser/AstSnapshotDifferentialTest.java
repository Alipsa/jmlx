package se.alipsa.jmlx.jinja.internal.parser;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.jinja.TemplateOptions;
import se.alipsa.jmlx.jinja.internal.lexer.Lexer;

/** Compares checked-in Node snapshots with the Java parser for each named fixture. */
class AstSnapshotDifferentialTest {
  private static final Pattern FIXTURE =
      Pattern.compile(
          "\\{\\s*\\\"name\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"\\s*,\\s*\\\"source\\\""
              + "\\s*:\\s*\\\"((?:\\\\.|[^\\\"])*)\\\"");

  @Test
  void matchesPinnedNodeParserSnapshots() throws Exception {
    var fixtures = Files.readString(Path.of("src/test/resources/ast-snapshots/fixtures.json"));
    var expected =
        blocks(Files.readString(Path.of("src/test/resources/ast-snapshots/upstream-parsed.txt")));
    var matcher = FIXTURE.matcher(fixtures);
    var actual = new LinkedHashMap<String, String>();
    var fixtureSources = new LinkedHashMap<String, String>();
    while (matcher.find()) {
      var name = matcher.group(1);
      var source = unescape(matcher.group(2));
      fixtureSources.put(name, source);
      actual.put(
          name,
          AstSnapshot.of(
              Parser.parse(
                  Lexer.tokenize(source, TemplateOptions.DEFAULT), TemplateOptions.DEFAULT)));
    }
    assertEquals(expected.keySet(), actual.keySet(), "fixture and snapshot names differ");
    for (var entry : actual.entrySet()) {
      assertEquals(
          expected.get(entry.getKey()),
          entry.getValue(),
          entry.getKey() + " (" + fixtureSources.get(entry.getKey()) + ")");
    }
  }

  private static Map<String, String> blocks(String input) {
    var result = new LinkedHashMap<String, String>();
    String name = null;
    var body = new StringBuilder();
    for (var line : input.split("(?<=\\n)")) {
      if (line.startsWith("=== ")) {
        if (name != null) {
          result.put(name, body.toString());
        }
        name = line.substring(4, line.indexOf(' ', 4));
        body.setLength(0);
      } else {
        body.append(line);
      }
    }
    if (name != null) {
      result.put(name, body.toString());
    }
    return result;
  }

  private static String unescape(String value) {
    var result = new StringBuilder();
    for (var index = 0; index < value.length(); index++) {
      var character = value.charAt(index);
      if (character != '\\') {
        result.append(character);
        continue;
      }
      if (++index == value.length()) {
        throw new IllegalArgumentException("Incomplete JSON escape");
      }
      switch (value.charAt(index)) {
        case '"' -> result.append('"');
        case '\\' -> result.append('\\');
        case '/' -> result.append('/');
        case 'b' -> result.append('\b');
        case 'f' -> result.append('\f');
        case 'n' -> result.append('\n');
        case 'r' -> result.append('\r');
        case 't' -> result.append('\t');
        case 'u' -> {
          if (index + 4 >= value.length()) {
            throw new IllegalArgumentException("Incomplete Unicode escape");
          }
          result.append((char) Integer.parseInt(value.substring(index + 1, index + 5), 16));
          index += 4;
        }
        default -> throw new IllegalArgumentException("Invalid JSON escape");
      }
    }
    return result.toString();
  }
}
