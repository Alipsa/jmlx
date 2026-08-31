package se.alipsa.jmlx.jinja;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

/** Compares every checked-in pinned-Node formatting golden with the public Java API. */
class FormatDifferentialTest {
  private static final Pattern RECORD =
      Pattern.compile(
          "\\{\\\"name\\\":\\\"(?<name>(?:\\\\.|[^\\\"])*)\\\",\\\"source\\\":\\\"(?<source>(?:\\\\.|[^\\\"])*)\\\",\\\"indent\\\":\\{(?:(?:\\\"default\\\":true)|(?:\\\"number\\\":(?<number>-?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][+-]?\\d+)?))|(?:\\\"string\\\":\\\"(?<string>(?:\\\\.|[^\\\"])*)\\\"))\\},\\\"roundTrip\\\":\\\"(?<roundTrip>[^\\\"]+)\\\",\\\"formatted\\\":\\\"(?<formatted>(?:\\\\.|[^\\\"])*)\\\",\\\"context\\\":\\\"(?<context>(?:\\\\.|[^\\\"])*)\\\",\\\"originalRendered\\\":(?:null|\\\"(?<originalRendered>(?:\\\\.|[^\\\"])*)\\\"),\\\"reformattedRendered\\\":(?:null|\\\"(?<reformattedRendered>(?:\\\\.|[^\\\"])*)\\\"),\\\"reformattedError\\\":(?:null|\\\"(?<reformattedError>(?:\\\\.|[^\\\"])*)\\\"),\\\"reformattedCategory\\\":(?:null|\\\"(?<reformattedCategory>[^\\\"]+)\\\")\\}");

  @Test
  void matchesEveryPinnedNodeFormatGolden() throws Exception {
    var input = Files.readString(Path.of("src/test/resources/format/upstream-formatted.json"));
    Matcher matcher = RECORD.matcher(input);
    int records = 0;
    while (matcher.find()) {
      String name = unescape(matcher.group("name"));
      String source = unescape(matcher.group("source"));
      String expected = unescape(matcher.group("formatted"));
      var template = Template.parse(source);
      String actual =
          matcher.group("number") != null
              ? template.format(Double.parseDouble(matcher.group("number")))
              : matcher.group("string") != null
                  ? template.format(unescape(matcher.group("string")))
                  : template.format();
      assertEquals(expected, actual, name);
      if (!matcher.group("roundTrip").equals("not-renderable")) {
        var context = context(unescape(matcher.group("context")));
        assertEquals(
            unescape(required(matcher, "originalRendered")),
            template.render(context),
            name + " original render");
        if (matcher.group("roundTrip").equals("reformat-fails")) {
          var failure =
              assertThrows(HfJinjaException.class, () -> Template.parse(actual).render(context));
          assertEquals(
              ErrorCategory.valueOf(required(matcher, "reformattedCategory")),
              failure.category(),
              name + " formatted error category");
        } else
          assertEquals(
              unescape(required(matcher, "reformattedRendered")),
              Template.parse(actual).render(context),
              name + " formatted render");
      }
      records++;
    }
    assertEquals(countNames(input), records, "golden parser must consume every record");
  }

  private static String required(Matcher matcher, String group) {
    var value = matcher.group(group);
    if (value == null)
      throw new AssertionError("Missing " + group + " in renderable golden record");
    return value;
  }

  private static Map<String, ?> context(String json) {
    return switch (json) {
      case "{}" -> Map.of();
      case "{\"a\":true,\"x\":\"x\"}" -> Map.of("a", true, "x", "x");
      case "{\"a\":true}" -> Map.of("a", true);
      case "{\"a\":1}" -> Map.of("a", 1);
      case "{\"a\":false,\"b\":true}" -> Map.of("a", false, "b", true);
      case "{\"a\":1,\"b\":2}" -> Map.of("a", 1, "b", 2);
      case "{\"c\":true}" -> Map.of("c", true);
      case "{\"a\":[0,1,2,3]}" -> Map.of("a", java.util.List.of(0, 1, 2, 3));
      case "{\"message\":{\"content\":\"<think>x</think>answer\"}}" ->
          Map.of("message", Map.of("content", "<think>x</think>answer"));
      default -> throw new AssertionError("Unsupported format-vector context: " + json);
    };
  }

  private static int countNames(String value) {
    return value.split("\\\"name\\\":", -1).length - 1;
  }

  private static String unescape(String value) {
    var out = new StringBuilder();
    for (int i = 0; i < value.length(); i++) {
      char c = value.charAt(i);
      if (c != '\\') out.append(c);
      else {
        if (++i == value.length()) throw new IllegalArgumentException("Incomplete JSON escape");
        char escaped = value.charAt(i);
        out.append(
            switch (escaped) {
              case '/' -> '/';
              case 'b' -> '\b';
              case 'f' -> '\f';
              case 'n' -> '\n';
              case 'r' -> '\r';
              case 't' -> '\t';
              case '\\' -> '\\';
              case '"' -> '"';
              case 'u' -> unicodeEscape(value, i);
              default -> throw new IllegalArgumentException("Unsupported JSON escape: " + escaped);
            });
        if (escaped == 'u') i += 4;
      }
    }
    return out.toString();
  }

  private static char unicodeEscape(String value, int escapeIndex) {
    if (escapeIndex + 4 >= value.length())
      throw new IllegalArgumentException("Incomplete JSON escape");
    return (char) Integer.parseInt(value.substring(escapeIndex + 1, escapeIndex + 5), 16);
  }
}
