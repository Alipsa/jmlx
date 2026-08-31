package se.alipsa.jmlx.jinja.internal.parser;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.regex.Pattern;
import se.alipsa.jmlx.jinja.ErrorCategory;
import se.alipsa.jmlx.jinja.TemplateOptions;
import se.alipsa.jmlx.jinja.TemplateRenderException;
import se.alipsa.jmlx.jinja.TemplateSyntaxException;
import se.alipsa.jmlx.jinja.internal.lexer.Lexer;

/** Persistent stdin/stdout parser runner used only by the external fuzz harness. */
final class FuzzParserRunner {
  private static final Pattern STRING =
      Pattern.compile("\\\"(id|family|source)\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
  private static final Pattern BOOLEAN =
      Pattern.compile("\\\"(trimBlocks|lstripBlocks)\\\"\\s*:\\s*(true|false)");
  private static final Pattern LENGTH = Pattern.compile("\\\"sourceCodeUnits\\\"\\s*:\\s*(\\d+)");

  private FuzzParserRunner() {}

  public static void main(String[] args) throws Exception {
    try (var input = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8))) {
      String line;
      while ((line = input.readLine()) != null) System.out.println(run(line));
    }
  }

  private static String run(String json) {
    String id = null;
    Candidate candidate = null;
    try {
      candidate = Candidate.read(json);
      id = candidate.id;
      var options =
          TemplateOptions.builder()
              .trimBlocks(candidate.trimBlocks)
              .lstripBlocks(candidate.lstripBlocks)
              .build();
      try {
        var ast = Parser.parse(Lexer.tokenize(candidate.source, options), options);
        var ast64 =
            candidate.family.equals("grammar")
                ? Base64.getEncoder()
                    .encodeToString(AstSnapshot.of(ast).getBytes(StandardCharsets.UTF_8))
                : null;
        return response(id, "PARSED", ast64);
      } catch (TemplateSyntaxException exception) {
        return response(id, "SYNTAX", null, exception.getMessage());
      } catch (TemplateRenderException exception) {
        if (exception.category() == ErrorCategory.RESOURCE_LIMIT)
          return response(id, "LIMIT", null);
        throw exception;
      }
    } catch (StackOverflowError error) {
      throw new AssertionError("Stack overflow in parser runner for " + id, error);
    } catch (Throwable error) {
      return response(
          id,
          "HARNESS",
          null,
          error.getClass().getName()
              + ": "
              + error.getMessage()
              + " sourceUtf16le="
              + (candidate == null ? "<unavailable>" : candidate.encodedSource));
    }
  }

  private static String response(String id, String result, String ast) {
    return response(id, result, ast, null);
  }

  private static String response(String id, String result, String ast, String message) {
    return "{\"id\":"
        + (id == null ? "null" : quote(id))
        + ",\"result\":"
        + quote(result)
        + (ast == null ? "" : ",\"ast\":" + quote(ast))
        + (message == null ? "" : ",\"message\":" + quote(message))
        + "}";
  }

  private static String quote(String value) {
    var result = new StringBuilder("\"");
    for (var index = 0; index < value.length(); index++) {
      var character = value.charAt(index);
      if (character == '\\') result.append("\\\\");
      else if (character == '"') result.append("\\\"");
      else if (character >= 0x20 && character <= 0x7e) result.append(character);
      else result.append(String.format("\\u%04x", (int) character));
    }
    return result.append('"').toString();
  }

  record Candidate(
      String id,
      String family,
      String source,
      String encodedSource,
      boolean trimBlocks,
      boolean lstripBlocks) {
    static Candidate read(String json) {
      String id = null, family = null, encoded = null;
      var strings = STRING.matcher(json);
      while (strings.find())
        switch (strings.group(1)) {
          case "id" -> id = strings.group(2);
          case "family" -> family = strings.group(2);
          case "source" -> encoded = strings.group(2);
          default -> {}
        }
      Boolean trim = null, lstrip = null;
      var booleans = BOOLEAN.matcher(json);
      while (booleans.find())
        if (booleans.group(1).equals("trimBlocks")) trim = Boolean.valueOf(booleans.group(2));
        else lstrip = Boolean.valueOf(booleans.group(2));
      var length = LENGTH.matcher(json);
      if (id == null
          || family == null
          || encoded == null
          || trim == null
          || lstrip == null
          || !length.find()) throw new IllegalArgumentException("Malformed candidate");
      var bytes = Base64.getDecoder().decode(encoded);
      if ((bytes.length & 1) != 0)
        throw new IllegalArgumentException("HARNESS odd UTF-16LE byte count");
      var chars = new char[bytes.length / 2];
      for (var index = 0; index < chars.length; index++)
        chars[index] = (char) ((bytes[2 * index] & 0xff) | ((bytes[2 * index + 1] & 0xff) << 8));
      var source = new String(chars);
      if (source.length() != Integer.parseInt(length.group(1)))
        throw new IllegalArgumentException("HARNESS source length differs");
      return new Candidate(id, family, source, encoded, trim, lstrip);
    }
  }
}
