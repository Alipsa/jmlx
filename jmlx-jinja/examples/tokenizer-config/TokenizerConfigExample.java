package example;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import se.alipsa.jmlx.jinja.Template;

/** Renders a local tokenizer_config.json chat_template without a JavaScript engine. */
public final class TokenizerConfigExample {
  private static final Pattern CHAT_TEMPLATE =
      Pattern.compile("\"chat_template\"\\s*:\\s*\"((?:\\\\.|[^\"\\\\])*)\"");

  private TokenizerConfigExample() {}

  public static void main(String[] args) throws IOException {
    var config = args.length == 0 ? Path.of("tokenizer_config.json") : Path.of(args[0]);
    var template = Template.parse(chatTemplate(Files.readString(config)));
    System.out.print(
        template.render(
            Map.of(
                "messages",
                List.of(Map.of("role", "user", "content", "Hello")),
                "add_generation_prompt",
                true)));
  }

  static String chatTemplate(String tokenizerConfig) {
    Matcher match = CHAT_TEMPLATE.matcher(tokenizerConfig);
    if (!match.find())
      throw new IllegalArgumentException("tokenizer_config.json has no string chat_template");
    return unescapeJsonString(match.group(1));
  }

  private static String unescapeJsonString(String value) {
    StringBuilder result = new StringBuilder();
    for (int index = 0; index < value.length(); index++) {
      char character = value.charAt(index);
      if (character != '\\') {
        result.append(character);
        continue;
      }
      if (++index == value.length())
        throw new IllegalArgumentException("Invalid JSON string escape");
      switch (value.charAt(index)) {
        case '"' -> result.append('"');
        case '\\' -> result.append('\\');
        case '/' -> result.append('/');
        case 'b' -> result.append('\b');
        case 'f' -> result.append('\f');
        case 'n' -> result.append('\n');
        case 'r' -> result.append('\r');
        case 't' -> result.append('\t');
        default -> throw new IllegalArgumentException("Unsupported JSON string escape");
      }
    }
    return result.toString();
  }
}
