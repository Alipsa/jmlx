package se.alipsa.jmlx.buildsrc;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

final class JsonParser {
  private final String source;
  private int index;
  private final boolean strict;

  JsonParser(String source) {
    this(source, false);
  }

  JsonParser(String source, boolean strict) {
    this.source = source;
    this.strict = strict;
  }

  Object parse() {
    Object value = value();
    whitespace();
    if (index != source.length()) {
      throw error("Unexpected trailing input");
    }
    return value;
  }

  private Object value() {
    whitespace();
    if (index == source.length()) {
      throw error("Expected JSON value");
    }
    return switch (source.charAt(index)) {
      case '{' -> object();
      case '[' -> array();
      case '"' -> string();
      default -> {
        if (strict) {
          throw error("Expected JSON object, array, or string");
        }
        yield scalar();
      }
    };
  }

  private Map<String, Object> object() {
    expect('{');
    Map<String, Object> result = new HashMap<>();
    whitespace();
    if (consume('}')) {
      return result;
    }
    do {
      whitespace();
      String key = string();
      whitespace();
      expect(':');
      if (result.containsKey(key)) {
        throw error("Duplicate JSON object key: " + key);
      }
      result.put(key, value());
      whitespace();
    } while (consume(','));
    expect('}');
    return result;
  }

  private List<Object> array() {
    expect('[');
    List<Object> result = new ArrayList<>();
    whitespace();
    if (consume(']')) {
      return result;
    }
    do {
      result.add(value());
      whitespace();
    } while (consume(','));
    expect(']');
    return result;
  }

  private String string() {
    expect('"');
    StringBuilder result = new StringBuilder();
    while (index < source.length() && source.charAt(index) != '"') {
      char character = source.charAt(index++);
      if (character == '\\') {
        if (index == source.length()) {
          throw error("Unterminated escape");
        }
        character = source.charAt(index++);
        if (strict && character != '"' && character != '\\' && character != '/') {
          throw error("Only JSON quote, slash, and backslash escapes are supported");
        }
        character =
            switch (character) {
              case '"', '\\', '/' -> character;
              case 'b' -> '\b';
              case 'f' -> '\f';
              case 'n' -> '\n';
              case 'r' -> '\r';
              case 't' -> '\t';
              case 'u' -> unicode();
              default -> throw error("Invalid escape");
            };
      } else if (character < 0x20 && !strict) {
        throw error("Unescaped control character");
      }
      result.append(character);
    }
    expect('"');
    String text = result.toString();
    if (!strict) {
      for (int i = 0; i < text.length(); i++) {
        char c = text.charAt(i);
        if (Character.isHighSurrogate(c)) {
          if (++i == text.length() || !Character.isLowSurrogate(text.charAt(i))) {
            throw error("Unpaired high surrogate");
          }
        } else if (Character.isLowSurrogate(c)) {
          throw error("Unpaired low surrogate");
        }
      }
    }
    return text;
  }

  private char unicode() {
    int value = 0;
    for (int i = 0; i < 4; i++) {
      if (index == source.length()) {
        throw error("Incomplete Unicode escape");
      }
      int digit = Character.digit(source.charAt(index++), 16);
      if (digit < 0) {
        throw error("Invalid Unicode escape");
      }
      value = value * 16 + digit;
    }
    return (char) value;
  }

  private Object scalar() {
    for (String literal : List.of("true", "false", "null")) {
      if (source.startsWith(literal, index)) {
        index += literal.length();
        return switch (literal) {
          case "true" -> Boolean.TRUE;
          case "false" -> Boolean.FALSE;
          default -> null;
        };
      }
    }
    final int start = index;
    consume('-');
    if (!consume('0')) {
      if (index == source.length() || source.charAt(index) < '1' || source.charAt(index) > '9') {
        throw error("Expected JSON number");
      }
      digits();
    }
    boolean decimal = false;
    if (consume('.')) {
      decimal = true;
      requiredDigits();
    }
    if (consume('e') || consume('E')) {
      decimal = true;
      if (!consume('+')) {
        consume('-');
      }
      requiredDigits();
    }
    String number = source.substring(start, index);
    try {
      if (decimal) {
        return new BigDecimal(number);
      }
      return Long.valueOf(number);
    } catch (NumberFormatException e) {
      throw error("Number outside supported range: " + number);
    }
  }

  private void requiredDigits() {
    int start = index;
    digits();
    if (start == index) {
      throw error("Expected digit");
    }
  }

  private void digits() {
    while (index < source.length() && source.charAt(index) >= '0' && source.charAt(index) <= '9') {
      index++;
    }
  }

  private void whitespace() {
    while (index < source.length()
        && (strict
            ? Character.isWhitespace(source.charAt(index))
            : " \t\r\n".indexOf(source.charAt(index)) >= 0)) {
      index++;
    }
  }

  private boolean consume(char expected) {
    if (index < source.length() && source.charAt(index) == expected) {
      index++;
      return true;
    }
    return false;
  }

  private void expect(char expected) {
    if (!consume(expected)) {
      throw error("Expected '" + expected + "'");
    }
  }

  private IllegalArgumentException error(String message) {
    return new IllegalArgumentException(message + " at character " + index);
  }
}
