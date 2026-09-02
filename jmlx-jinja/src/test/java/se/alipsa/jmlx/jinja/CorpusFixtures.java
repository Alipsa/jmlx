package se.alipsa.jmlx.jinja;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeFormatterBuilder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

/** Strict, test-only reader for the checked-in differential corpus. */
final class CorpusFixtures {
  static final Instant DEFAULT_INSTANT = Instant.parse("2000-01-02T03:04:05Z");
  static final ZoneId DEFAULT_ZONE = ZoneId.of("UTC");

  private static final Set<String> RECORD_KEYS =
      Set.of(
          "id",
          "source",
          "template",
          "templateOptions",
          "templateSha256",
          "modelRepo",
          "modelRevision",
          "templatePath",
          "context",
          "instant",
          "zone",
          "globals",
          "expected");
  private static final Set<String> CATEGORIES =
      Set.of(
          "SYNTAX",
          "UNDEFINED_OR_ACCESS",
          "TYPE",
          "ARITY",
          "VALUE",
          "EXPLICIT_RAISE",
          "HOST_FUNCTION",
          "HOST_CONVERSION",
          "RESOURCE_LIMIT",
          "OUTPUT");
  private static final Pattern INSTANT =
      Pattern.compile("\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(?:\\.(\\d{1,3}))?Z");
  private static final DateTimeFormatter CANONICAL_INSTANT =
      new DateTimeFormatterBuilder().appendInstant(3).toFormatter();
  private static final Pattern IANA_ZONE =
      Pattern.compile("[A-Za-z][A-Za-z_.+-]*(?:/[A-Za-z][A-Za-z_.+-]*)+");
  private static final Pattern SHA256 = Pattern.compile("[0-9a-f]{64}");
  private static final Pattern REVISION = Pattern.compile("[0-9a-f]{40}");
  private static final Set<String> CANONICAL_ZONES = canonicalZones();

  private CorpusFixtures() {}

  static List<Record> readResource(String path) {
    try (InputStream input = CorpusFixtures.class.getResourceAsStream(path)) {
      if (input == null) {
        throw failure(path, "resource not found");
      }
      return readContent(new String(input.readAllBytes(), StandardCharsets.UTF_8), path);
    } catch (IOException error) {
      throw failure(path, "could not read resource", error);
    }
  }

  static List<Record> readContent(String content, String label) {
    var records = new ArrayList<Record>();
    var ids = new java.util.HashSet<String>();
    String[] lines = content.split("\\r?\\n", -1);
    for (int index = 0; index < lines.length; index++) {
      String line = lines[index];
      if (line.isEmpty()) {
        continue;
      }
      int lineNumber = index + 1;
      String recordLabel = label + ":" + lineNumber;
      Object parsed;
      try {
        parsed = new Json(line).parse();
      } catch (IllegalArgumentException error) {
        throw failure(recordLabel, error.getMessage(), error);
      }
      if (!(parsed instanceof Map<?, ?> map)) {
        throw failure(recordLabel, "record must be a JSON object");
      }
      Record record = validate(castMap(map, recordLabel), recordLabel, lineNumber);
      if (!ids.add(record.id())) {
        throw failure(recordLabel, "duplicate id " + record.id());
      }
      records.add(record);
    }
    return List.copyOf(records);
  }

  private static Record validate(Map<String, Object> value, String label, int line) {
    unknownKeys(value, RECORD_KEYS, label);
    final String id = requiredNonBlank(value, "id", label);
    final String source = requiredNonBlank(value, "source", label);
    Object context = value.get("context");
    if (!(context instanceof Map<?, ?>)) {
      throw failure(label, "id, source, and object context are required");
    }
    boolean text = value.get("template") instanceof String;
    boolean hashOnly = value.containsKey("templateSha256");
    if (text == hashOnly) {
      throw failure(label, "provide exactly one of template or templateSha256");
    }
    if (hashOnly) {
      validateHashOnly(value, label);
    }
    if (text && hasAny(value, "modelRepo", "modelRevision", "templatePath")) {
      throw failure(label, "text records must not include hash-only provenance metadata");
    }
    final boolean hasTemplateOptions = value.containsKey("templateOptions");
    final Map<String, Boolean> templateOptions =
        validateTemplateOptions(value.get("templateOptions"), label);
    if (value.containsKey("globals")) {
      throw failure(label, "record globals are not supported by the pinned Template API");
    }
    boolean hasInstant = value.containsKey("instant");
    boolean hasZone = value.containsKey("zone");
    if (hasInstant != hasZone) {
      throw failure(
          label,
          hasInstant ? "instant requires an explicit zone" : "zone requires an explicit instant");
    }
    Instant instant = null;
    ZoneId zone = null;
    if (hasInstant) {
      instant = parseInstant(value.get("instant"), label);
      zone = parseZone(value.get("zone"), label);
    }
    Expected expected = validateExpected(value.get("expected"), hashOnly, label);
    @SuppressWarnings("unchecked")
    Map<String, Object> typedContext = (Map<String, Object>) context;
    return new Record(
        id,
        source,
        text ? (String) value.get("template") : null,
        templateOptions,
        hasTemplateOptions,
        immutableMap(typedContext),
        instant,
        zone,
        expected,
        hashOnly,
        line);
  }

  private static void validateHashOnly(Map<String, Object> value, String label) {
    if (!(value.get("templateSha256") instanceof String hash) || !SHA256.matcher(hash).matches()) {
      throw failure(label, "templateSha256 must be 64 lowercase hex characters");
    }
    if (!nonBlank(value.get("modelRepo"))
        || !(value.get("modelRevision") instanceof String revision
            && REVISION.matcher(revision).matches())
        || !nonBlank(value.get("templatePath"))) {
      throw failure(
          label, "hash-only records require modelRepo, 40-hex modelRevision, and templatePath");
    }
  }

  private static Expected validateExpected(Object raw, boolean hashOnly, String label) {
    if (!(raw instanceof Map<?, ?> map)) {
      throw failure(label, "expected is required");
    }
    Map<String, Object> expected = castMap(map, label);
    boolean exactError =
        expected.size() == 2
            && expected.containsKey("errorCategory")
            && expected.containsKey("errorMessage");
    if (expected.size() != 1 && !exactError) {
      throw failure(label, "expected must have exactly one outcome");
    }
    if (expected.containsKey("text") && !hashOnly && expected.get("text") instanceof String text) {
      return new Expected(text, null, null);
    }
    if (expected.containsKey("errorCategory")
        && expected.get("errorCategory") instanceof String category
        && CATEGORIES.contains(category)) {
      if (exactError && !(expected.get("errorMessage") instanceof String)) {
        throw failure(label, "errorMessage must be a string");
      }
      return new Expected(
          null,
          ErrorCategory.valueOf(category),
          exactError ? (String) expected.get("errorMessage") : null);
    }
    if (expected.containsKey("sha256")
        && hashOnly
        && expected.get("sha256") instanceof String hash
        && SHA256.matcher(hash).matches()) {
      return new Expected(null, null, null);
    }
    throw failure(label, "expected outcome does not match fixture form");
  }

  private static Map<String, Boolean> validateTemplateOptions(Object raw, String label) {
    if (raw == null) {
      return Map.of();
    }
    if (!(raw instanceof Map<?, ?> map)) {
      throw failure(label, "templateOptions must be an object");
    }
    Map<String, Object> options = castMap(map, label);
    unknownKeys(options, Set.of("trimBlocks", "lstripBlocks"), label);
    var result = new LinkedHashMap<String, Boolean>();
    for (var entry : options.entrySet()) {
      if (!(entry.getValue() instanceof Boolean enabled)) {
        throw failure(label, "templateOptions." + entry.getKey() + " must be boolean");
      }
      result.put(entry.getKey(), enabled);
    }
    return Collections.unmodifiableMap(result);
  }

  private static Instant parseInstant(Object value, String label) {
    if (!(value instanceof String instant)) {
      throw failure(label, "instant must be an ISO-8601 instant");
    }
    var matcher = INSTANT.matcher(instant);
    if (!matcher.matches()) {
      throw failure(label, "instant must be an ISO-8601 instant");
    }
    try {
      Instant parsed = Instant.parse(instant);
      String fraction = matcher.group(1);
      int suffixLength = fraction == null ? 1 : fraction.length() + 2;
      String canonical =
          instant.substring(0, instant.length() - suffixLength)
              + "."
              + (fraction == null ? "" : fraction)
              + "000".substring(fraction == null ? 0 : fraction.length())
              + "Z";
      if (!CANONICAL_INSTANT.format(parsed).equals(canonical)) {
        throw failure(label, "instant must be an ISO-8601 instant");
      }
      return parsed;
    } catch (RuntimeException error) {
      throw failure(label, "instant must be an ISO-8601 instant", error);
    }
  }

  private static ZoneId parseZone(Object value, String label) {
    if (!(value instanceof String zone)
        || !(zone.equals("UTC")
            || (IANA_ZONE.matcher(zone).matches() && CANONICAL_ZONES.contains(zone)))) {
      throw failure(label, "zone must be an IANA time-zone identifier");
    }
    try {
      return ZoneId.of(zone);
    } catch (RuntimeException error) {
      throw failure(label, "zone must be an IANA time-zone identifier", error);
    }
  }

  private static void unknownKeys(Map<String, Object> value, Set<String> keys, String label) {
    var unknown = value.keySet().stream().filter(key -> !keys.contains(key)).toList();
    if (!unknown.isEmpty()) {
      throw failure(label, "unknown fields: " + String.join(", ", unknown));
    }
  }

  private static boolean hasAny(Map<String, Object> value, String... keys) {
    for (String key : keys) {
      if (value.containsKey(key)) {
        return true;
      }
    }
    return false;
  }

  private static String requiredNonBlank(Map<String, Object> value, String key, String label) {
    Object field = value.get(key);
    if (!nonBlank(field)) {
      throw failure(label, "id, source, and object context are required");
    }
    return (String) field;
  }

  private static boolean nonBlank(Object value) {
    return value instanceof String text && !text.isEmpty();
  }

  private static Map<String, Object> castMap(Map<?, ?> map, String label) {
    var result = new LinkedHashMap<String, Object>();
    for (var entry : map.entrySet()) {
      if (!(entry.getKey() instanceof String key)) {
        throw failure(label, "object key must be a string");
      }
      result.put(key, entry.getValue());
    }
    return result;
  }

  private static Map<String, Object> immutableMap(Map<String, Object> map) {
    var result = new LinkedHashMap<String, Object>();
    for (var entry : map.entrySet()) {
      result.put(entry.getKey(), immutableValue(entry.getValue()));
    }
    return Collections.unmodifiableMap(result);
  }

  private static Object immutableValue(Object value) {
    if (value instanceof Map<?, ?> map) {
      return immutableMap(castMap(map, "context"));
    }
    if (value instanceof List<?> list) {
      var result = new ArrayList<>();
      for (Object item : list) {
        result.add(immutableValue(item));
      }
      return Collections.unmodifiableList(result);
    }
    return value;
  }

  private static IllegalArgumentException failure(String label, String message) {
    return new IllegalArgumentException(label + ": " + message);
  }

  private static IllegalArgumentException failure(String label, String message, Throwable cause) {
    return new IllegalArgumentException(label + ": " + message, cause);
  }

  private static Set<String> canonicalZones() {
    String configuredPath =
        System.getProperty("jmlx.jinja.canonicalIanaZones", "tools/corpus/iana-time-zones.txt");
    try {
      return Set.copyOf(Files.readAllLines(Path.of(configuredPath), StandardCharsets.UTF_8));
    } catch (IOException error) {
      throw new IllegalStateException(
          "could not read canonical IANA zone list at " + configuredPath, error);
    }
  }

  record Record(
      String id,
      String source,
      String template,
      Map<String, Boolean> templateOptions,
      boolean hasTemplateOptions,
      Map<String, Object> context,
      Instant instant,
      ZoneId zone,
      Expected expected,
      boolean hashOnly,
      int line) {
    boolean templateBearing() {
      return template != null;
    }

    Instant instantOrDefault() {
      return instant == null ? DEFAULT_INSTANT : instant;
    }

    ZoneId zoneOrDefault() {
      return zone == null ? DEFAULT_ZONE : zone;
    }
  }

  record Expected(String text, ErrorCategory errorCategory, String errorMessage) {}

  private static final class Json {
    private final String input;
    private int index;

    Json(String input) {
      this.input = input;
    }

    Object parse() {
      skipWhitespace();
      Object value = value();
      skipWhitespace();
      if (index != input.length()) {
        throw error("trailing data");
      }
      return value;
    }

    private Object value() {
      if (index == input.length()) {
        throw error("unexpected end of JSON");
      }
      return switch (input.charAt(index)) {
        case '{' -> object();
        case '[' -> array();
        case '"' -> string();
        case 't' -> literal("true", Boolean.TRUE);
        case 'f' -> literal("false", Boolean.FALSE);
        case 'n' -> literal("null", null);
        default -> number();
      };
    }

    private Map<String, Object> object() {
      index++;
      var result = new LinkedHashMap<String, Object>();
      skipWhitespace();
      if (take('}')) {
        return result;
      }
      while (true) {
        skipWhitespace();
        if (index == input.length() || input.charAt(index) != '"') {
          throw error("object key must be a string");
        }
        final String key = string();
        skipWhitespace();
        expect(':');
        skipWhitespace();
        result.put(key, value());
        skipWhitespace();
        if (take('}')) {
          return result;
        }
        expect(',');
      }
    }

    private List<Object> array() {
      index++;
      var result = new ArrayList<>();
      skipWhitespace();
      if (take(']')) {
        return result;
      }
      while (true) {
        result.add(value());
        skipWhitespace();
        if (take(']')) {
          return result;
        }
        expect(',');
        skipWhitespace();
      }
    }

    private String string() {
      expect('"');
      var result = new StringBuilder();
      while (index < input.length()) {
        char character = input.charAt(index++);
        if (character == '"') {
          return result.toString();
        }
        if (character < 0x20) {
          throw error("control character in string");
        }
        if (character != '\\') {
          result.append(character);
          continue;
        }
        if (index == input.length()) {
          throw error("incomplete escape");
        }
        char escaped = input.charAt(index++);
        result.append(
            switch (escaped) {
              case '"' -> '"';
              case '\\' -> '\\';
              case '/' -> '/';
              case 'b' -> '\b';
              case 'f' -> '\f';
              case 'n' -> '\n';
              case 'r' -> '\r';
              case 't' -> '\t';
              case 'u' -> unicode();
              default -> throw error("invalid escape");
            });
      }
      throw error("unterminated string");
    }

    private char unicode() {
      if (index + 4 > input.length()) {
        throw error("incomplete unicode escape");
      }
      String hex = input.substring(index, index + 4);
      index += 4;
      int value = 0;
      for (int character = 0; character < hex.length(); character++) {
        int digit = asciiHexDigit(hex.charAt(character));
        if (digit < 0) {
          throw error("invalid unicode escape");
        }
        value = value * 16 + digit;
      }
      return (char) value;
    }

    private int asciiHexDigit(char character) {
      if (character >= '0' && character <= '9') {
        return character - '0';
      }
      char lower = (char) (character | 0x20);
      return lower >= 'a' && lower <= 'f' ? lower - 'a' + 10 : -1;
    }

    private Object literal(String expected, Object value) {
      if (!input.startsWith(expected, index)) {
        throw error("invalid JSON value");
      }
      index += expected.length();
      return value;
    }

    private Double number() {
      final int start = index;
      take('-');
      if (take('0')) {
        // A following digit is deliberately rejected below.
      } else {
        digits();
      }
      if (take('.')) {
        digits();
      }
      if (take('e') || take('E')) {
        take('+');
        take('-');
        digits();
      }
      String token = input.substring(start, index);
      if (token.isEmpty() || token.equals("-")) {
        throw error("invalid JSON value");
      }
      try {
        return Double.valueOf(token);
      } catch (NumberFormatException error) {
        throw error("invalid JSON number");
      }
    }

    private void digits() {
      int start = index;
      while (index < input.length() && Character.isDigit(input.charAt(index))) {
        index++;
      }
      if (start == index) {
        throw error("expected digit");
      }
    }

    private void expect(char expected) {
      if (!take(expected)) {
        throw error("expected '" + expected + "'");
      }
    }

    private boolean take(char expected) {
      if (index < input.length() && input.charAt(index) == expected) {
        index++;
        return true;
      }
      return false;
    }

    private void skipWhitespace() {
      while (index < input.length() && " \t\r\n".indexOf(input.charAt(index)) >= 0) {
        index++;
      }
    }

    private IllegalArgumentException error(String message) {
      return new IllegalArgumentException("invalid JSON: " + message);
    }
  }
}
