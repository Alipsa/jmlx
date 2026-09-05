package se.alipsa.jmlx.buildsrc;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Parsed native pins from the repository's single authoritative bootstrap script. */
public final class MlxPins {

  private static final Pattern ASSIGNMENT = Pattern.compile("(?m)^([A-Z][A-Z0-9_]*)=\"([^\"]+)\"$");
  private static final Pattern REFERENCE = Pattern.compile("\\$\\{([A-Z][A-Z0-9_]*)}");
  private final Map<String, String> values;

  private MlxPins(Map<String, String> values) {
    this.values = Map.copyOf(values);
  }

  /** Reads all quoted uppercase assignments from {@code scripts/bootstrap-native.sh}. */
  public static MlxPins read(Path repositoryRoot) throws IOException {
    String bootstrap =
        Files.readString(
            repositoryRoot.resolve("scripts/bootstrap-native.sh"), StandardCharsets.UTF_8);
    Map<String, String> values = new HashMap<>();
    Matcher matcher = ASSIGNMENT.matcher(bootstrap);
    while (matcher.find()) {
      values.put(matcher.group(1), matcher.group(2));
    }
    return new MlxPins(values);
  }

  /** Returns a required pin, failing with its bootstrap assignment name when absent. */
  public String required(String name) {
    return resolve(name, new HashSet<>());
  }

  private String resolve(String name, Set<String> resolving) {
    String value = values.get(name);
    if (value == null) {
      throw new IllegalArgumentException("scripts/bootstrap-native.sh does not define " + name);
    }
    if (!resolving.add(name)) {
      throw new IllegalArgumentException(
          "scripts/bootstrap-native.sh has a cyclic pin reference involving " + name);
    }
    Matcher matcher = REFERENCE.matcher(value);
    StringBuilder expanded = new StringBuilder();
    while (matcher.find()) {
      matcher.appendReplacement(
          expanded, Matcher.quoteReplacement(resolve(matcher.group(1), resolving)));
    }
    matcher.appendTail(expanded);
    resolving.remove(name);
    return expanded.toString();
  }
}
