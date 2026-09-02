package se.alipsa.jmlx.jinja.internal.ast;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class AstInventoryTest {
  private static final Pattern KEY = Pattern.compile("\\\"([A-Za-z0-9_]+)\\\"\\s*:");

  @Test
  void javaNodesCoverEveryUpstreamAstNode() throws Exception {
    var expected = new TreeSet<String>();
    var matcher = KEY.matcher(Files.readString(Path.of("upstream/ast-allowlist.json")));
    while (matcher.find()) {
      expected.add(matcher.group(1));
    }
    expected.removeAll(Set.of("Statement", "Expression", "Literal"));
    var actual = new TreeSet<String>();
    actual.addAll(records(Statement.class));
    actual.addAll(records(Expression.class));
    actual.remove("ObjectEntry");
    assertEquals(expected, actual);
  }

  private static Set<String> records(Class<?> type) {
    return Arrays.stream(type.getDeclaredClasses())
        .filter(Class::isRecord)
        .map(Class::getSimpleName)
        .collect(Collectors.toCollection(TreeSet::new));
  }
}
