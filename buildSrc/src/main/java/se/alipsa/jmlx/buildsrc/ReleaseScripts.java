package se.alipsa.jmlx.buildsrc;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Verifies that every releasable module's top-level {@code release.sh} stays byte-identical.
 *
 * <p>Each module's own copy derives its module name from its own directory (see release.sh's own
 * header comment), so there is nothing module-specific to keep in sync -- any difference between
 * copies is drift, not an intentional variant. The caller passes the exact set of module
 * directories to check (the declared subprojects, resolved from settings.gradle at configuration
 * time), not a filesystem scan of the whole repo root -- this stays honest as modules are added or
 * removed without also sweeping in an unrelated release.sh some other top-level directory (a
 * vendored tree, a tooling dir) might someday contain.
 */
public final class ReleaseScripts {

  private ReleaseScripts() {}

  /**
   * Throws if two or more {@code release.sh} files directly under the given {@code moduleDirs}
   * differ. Directories without a {@code release.sh} are silently skipped.
   */
  public static void assertAllMatch(List<File> moduleDirs) throws IOException {
    List<File> scripts = new ArrayList<>();
    for (File moduleDir : moduleDirs) {
      File script = new File(moduleDir, "release.sh");
      if (script.isFile()) {
        scripts.add(script);
      }
    }
    List<String> nonExecutable = new ArrayList<>();
    for (File script : scripts) {
      if (!script.canExecute()) {
        nonExecutable.add(script.getPath());
      }
    }
    if (!nonExecutable.isEmpty()) {
      throw new IllegalStateException("release.sh scripts must be executable: " + nonExecutable);
    }
    if (scripts.size() < 2) {
      return;
    }
    // Sorted so the "byte-identical to" reference in the failure message is stable across
    // machines instead of depending on filesystem/collection iteration order.
    scripts.sort(Comparator.comparing(File::getPath));
    File first = scripts.get(0);
    byte[] firstBytes = Files.readAllBytes(first.toPath());
    List<String> mismatched = new ArrayList<>();
    for (File script : scripts.subList(1, scripts.size())) {
      if (!Arrays.equals(firstBytes, Files.readAllBytes(script.toPath()))) {
        mismatched.add(script.getPath());
      }
    }
    if (!mismatched.isEmpty()) {
      throw new IllegalStateException(
          "release.sh has drifted -- these must stay byte-identical to "
              + first
              + ": "
              + mismatched);
    }
  }
}
