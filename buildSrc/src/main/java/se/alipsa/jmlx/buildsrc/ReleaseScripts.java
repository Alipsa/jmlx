package se.alipsa.jmlx.buildsrc;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Verifies that every releasable module's top-level {@code release.sh} stays byte-identical.
 *
 * <p>Each module's own copy derives its module name from its own directory (see release.sh's own
 * header comment), so there is nothing module-specific to keep in sync -- any difference between
 * copies is drift, not an intentional variant. Discovering the set by scanning for {@code
 * <child>/release.sh} rather than naming the current two modules keeps this honest as modules are
 * added or removed, and keeps every module's own check self-contained (no cross-project task
 * dependency on a shared root-level task).
 */
public final class ReleaseScripts {

  private ReleaseScripts() {}

  /** Throws if two or more {@code <child>/release.sh} files directly under {@code repoRoot} differ. */
  public static void assertAllMatch(File repoRoot) throws IOException {
    File[] children = repoRoot.listFiles(File::isDirectory);
    if (children == null) {
      return;
    }
    List<File> scripts = new ArrayList<>();
    for (File child : children) {
      File script = new File(child, "release.sh");
      if (script.isFile()) {
        scripts.add(script);
      }
    }
    if (scripts.size() < 2) {
      return;
    }
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
