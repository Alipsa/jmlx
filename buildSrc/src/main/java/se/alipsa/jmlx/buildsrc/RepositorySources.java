package se.alipsa.jmlx.buildsrc;

import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/** Complete, immutable view of handwritten Java sources relevant to inventory validation. */
final class RepositorySources {

  private static final String GENERATED_DIRECTORY = "jmlx-ffi/src/main/generated/java";

  private final List<Path> handwrittenJavaSources;
  private final Set<String> testJavaFileNames;

  private RepositorySources(List<Path> handwrittenJavaSources, Set<String> testJavaFileNames) {
    this.handwrittenJavaSources = List.copyOf(handwrittenJavaSources);
    this.testJavaFileNames = Set.copyOf(testJavaFileNames);
  }

  static RepositorySources discover(Path repositoryRoot) throws IOException {
    List<Path> handwrittenJavaSources = new ArrayList<>();
    Files.walkFileTree(
        repositoryRoot,
        new SimpleFileVisitor<>() {
          @Override
          public FileVisitResult preVisitDirectory(Path directory, BasicFileAttributes attributes) {
            return excluded(repositoryRoot, directory)
                ? FileVisitResult.SKIP_SUBTREE
                : FileVisitResult.CONTINUE;
          }

          @Override
          public FileVisitResult visitFile(Path file, BasicFileAttributes attributes) {
            if (file.toString().endsWith(".java")) {
              handwrittenJavaSources.add(file);
            }
            return FileVisitResult.CONTINUE;
          }
        });
    handwrittenJavaSources.sort(Comparator.naturalOrder());
    Set<String> testJavaFileNames =
        handwrittenJavaSources.stream()
            .filter(path -> isTestJavaSource(repositoryRoot.relativize(path)))
            .map(path -> path.getFileName().toString())
            .collect(Collectors.toUnmodifiableSet());
    return new RepositorySources(handwrittenJavaSources, testJavaFileNames);
  }

  List<Path> handwrittenJavaSources() {
    return handwrittenJavaSources;
  }

  Set<String> testJavaFileNames() {
    return testJavaFileNames;
  }

  private static boolean excluded(Path repositoryRoot, Path path) {
    Path relative = repositoryRoot.relativize(path);
    String directoryName = relative.getFileName() == null ? "" : relative.getFileName().toString();
    return relative.startsWith(GENERATED_DIRECTORY)
        || directoryName.equals(".git")
        || directoryName.equals(".gradle")
        || directoryName.equals(".venv")
        || directoryName.equals("native")
        || directoryName.equals("build");
  }

  private static boolean isTestJavaSource(Path relative) {
    for (int index = 0; index + 2 < relative.getNameCount(); index++) {
      if (relative.getName(index).toString().equals("src")
          && (relative.getName(index + 1).toString().equals("test")
              || relative.getName(index + 1).toString().equals("testFixtures"))
          && relative.getName(index + 2).toString().equals("java")) {
        return true;
      }
    }
    return false;
  }
}
