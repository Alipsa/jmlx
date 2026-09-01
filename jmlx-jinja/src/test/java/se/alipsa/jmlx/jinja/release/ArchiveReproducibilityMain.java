package se.alipsa.jmlx.jinja.release;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.zip.ZipFile;

/** Runs the release-only, independent archive reproducibility builds. */
public final class ArchiveReproducibilityMain {
  private ArchiveReproducibilityMain() {}

  /** Runs the reproducible archive verification workflow. */
  public static void main(String[] args) throws Exception {
    if (args.length != 5) {
      throw new IllegalArgumentException(
          "usage: ArchiveReproducibilityMain <project-dir> <gradle-user-home> <offline>"
              + " <report> <bytecode-major>");
    }
    Path project = Path.of(args[0]).toAbsolutePath().normalize();
    String moduleDirectory = project.getFileName().toString();
    Path userHome = Path.of(args[1]).toAbsolutePath().normalize();
    boolean offline = Boolean.parseBoolean(args[2]);
    Path report = Path.of(args[3]).toAbsolutePath().normalize();
    int bytecodeMajor = Integer.parseInt(args[4]);
    Path evidence = Files.createTempDirectory("jmlx-jinja-archive-evidence-");
    Path sandboxParent = Files.createTempDirectory("jmlx-jinja-archive-worktree-");
    Path sandbox = sandboxParent.resolve("candidate");
    try {
      runGit(project, "git", "worktree", "add", "--detach", sandbox.toString(), "HEAD");
      Path candidateModule = sandbox.resolve(moduleDirectory);
      List<Path> first =
          buildAndCopy(candidateModule, userHome, offline, evidence.resolve("first"));
      List<Path> second =
          buildAndCopy(candidateModule, userHome, offline, evidence.resolve("second"));
      for (int index = 0; index < first.size(); index++) {
        if (!sha256(first.get(index)).equals(sha256(second.get(index)))) {
          throw new IllegalStateException(
              "archive is not reproducible: " + first.get(index).getFileName());
        }
      }
      verifyModule(first.get(0), bytecodeMajor);
      Files.createDirectories(report.getParent());
      Files.writeString(report, report(first, second));
      System.out.println("archive evidence: " + report);
      for (Path archive : first) {
        System.out.println(archive.getFileName() + " " + sha256(archive));
      }
    } finally {
      runQuietly(project, "git", "worktree", "remove", "--force", sandbox.toString());
      runQuietly(project, "git", "worktree", "prune");
      deleteTree(sandboxParent);
      deleteTree(evidence);
    }
  }

  private static List<Path> buildAndCopy(
      Path project, Path userHome, boolean offline, Path destination) throws Exception {
    runGradle(project, userHome, offline, "clean", "jar", "sourcesJar", "javadocJar");
    Files.createDirectories(destination);
    List<Path> result = new ArrayList<>();
    for (String suffix : List.of(".jar", "-sources.jar", "-javadoc.jar")) {
      try (var files = Files.list(project.resolve("build/libs"))) {
        Path archive =
            files
                .filter(path -> matchesArchive(path, suffix))
                .findFirst()
                .orElseThrow(() -> new IllegalStateException("missing archive ending " + suffix));
        Path copy = destination.resolve(archive.getFileName());
        Files.copy(archive, copy);
        result.add(copy);
      }
    }
    return result;
  }

  private static boolean matchesArchive(Path path, String suffix) {
    String name = path.getFileName().toString();
    return suffix.equals(".jar")
        ? name.endsWith(suffix) && !name.endsWith("-sources.jar") && !name.endsWith("-javadoc.jar")
        : name.endsWith(suffix);
  }

  private static void runGradle(Path project, Path userHome, boolean offline, String... tasks)
      throws IOException, InterruptedException {
    Path repositoryRoot = repositoryRoot(project);
    String wrapper =
        System.getProperty("os.name").toLowerCase().contains("win") ? "gradlew.bat" : "gradlew";
    List<String> command =
        new ArrayList<>(
            List.of(
                repositoryRoot.resolve(wrapper).toString(),
                "-Dorg.gradle.java.installations.auto-download=false",
                "--no-build-cache",
                "--gradle-user-home",
                userHome.toString()));
    if (offline) {
      command.add("--offline");
    }
    command.addAll(List.of(tasks));
    Process process = new ProcessBuilder(command).directory(project.toFile()).inheritIO().start();
    if (process.waitFor() != 0) {
      throw new IllegalStateException("archive build failed: " + String.join(" ", command));
    }
  }

  private static void verifyModule(Path archive, int expectedBytecodeMajor) throws IOException {
    try (ZipFile zip = new ZipFile(archive.toFile())) {
      var entry = zip.getEntry("module-info.class");
      if (entry == null) {
        throw new IllegalStateException("archive has no module-info.class: " + archive);
      }
      byte[] moduleInfo = zip.getInputStream(entry).readAllBytes();
      int major = ((moduleInfo[6] & 0xff) << 8) | (moduleInfo[7] & 0xff);
      if (major != expectedBytecodeMajor) {
        throw new IllegalStateException(
            "expected bytecode major " + expectedBytecodeMajor + ", got " + major);
      }
    }
  }

  static String sha256(Path file) throws Exception {
    return HexFormat.of()
        .formatHex(MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(file)));
  }

  private static void runGit(Path directory, String... command)
      throws IOException, InterruptedException {
    Process process = new ProcessBuilder(command).directory(directory.toFile()).inheritIO().start();
    if (process.waitFor() != 0) {
      throw new IllegalStateException("command failed: " + String.join(" ", command));
    }
  }

  private static void runQuietly(Path directory, String... command) {
    try {
      runGit(directory, command);
    } catch (Exception ignored) {
      // Cleanup is best effort; do not hide the original verification failure.
    }
  }

  private static Path repositoryRoot(Path directory) throws IOException, InterruptedException {
    Process process =
        new ProcessBuilder("git", "rev-parse", "--show-toplevel")
            .directory(directory.toFile())
            .redirectErrorStream(true)
            .start();
    String output = new String(process.getInputStream().readAllBytes()).trim();
    if (process.waitFor() != 0) {
      throw new IllegalStateException("cannot find repository root: " + output);
    }
    return Path.of(output).toAbsolutePath().normalize();
  }

  private static String report(List<Path> first, List<Path> second) throws Exception {
    StringBuilder result = new StringBuilder("{\n  \"archives\": [\n");
    for (int index = 0; index < first.size(); index++) {
      Path archive = first.get(index);
      result
          .append("    {\"name\": \"")
          .append(archive.getFileName())
          .append("\", \"firstSha256\": \"")
          .append(sha256(archive))
          .append("\", \"secondSha256\": \"")
          .append(sha256(second.get(index)))
          .append("\"}");
      if (index + 1 < first.size()) {
        result.append(',');
      }
      result.append('\n');
    }
    return result.append("  ]\n}\n").toString();
  }

  private static void deleteTree(Path path) {
    try (var files = Files.walk(path)) {
      files
          .sorted(java.util.Comparator.reverseOrder())
          .forEach(
              file -> {
                try {
                  Files.deleteIfExists(file);
                } catch (IOException ignored) {
                  // Cleanup is best effort.
                }
              });
    } catch (IOException ignored) {
      // Cleanup is best effort.
    }
  }
}
