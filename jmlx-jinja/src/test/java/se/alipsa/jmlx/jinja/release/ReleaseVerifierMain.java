package se.alipsa.jmlx.jinja.release;

import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Creates an isolated detached worktree and runs the release candidate matrix. */
public final class ReleaseVerifierMain {
  private static final Pattern COORDINATES =
      Pattern.compile("\\\"coordinates\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
  private static final Pattern JDK_MAJOR = Pattern.compile("\\\"jdkMajor\\\"\\s*:\\s*(\\d+)");
  private static final Pattern NODE_VERSION =
      Pattern.compile("\\\"nodeVersion\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
  private static final Pattern ARRAY =
      Pattern.compile("\\\"%s\\\"\\s*:\\s*\\[([^]]*)]", Pattern.DOTALL);
  private static final Pattern JSON_STRING = Pattern.compile("\\\"([^\\\"]+)\\\"");

  private ReleaseVerifierMain() {}

  /** Runs the release verification workflow. */
  public static void main(String[] args) throws Exception {
    if (args.length < 5 || args.length > 6) {
      throw new IllegalArgumentException(
          "usage: ReleaseVerifierMain <source-dir> <gradle-user-home> <daemon-java-home>"
              + " <daemon-vendor> <daemon-version> [--allow-dirty]");
    }
    Path source = Path.of(args[0]).toAbsolutePath().normalize();
    final String moduleDirectory = source.getFileName().toString();
    final Path userHome = Path.of(args[1]).toAbsolutePath().normalize();
    final String daemonJavaHome = args[2];
    final String daemonVendor = args[3];
    String daemonVersion = args[4];
    boolean allowDirty = args.length == 6 && "--allow-dirty".equals(args[5]);
    Path report = source.resolve("build/reports/release-verification.md");
    Path retainedEvidence = source.resolve("build/reports/release-verification-evidence");
    prepareReport(report, retainedEvidence);
    verifyEnvironment(source, daemonVersion);
    // Repo-wide, not `-- "."` scoped to the module directory: in this monorepo the build genuinely
    // depends on root build.gradle/settings.gradle/gradle.properties/gradle/libs.versions.toml/
    // gradle/wrapper -- the isolated worktree below materializes only committed state, so an
    // uncommitted edit to any of those would otherwise pass this guard while the candidate matrix
    // silently verifies a different build than the one just tested.
    String status = output(source, "git", "status", "--porcelain");
    if (!status.isBlank() && !allowDirty) {
      throw new IllegalStateException("source checkout is dirty:\n" + status);
    }
    String head = output(source, "git", "rev-parse", "HEAD").trim();
    String coordinates = contractCoordinates(source.resolve("req/release-verification.json"));
    Path parent = Files.createTempDirectory("jmlx-jinja-release-worktree-");
    Path worktree = parent.resolve("candidate");
    Path dependencyEvidence = Files.createTempDirectory("jmlx-jinja-dependency-evidence-");
    Path repository = Files.createTempDirectory("jmlx-jinja-release-repository-");
    try {
      run(source, "git", "worktree", "add", "--detach", worktree.toString(), head);
      Path candidateModule = worktree.resolve(moduleDirectory);
      Files.createDirectories(userHome);
      Files.writeString(
          worktree.resolve("gradle.properties"),
          "\norg.gradle.java.installations.auto-download=false\n",
          java.nio.file.StandardOpenOption.APPEND);

      // Each destructive operation gets a separate Gradle process and graph.
      gradle(candidateModule, userHome, false, "verifyReproducibleArchives");
      gradle(candidateModule, userHome, false, "clean", "check");
      gradle(candidateModule, userHome, false, "dependencyUpdates");
      Path stagedReport = dependencyEvidence.resolve("dependency-updates.txt");
      Files.copy(candidateModule.resolve("build/dependencyUpdates/report.txt"), stagedReport);

      gradle(candidateModule, userHome, true, "verifyReproducibleArchives");
      final Path archiveEvidence = dependencyEvidence.resolve("archive-reproducibility.json");
      copyEvidence(
          candidateModule, dependencyEvidence, "build/reports/archive-reproducibility.json");
      gradle(candidateModule, userHome, true, "clean", "check");
      gradle(
          candidateModule,
          userHome,
          true,
          "corpusCoverage",
          "formatGoldenVerify",
          "fuzzParserVerify");
      verifyRequiredTaskEvidence(candidateModule, source.resolve("req/release-verification.json"));
      gradle(
          candidateModule,
          userHome,
          true,
          "dependencyReview",
          "-PreleaseVerificationDependencyReport=" + stagedReport);
      Files.createDirectories(retainedEvidence);
      copyEvidence(candidateModule, retainedEvidence, "build/reports/corpus-coverage.md");
      copyEvidence(candidateModule, retainedEvidence, "build/publications/maven/pom-default.xml");
      copyEvidence(candidateModule, retainedEvidence, "build/reports/dependency-review.md");
      Files.copy(
          archiveEvidence,
          retainedEvidence.resolve("archive-reproducibility.json"),
          java.nio.file.StandardCopyOption.REPLACE_EXISTING);
      gradle(
          candidateModule,
          userHome,
          true,
          "publishMavenPublicationToReleaseVerificationRepository",
          "-PreleaseVerificationRepository=" + repository);
      String resolvedDigest = runConsumer(worktree, userHome, repository, coordinates);
      Path published = publishedJar(repository, coordinates);
      String publishedDigest = sha256(published);
      String archiveDigest = mainArchiveDigest(archiveEvidence, coordinates);
      if (!archiveDigest.equals(publishedDigest) || !publishedDigest.equals(resolvedDigest)) {
        throw new IllegalStateException(
            "archive, published candidate, and consumer-resolved main JAR digests must all match");
      }
      writeReport(
          report,
          head,
          status,
          allowDirty,
          worktree,
          source,
          published,
          resolvedDigest,
          coordinates,
          archiveEvidence,
          daemonJavaHome,
          daemonVendor,
          daemonVersion);
    } finally {
      runQuietly(source, "git", "worktree", "remove", "--force", worktree.toString());
      runQuietly(source, "git", "worktree", "prune");
      deleteTree(parent);
      deleteTree(dependencyEvidence);
      deleteTree(repository);
    }
  }

  private static void gradle(Path project, Path userHome, boolean offline, String... tasks)
      throws Exception {
    Path repositoryRoot = repositoryRoot(project);
    String wrapper =
        System.getProperty("os.name").toLowerCase().contains("win") ? "gradlew.bat" : "gradlew";
    List<String> command =
        new ArrayList<>(
            List.of(
                repositoryRoot.resolve(wrapper).toString(),
                "-Dorg.gradle.java.installations.auto-download=false",
                "--gradle-user-home",
                userHome.toString()));
    if (offline) {
      command.add("--offline");
    }
    command.addAll(List.of(tasks));
    requireIsolation(command, offline);
    run(project, command.toArray(String[]::new));
  }

  static void requireIsolation(List<String> command, boolean offline) {
    if (!command.contains("--gradle-user-home") || (offline && !command.contains("--offline"))) {
      throw new IllegalStateException(
          "refusing to launch a candidate Gradle command without required isolation flags");
    }
  }

  static void verifyEnvironment(Path source, String daemonVersion) throws Exception {
    int requiredJdk =
        contractInt(source.resolve("req/release-verification.json"), JDK_MAJOR, "jdkMajor");
    int actualJdk = Runtime.Version.parse(daemonVersion).feature();
    if (actualJdk != requiredJdk) {
      throw new IllegalStateException(
          "required Gradle daemon JDK major " + requiredJdk + ", got " + actualJdk);
    }
    String expectedNode =
        match(source.resolve("upstream/upstream-lock.json"), NODE_VERSION, "nodeVersion");
    String actualNode = output(source, "node", "--version").trim();
    if (!expectedNode.equals(actualNode)) {
      throw new IllegalStateException("required Node " + expectedNode + ", got " + actualNode);
    }
  }

  static void verifyRequiredTaskEvidence(Path worktree, Path contract) throws Exception {
    java.util.Map<String, String> evidence =
        java.util.Map.of(
            "upstreamVerify",
            "build/upstreamVerify/verified",
            "corpusCoverage",
            "build/reports/corpus-coverage.md",
            "formatGoldenVerify",
            "build/formatGoldenVerify/verified",
            "fuzzParserVerify",
            "build/reports/fuzz-parser.md");
    for (String task : stringArray(contract, "requiredTasks")) {
      String relative = evidence.get(task);
      if (relative == null) {
        throw new IllegalStateException("no evidence mapping for required task: " + task);
      }
      if (!Files.isRegularFile(worktree.resolve(relative))) {
        throw new IllegalStateException("required verification evidence is missing: " + relative);
      }
    }
  }

  private static String runConsumer(
      Path candidate, Path userHome, Path repository, String coordinates) throws Exception {
    Path consumer = Files.createTempDirectory("jmlx-jinja-release-consumer-");
    String[] coordinate = coordinates.split(":", -1);
    if (coordinate.length != 3) {
      throw new IllegalStateException("invalid contract coordinates: " + coordinates);
    }
    try {
      Files.writeString(
          consumer.resolve("settings.gradle"),
          "pluginManagement { repositories {"
              + "} }\nrootProject.name = 'jmlx-jinja-release-consumer'\n");
      String repositoryUri = repository.toUri().toString().replace("'", "\\'");
      Files.writeString(
          consumer.resolve("build.gradle"),
          """
          plugins { id 'java' }
          repositories { maven { url = uri('%s') } }
          dependencies { implementation '%s' }
          java { toolchain { languageVersion = JavaLanguageVersion.of(21) } }
          configurations.configureEach {
            resolutionStrategy.cacheChangingModulesFor 0, 'seconds'
            resolutionStrategy.cacheDynamicVersionsFor 0, 'seconds'
          }
          tasks.register('consumerVerify', JavaExec) {
            dependsOn tasks.named('classes')
            classpath = sourceSets.main.runtimeClasspath
            mainClass = 'consumer.ConsumerMain'
          }
          tasks.register('copyResolvedRuntime', Copy) {
            from configurations.runtimeClasspath
            into layout.buildDirectory.dir('resolved')
          }
          tasks.named('consumerVerify') { finalizedBy tasks.named('copyResolvedRuntime') }
          """
              .formatted(repositoryUri, coordinates));
      Files.writeString(
          consumer.resolve("gradle.properties"),
          "org.gradle.java.installations.auto-download=false\n");
      verifyConsumerStructure(consumer, candidate, repository);
      Path source = consumer.resolve("src/main/java/consumer/ConsumerMain.java");
      Files.createDirectories(source.getParent());
      Files.writeString(
          source,
          """
          package consumer;
          import java.util.Map;
          import se.alipsa.jmlx.jinja.Template;
          public final class ConsumerMain {
            public static void main(String[] args) { System.out.print(Template.parse(\"{{ value }}\").render(Map.of(\"value\", \"verified\"))); }
          }
          """);
      String wrapper =
          System.getProperty("os.name").toLowerCase().contains("win") ? "gradlew.bat" : "gradlew";
      List<String> consumerCommand =
          new ArrayList<>(
              List.of(
                  candidate.resolve(wrapper).toString(),
                  "-Dorg.gradle.java.installations.auto-download=false",
                  "--gradle-user-home",
                  userHome.toString(),
                  "consumerVerify"));
      requireIsolation(consumerCommand, false);
      run(consumer, consumerCommand.toArray(String[]::new));
      try (var files = Files.list(consumer.resolve("build/resolved"))) {
        List<Path> resolved = files.toList();
        if (resolved.size() != 1) {
          throw new IllegalStateException(
              "consumer runtime graph must contain only the candidate JAR: " + resolved);
        }
        if (!resolved.getFirst().getFileName().toString().startsWith(coordinate[1] + "-")) {
          throw new IllegalStateException(
              "consumer resolved an artifact other than the candidate JAR: " + resolved);
        }
        return sha256(resolved.getFirst());
      }
    } finally {
      deleteTree(consumer);
    }
  }

  static void verifyConsumerStructure(Path consumer, Path candidate, Path repository)
      throws Exception {
    String settings = Files.readString(consumer.resolve("settings.gradle"));
    String build = Files.readString(consumer.resolve("build.gradle"));
    String properties = Files.readString(consumer.resolve("gradle.properties"));
    if (!settings.contains("pluginManagement { repositories {" + "} }")
        || !build.contains(repository.toUri().toString())
        || build.contains("mavenCentral")
        || build.contains("mavenLocal")
        || !properties.contains("org.gradle.java.installations.auto-download=false")) {
      throw new IllegalStateException("consumer structural network-isolation contract is invalid");
    }
    String wrapper =
        System.getProperty("os.name").toLowerCase().contains("win") ? "gradlew.bat" : "gradlew";
    if (!Files.isRegularFile(candidate.resolve(wrapper))) {
      throw new IllegalStateException("consumer must use the candidate worktree wrapper");
    }
  }

  private static Path publishedJar(Path repository, String coordinates) throws Exception {
    String[] coordinate = coordinates.split(":", -1);
    Path directory =
        repository
            .resolve(coordinate[0].replace('.', '/'))
            .resolve(coordinate[1])
            .resolve(coordinate[2]);
    try (var files = Files.list(directory)) {
      return files
          .filter(
              path ->
                  path.getFileName().toString().startsWith(coordinate[1] + "-")
                      && path.getFileName().toString().endsWith(".jar"))
          .filter(
              path ->
                  !path.getFileName().toString().endsWith("-sources.jar")
                      && !path.getFileName().toString().endsWith("-javadoc.jar"))
          .findFirst()
          .orElseThrow(
              () ->
                  new IllegalStateException(
                      "published candidate JAR is missing from " + directory));
    }
  }

  private static void writeReport(
      Path report,
      String head,
      String status,
      boolean allowDirty,
      Path worktree,
      Path source,
      Path published,
      String resolvedDigest,
      String coordinates,
      Path archiveEvidence,
      String daemonJavaHome,
      String daemonVendor,
      String daemonVersion)
      throws Exception {
    Files.createDirectories(report.getParent());
    String candidate =
        status.isBlank() && !allowDirty ? "PASSING CANDIDATE" : "NOT A RELEASE CANDIDATE";
    String javaVersion = daemonVendor + " " + daemonVersion + " (" + daemonJavaHome + ")";
    String nodeVersion = output(source, "node", "--version").trim();
    List<String> policyInputs =
        stringArray(source.resolve("req/release-verification.json"), "policyInputs");
    policyInputs.add("upstream/upstream-lock.json");
    String reviewInputs = digests(source, policyInputs);
    String body =
        "# Release verification\n\nStatus: "
            + candidate
            + "\n\n- Generated: "
            + Instant.now()
            + "\n- Git HEAD: `"
            + head
            + "`\n- Dirty status: `"
            + status.replace("`", "'").replace("\n", "; ")
            + "`\n- Coordinates: `"
            + coordinates
            + "`\n- JDK: `"
            + javaVersion
            + "`\n- Node: `"
            + nodeVersion
            + "`\n- Candidate worktree: `"
            + worktree
            + "`\n- Published candidate JAR SHA-256: `"
            + sha256(published)
            + "`\n- Consumer-resolved JAR SHA-256: `"
            + resolvedDigest
            + "`\n\n## Reproducible archive hashes\n\n```json\n"
            + Files.readString(archiveEvidence)
            + "```\n\n## Contract inputs\n\n"
            + reviewInputs
            + "\n"
            + "## Human sign-off\n\n"
            + "- [ ] NOTICE, changelog, POM, Javadoc, dependency report, and corpus evidence"
            + " reviewed.\n";
    Files.writeString(report, body);
    Files.writeString(
        report.resolveSibling("release-verification.json"),
        "{\n  \"status\": \""
            + json(candidate)
            + "\",\n  \"gitHead\": \""
            + json(head)
            + "\",\n  \"coordinates\": \""
            + json(coordinates)
            + "\",\n  \"worktree\": \""
            + json(worktree.toString())
            + "\"\n}\n");
  }

  private static String contractCoordinates(Path contract) throws Exception {
    return match(contract, COORDINATES, "coordinates");
  }

  private static int contractInt(Path contract, Pattern pattern, String name) throws Exception {
    return Integer.parseInt(match(contract, pattern, name));
  }

  private static String match(Path file, Pattern pattern, String name) throws Exception {
    Matcher matcher = pattern.matcher(Files.readString(file));
    if (!matcher.find()) {
      throw new IllegalStateException("missing " + name + " in " + file);
    }
    return matcher.group(1);
  }

  static List<String> stringArray(Path file, String name) throws Exception {
    Matcher array =
        Pattern.compile(ARRAY.pattern().formatted(name), Pattern.DOTALL)
            .matcher(Files.readString(file));
    if (!array.find()) {
      throw new IllegalStateException("missing " + name + " in " + file);
    }
    List<String> result = new ArrayList<>();
    Matcher value = JSON_STRING.matcher(array.group(1));
    while (value.find()) {
      result.add(value.group(1));
    }
    if (result.isEmpty()) {
      throw new IllegalStateException(name + " must not be empty");
    }
    return result;
  }

  static void prepareReport(Path report, Path retainedEvidence) throws Exception {
    Files.deleteIfExists(report);
    Files.deleteIfExists(report.resolveSibling("release-verification.json"));
    deleteTree(retainedEvidence);
    Files.createDirectories(report.getParent());
    Files.writeString(report, "# Release verification\n\nStatus: IN PROGRESS\n");
    Files.writeString(
        report.resolveSibling("release-verification.json"),
        "{\n  \"status\": \"IN PROGRESS\"\n}\n");
  }

  static String mainArchiveDigest(Path archiveEvidence, String coordinates) throws Exception {
    String[] coordinate = coordinates.split(":", -1);
    if (coordinate.length != 3) {
      throw new IllegalArgumentException("invalid coordinates: " + coordinates);
    }
    Pattern mainArchiveDigest =
        Pattern.compile(
            "\\\"name\\\"\\s*:\\s*\\\""
                + Pattern.quote(coordinate[1])
                + "-(?![^\\\"]+-(?:sources|javadoc)\\.jar)[^\\\"]+\\.jar\\\""
                + "\\s*,\\s*\\\"firstSha256\\\"\\s*:\\s*\\\"([0-9a-f]{64})\\\""
                + "\\s*,\\s*\\\"secondSha256\\\"\\s*:\\s*\\\"([0-9a-f]{64})\\\"");
    Matcher match = mainArchiveDigest.matcher(Files.readString(archiveEvidence));
    if (!match.find()) {
      throw new IllegalStateException("archive evidence has no main JAR digest");
    }
    if (!match.group(1).equals(match.group(2))) {
      throw new IllegalStateException("archive evidence records non-identical main JARs");
    }
    return match.group(1);
  }

  private static void copyEvidence(Path worktree, Path destination, String relative)
      throws Exception {
    Path source = worktree.resolve(relative);
    if (!Files.isRegularFile(source)) {
      throw new IllegalStateException("required candidate evidence is missing: " + relative);
    }
    Files.copy(
        source,
        destination.resolve(source.getFileName()),
        java.nio.file.StandardCopyOption.REPLACE_EXISTING);
  }

  private static String digests(Path source, List<String> names) throws Exception {
    StringBuilder result = new StringBuilder();
    for (String name : names) {
      result
          .append("- `")
          .append(name)
          .append("`: `")
          .append(sha256(source.resolve(name)))
          .append("`\n");
    }
    return result.toString();
  }

  private static String json(String value) {
    return value
        .replace("\\", "\\\\")
        .replace("\"", "\\\"")
        .replace("\n", "\\n")
        .replace("\r", "\\r");
  }

  private static String sha256(Path file) throws Exception {
    return HexFormat.of()
        .formatHex(MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(file)));
  }

  private static Path repositoryRoot(Path directory) throws Exception {
    return Path.of(output(directory, "git", "rev-parse", "--show-toplevel").trim())
        .toAbsolutePath()
        .normalize();
  }

  private static String output(Path directory, String... command) throws Exception {
    Process process =
        new ProcessBuilder(command).directory(directory.toFile()).redirectErrorStream(true).start();
    String text = new String(process.getInputStream().readAllBytes());
    if (process.waitFor() != 0) {
      throw new IllegalStateException(text);
    }
    return text;
  }

  private static void run(Path directory, String... command) throws Exception {
    Process process = new ProcessBuilder(command).directory(directory.toFile()).inheritIO().start();
    if (process.waitFor() != 0) {
      throw new IllegalStateException("command failed: " + String.join(" ", command));
    }
  }

  private static void runQuietly(Path directory, String... command) {
    try {
      run(directory, command);
    } catch (Exception ignored) {
      // Cleanup is best effort.
    }
  }

  private static void deleteTree(Path path) {
    try (var files = Files.walk(path)) {
      files
          .sorted(java.util.Comparator.reverseOrder())
          .forEach(
              file -> {
                try {
                  Files.deleteIfExists(file);
                } catch (Exception ignored) {
                  // Cleanup is best effort.
                }
              });
    } catch (Exception ignored) {
      // Cleanup is best effort.
    }
  }
}
