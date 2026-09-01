package se.alipsa.jmlx.jinja.release;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ReleaseVerifierMainTest {
  @TempDir Path temporaryDirectory;

  @Test
  void requiresGradleUserHome() {
    assertThrows(
        IllegalStateException.class,
        () -> ReleaseVerifierMain.requireIsolation(List.of("gradlew", "check"), false));
  }

  @Test
  void requiresOfflineWhenRequested() {
    assertThrows(
        IllegalStateException.class,
        () ->
            ReleaseVerifierMain.requireIsolation(
                List.of("gradlew", "--gradle-user-home", "/tmp/home", "check"), true));
  }

  @Test
  void acceptsCompleteIsolatedCommand() {
    assertDoesNotThrow(
        () ->
            ReleaseVerifierMain.requireIsolation(
                List.of("gradlew", "--gradle-user-home", "/tmp/home", "--offline", "check"), true));
  }

  @Test
  void readsContractPolicyInputs() throws Exception {
    var contract = temporaryDirectory.resolve("release-contract.json");
    Files.writeString(contract, "{\"policyInputs\":[\"NOTICE\",\"CHANGELOG.md\"]}");
    assertEquals(
        List.of("NOTICE", "CHANGELOG.md"),
        ReleaseVerifierMain.stringArray(contract, "policyInputs"));
  }

  @Test
  void acceptsNewerDaemonJdk() throws Exception {
    var source = temporaryDirectory.resolve("jmlx-jinja-release-environment");
    Files.createDirectories(source.resolve("req"));
    Files.createDirectories(source.resolve("upstream"));
    Files.writeString(source.resolve("req/release-verification.json"), "{\"jdkMajor\":21}");
    Files.writeString(
        source.resolve("upstream/upstream-lock.json"), "{\"nodeVersion\":\"v26.7.0\"}");
    assertDoesNotThrow(() -> ReleaseVerifierMain.verifyEnvironment(source, "25.0.4"));
  }

  @Test
  void rejectsOlderDaemonJdk() throws Exception {
    var source = temporaryDirectory.resolve("jmlx-jinja-release-environment");
    Files.createDirectories(source.resolve("req"));
    Files.createDirectories(source.resolve("upstream"));
    Files.writeString(source.resolve("req/release-verification.json"), "{\"jdkMajor\":21}");
    Files.writeString(
        source.resolve("upstream/upstream-lock.json"), "{\"nodeVersion\":\"v26.7.0\"}");
    var failure =
        assertThrows(
            IllegalStateException.class,
            () -> ReleaseVerifierMain.verifyEnvironment(source, "17.0.14"));
    assertEquals("required Gradle daemon JDK major at least 21, got 17", failure.getMessage());
  }

  @Test
  void rejectsConsumerWithPluginRepository() throws Exception {
    var consumer = Files.createDirectories(temporaryDirectory.resolve("jmlx-jinja-consumer"));
    var candidate = Files.createDirectories(temporaryDirectory.resolve("jmlx-jinja-candidate"));
    var repository = Files.createDirectories(temporaryDirectory.resolve("jmlx-jinja-repository"));
    Files.writeString(candidate.resolve("gradlew"), "#!/bin/sh\n");
    Files.writeString(
        consumer.resolve("settings.gradle"),
        "pluginManagement { repositories { mavenCentral() } }");
    Files.writeString(
        consumer.resolve("build.gradle"),
        "repositories { maven { url = uri('" + repository.toUri() + "') } }");
    Files.writeString(
        consumer.resolve("gradle.properties"),
        "org.gradle.java.installations.auto-download=false\n");
    assertThrows(
        IllegalStateException.class,
        () -> ReleaseVerifierMain.verifyConsumerStructure(consumer, candidate, repository));
  }

  @Test
  void rejectsArchiveEvidenceWithoutMainJar() throws Exception {
    var evidence = temporaryDirectory.resolve("jmlx-jinja-archive-evidence.json");
    Files.writeString(evidence, "{\"archives\":[]}");
    assertThrows(
        IllegalStateException.class,
        () ->
            ReleaseVerifierMain.mainArchiveDigest(evidence, "se.alipsa:jmlx-jinja:0.6.0-SNAPSHOT"));
  }

  @Test
  void rejectsNonIdenticalArchiveEvidence() throws Exception {
    var evidence = temporaryDirectory.resolve("jmlx-jinja-archive-mismatch.json");
    Files.writeString(
        evidence,
        "{\"name\":\"jmlx-jinja-0.6.0-SNAPSHOT.jar\",\"firstSha256\":\""
            + "a".repeat(64)
            + "\",\"secondSha256\":\""
            + "b".repeat(64)
            + "\"}");
    assertThrows(
        IllegalStateException.class,
        () ->
            ReleaseVerifierMain.mainArchiveDigest(evidence, "se.alipsa:jmlx-jinja:0.6.0-SNAPSHOT"));
  }

  @Test
  void findsTheMainJarForTheConfiguredArtifact() throws Exception {
    var evidence = temporaryDirectory.resolve("jmlx-jinja-archive-evidence.json");
    String digest = "a".repeat(64);
    Files.writeString(
        evidence,
        "{\"name\":\"jmlx-jinja-0.6.0-SNAPSHOT.jar\",\"firstSha256\":\""
            + digest
            + "\",\"secondSha256\":\""
            + digest
            + "\"}");
    assertEquals(
        digest,
        ReleaseVerifierMain.mainArchiveDigest(evidence, "se.alipsa:jmlx-jinja:0.6.0-SNAPSHOT"));
  }

  @Test
  void rejectsMissingRequiredTaskEvidence() throws Exception {
    var contract = temporaryDirectory.resolve("release-contract-required.json");
    Files.writeString(contract, "{\"requiredTasks\":[\"corpusCoverage\"]}");
    assertThrows(
        IllegalStateException.class,
        () -> ReleaseVerifierMain.verifyRequiredTaskEvidence(temporaryDirectory, contract));
  }
}
