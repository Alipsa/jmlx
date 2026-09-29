package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import org.junit.jupiter.api.Test;
import tools.jackson.databind.ObjectMapper;

class HfReferenceProvenanceTest {
  @Test
  void everyCommittedGoldenMatchesItsRecordedHash() throws Exception {
    Path root = Path.of(System.getProperty("jmlx.repository.root"), "tools", "hf-reference");
    var manifest = new ObjectMapper().readTree(root.resolve("provenance.json").toFile());
    var expected = manifest.path("files");
    try (var files = Files.walk(root.resolve("goldens"))) {
      var actual = files.filter(Files::isRegularFile).toList();
      assertEquals(expected.size(), actual.size(), "golden file count");
      for (Path file : actual) {
        String name = root.resolve("goldens").relativize(file).toString().replace('\\', '/');
        String hash =
            HexFormat.of()
                .formatHex(MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(file)));
        assertEquals(expected.path(name).asString(), hash, name);
      }
    }
  }
}
