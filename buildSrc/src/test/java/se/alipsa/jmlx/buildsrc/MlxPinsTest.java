package se.alipsa.jmlx.buildsrc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class MlxPinsTest {

  @Test
  void readsPinsAndExpandsReferences(@TempDir Path repositoryRoot) throws Exception {
    Path scripts = Files.createDirectories(repositoryRoot.resolve("scripts"));
    Files.writeString(
        scripts.resolve("bootstrap-native.sh"),
        "MLX_METAL_VERSION=\"1.2.3\"\n"
            + "MLX_METAL_WHEEL_URL=\"https://example.test/mlx-${MLX_METAL_VERSION}.whl\"\n"
            + "MLX_METAL_WHEEL_SHA256=\"abc123\"\n"
            + "INSTALL_DIR=\"/tmp/native\"\n"
            + "RUNTIME_LIB_DIR=\"$INSTALL_DIR/lib\"\n");

    MlxPins pins = MlxPins.read(repositoryRoot);

    assertEquals("1.2.3", pins.required("MLX_METAL_VERSION"));
    assertEquals("https://example.test/mlx-1.2.3.whl", pins.required("MLX_METAL_WHEEL_URL"));
    assertEquals("abc123", pins.required("MLX_METAL_WHEEL_SHA256"));
    assertEquals("/tmp/native/lib", pins.required("RUNTIME_LIB_DIR"));
  }

  @Test
  void rejectsMissingAndCyclicPins(@TempDir Path repositoryRoot) throws Exception {
    Path scripts = Files.createDirectories(repositoryRoot.resolve("scripts"));
    Files.writeString(
        scripts.resolve("bootstrap-native.sh"), "FIRST=\"${SECOND}\"\nSECOND=\"${FIRST}\"\n");
    MlxPins pins = MlxPins.read(repositoryRoot);

    IllegalArgumentException missing =
        assertThrows(IllegalArgumentException.class, () -> pins.required("MISSING"));
    assertTrue(missing.getMessage().contains("MISSING"), missing::getMessage);
    IllegalArgumentException cyclic =
        assertThrows(IllegalArgumentException.class, () -> pins.required("FIRST"));
    assertTrue(cyclic.getMessage().contains("cyclic"), cyclic::getMessage);
  }

  @Test
  void rejectsUnsupportedShellExpansion(@TempDir Path repositoryRoot) throws Exception {
    Path scripts = Files.createDirectories(repositoryRoot.resolve("scripts"));
    Files.writeString(scripts.resolve("bootstrap-native.sh"), "DYNAMIC=\"$(pwd)/native\"\n");

    IllegalArgumentException failure =
        assertThrows(
            IllegalArgumentException.class, () -> MlxPins.read(repositoryRoot).required("DYNAMIC"));

    assertTrue(failure.getMessage().contains("unsupported shell expansion"), failure::getMessage);
  }
}
