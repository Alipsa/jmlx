package se.alipsa.jmlx.buildsrc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class MlxApiInventoryTest {

  @TempDir Path temporaryDirectory;

  @Test
  void rendersEveryGeneratedCategoryInStableOrder() throws Exception {
    Path root = fixture(record("mlx_h.mlx_array_new", "implemented"));

    String first = MlxApiInventory.render(root);

    assertEquals(first, MlxApiInventory.render(root));
    assertTrue(first.contains("generated entries: 7"));
    assertTrue(first.contains("downcall=4, constant=1, layout/accessor=1, upcall interface=1"));
    assertTrue(first.contains("| `mlx_h.MLX_FLOAT32` | constant | unplanned"));
    assertTrue(first.contains("| `mlx_h.mlx_array_new` | downcall | implemented | reason | test"));
    assertTrue(first.indexOf("`mlx_array_`") < first.indexOf("`mlx_h.MLX_FLOAT32`"));
  }

  @Test
  void rejectsUnknownAndDuplicateMappings() throws Exception {
    Path unknown = fixture(record("mlx_h.mlx_missing", "planned"));
    assertTrue(
        assertThrows(IllegalArgumentException.class, () -> MlxApiInventory.render(unknown))
            .getMessage()
            .contains("Unknown or stale"));

    Path duplicate =
        fixture(
            "{\"records\":["
                + recordBody("mlx_h.mlx_array_new", "planned")
                + ","
                + recordBody("mlx_h.mlx_array_new", "planned")
                + "]}");
    assertTrue(
        assertThrows(IllegalArgumentException.class, () -> MlxApiInventory.render(duplicate))
            .getMessage()
            .contains("Duplicate"));
  }

  @Test
  void callSiteGuardRejectsOnlyUnmappedSyntaxUses() throws Exception {
    Path root = fixture(record("mlx_h.mlx_array_new", "implemented"));
    Path source = root.resolve("sample/Example.java");
    Files.createDirectories(source.getParent());
    Files.writeString(
        source,
        """
        import se.alipsa.jmlx.ffi.mlx_h;
        class Example {
          void use() {
            // mlx_h.mlx_missing();
            String text = "mlx_h.mlx_missing";
            mlx_h.mlx_array_new();
            mlx_h.mlx_array_free();
          }
        }
        """);

    IllegalStateException failure =
        assertThrows(IllegalStateException.class, () -> MlxApiCallSites.verify(root));

    assertTrue(failure.getMessage().contains("mlx_h.mlx_array_free"));
    assertTrue(!failure.getMessage().contains("mlx_h.mlx_missing"));
  }

  private Path fixture(String mappings) throws IOException {
    Path root = Files.createTempDirectory(temporaryDirectory, "inventory-");
    Path bindings = root.resolve("jmlx-ffi/src/main/generated/java/se/alipsa/jmlx/ffi");
    Files.createDirectories(bindings);
    Files.createDirectories(root.resolve("req"));
    Files.createDirectories(root.resolve("scripts"));
    Files.writeString(
        root.resolve("scripts/bootstrap-native.sh"),
        "MLX_METAL_VERSION=\"1.2.3\"\nMLX_C_COMMIT=\"abc\"\n");
    Files.writeString(root.resolve("req/mlx-api-inventory-overrides.json"), mappings);
    Files.writeString(bindings.resolve("mlx_h.java"), binding());
    Files.writeString(bindings.resolve("mlx_array_.java"), "package fixture;");
    Files.writeString(bindings.resolve("mlx_callback$fun.java"), "package fixture;");
    return root;
  }

  private static String binding() {
    return String.join(
        "\n",
        "public class mlx_h {",
        "    private static class mlx_array_new {",
        "    private static class mlx_array_free {",
        "    private static class mlx_array_new_data {",
        "    private static class mlx_array_new_data_managed {",
        "    public static int mlx_array_new() {",
        "    public static int mlx_array_free() {",
        "    public static int mlx_array_new_data() {",
        "    public static int mlx_array_new_data_managed() {",
        "    public static int MLX_FLOAT32() {");
  }

  private static String record(String binding, String status) {
    return "{\"records\":[" + recordBody(binding, status) + "]}";
  }

  private static String recordBody(String binding, String status) {
    return "{\"binding\":\""
        + binding
        + "\",\"category\":\"downcall\",\"status\":\""
        + status
        + "\",\"facadeOrReason\":\"reason\",\"tests\":\"test\"}";
  }
}
