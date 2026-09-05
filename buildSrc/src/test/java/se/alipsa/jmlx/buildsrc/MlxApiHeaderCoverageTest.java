package se.alipsa.jmlx.buildsrc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class MlxApiHeaderCoverageTest {

  @TempDir Path temporaryDirectory;

  @Test
  void rendersStableCoverageAndMismatchEvidence() throws Exception {
    Fixture fixture = fixture();

    String first =
        MlxApiHeaderCoverage.render(fixture.root(), fixture.headers(), fixture.nativePins());

    assertEquals(
        first,
        MlxApiHeaderCoverage.render(fixture.root(), fixture.headers(), fixture.nativePins()));
    assertTrue(first.contains("header declarations: 14"), first);
    assertTrue(first.contains("comparable generated bindings: 10"), first);
    assertTrue(first.contains("declared-and-bound: 8"), first);
    assertTrue(first.contains("declared-but-not-bound: 6"), first);
    assertTrue(first.contains("bound-without-header-match: 2"), first);
    assertTrue(first.contains("`mlx_h.mlx_header_only` | function | `array.h`"), first);
    assertTrue(first.contains("`mlx_header_only_type` | named type | `array.h`"), first);
    assertTrue(first.contains("`mlx_callback$fun` | upcall interface"), first);
  }

  @Test
  void rejectsMissingHeadersAndStaleNativePins() throws Exception {
    Fixture fixture = fixture();
    Path missing = fixture.root().resolve("missing");

    IllegalArgumentException missingHeaders =
        assertThrows(
            IllegalArgumentException.class,
            () -> MlxApiHeaderCoverage.render(fixture.root(), missing, fixture.nativePins()));
    assertTrue(missingHeaders.getMessage().contains("headers are missing"));

    Files.writeString(fixture.nativePins(), "mlxMetalVersion=9.9.9\nmlxcCommit=abc\n");
    IllegalArgumentException stale =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                MlxApiHeaderCoverage.render(
                    fixture.root(), fixture.headers(), fixture.nativePins()));
    assertTrue(stale.getMessage().contains("mlxMetalVersion mismatch"), stale::getMessage);
  }

  private Fixture fixture() throws Exception {
    Path root = Files.createTempDirectory(temporaryDirectory, "header-coverage-");
    Path scripts = Files.createDirectories(root.resolve("scripts"));
    Files.writeString(
        scripts.resolve("bootstrap-native.sh"),
        "MLX_METAL_VERSION=\"1.2.3\"\nMLX_C_COMMIT=\"abc\"\n");
    Path bindings =
        Files.createDirectories(
            root.resolve("jmlx-ffi/src/main/generated/java/se/alipsa/jmlx/ffi"));
    Files.writeString(bindings.resolve("mlx_h.java"), binding());
    Files.writeString(bindings.resolve("mlx_array.java"), "package fixture;");
    Files.writeString(bindings.resolve("mlx_array_.java"), "package fixture;");
    Files.writeString(bindings.resolve("mlx_callback$fun.java"), "package fixture;");

    Path headers = Files.createDirectories(root.resolve("native/install/include/mlx/c"));
    Files.writeString(
        headers.resolve("array.h"),
        """
        #define MLX_ARRAY_H
        #define mlx_macro(x) x
        /* mlx_h.mlx_comment_only(); */
        typedef struct mlx_array_ { void* ctx; } mlx_array;
        typedef float _Complex mlx_complex64_t;
        typedef enum mlx_dtype_ { MLX_FLOAT32, MLX_HEADER_ONLY } mlx_dtype;
        typedef void (*mlx_header_only_type)(void* data);
        int mlx_array_new(void);
        int mlx_array_free(void);
        int mlx_array_new_data(void);
        int mlx_array_new_data_managed(void);
        int _mlx_error(const char* format, ...);
        int mlx_header_only(void);
        """);
    Path nativePins = root.resolve("native/install/lib/native-pin.properties");
    Files.createDirectories(nativePins.getParent());
    Files.writeString(nativePins, "mlxMetalVersion=1.2.3\nmlxcCommit=abc\n");
    return new Fixture(root, headers, nativePins);
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
        "    public static class _mlx_error {",
        "        public static _mlx_error makeInvoker(MemoryLayout... layouts) {",
        "    public static class mlx_node_namer_new {",
        "        public static mlx_node_namer_new makeInvoker(MemoryLayout... layouts) {",
        "    public static int MLX_FLOAT32() {");
  }

  private record Fixture(Path root, Path headers, Path nativePins) {}
}
