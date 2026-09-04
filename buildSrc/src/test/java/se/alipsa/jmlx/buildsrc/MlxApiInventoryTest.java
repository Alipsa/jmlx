package se.alipsa.jmlx.buildsrc;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
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
    assertTrue(first.contains("generated entries: 9"));
    assertTrue(first.contains("downcall=6, constant=1, layout/accessor=1, upcall interface=1"));
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
  void identifiesTheMappingFileWhenItsJsonIsMalformed() throws Exception {
    Path root = fixture("{\"records\":[");

    IllegalArgumentException failure =
        assertThrows(IllegalArgumentException.class, () -> MlxApiInventory.render(root));

    assertTrue(failure.getMessage().contains("Invalid inventory mapping file"));
    assertTrue(failure.getMessage().contains("mlx-api-inventory-overrides.json"));
  }

  @Test
  void identifiesTheRecordAndFieldForMappingSchemaErrors() throws Exception {
    Path root =
        fixture(
            "{\"records\":[{\"binding\":\"mlx_h.mlx_array_new\",\"category\":\"downcall\","
                + "\"status\":\"implemented\",\"facadeOrReason\":\"reason\",\"tests\":\"test\","
                + "\"notes\":\"remove me\"}]}");

    IllegalArgumentException failure =
        assertThrows(IllegalArgumentException.class, () -> MlxApiInventory.render(root));

    assertTrue(failure.getMessage().contains("mlx-api-inventory-overrides.json"));
    assertTrue(failure.getMessage().contains("at index 0 for 'mlx_h.mlx_array_new'"));
    assertTrue(failure.getMessage().contains("unknown field: 'notes'"));
  }

  @Test
  void callSiteGuardRejectsOnlyUnmappedSyntaxUses() throws Exception {
    Path root = fixture(record("mlx_h.mlx_array_new", "implemented"));
    Path source = root.resolve("jmlx-core/src/main/java/sample/Example.java");
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

  @Test
  void callSiteGuardRejectsGeneratedAliasesQualifiedUsesAndStarImports() throws Exception {
    Path root = fixture(record("mlx_h.mlx_array_new", "implemented"));
    Files.writeString(
        root.resolve("jmlx-ffi/src/main/generated/java/se/alipsa/jmlx/ffi/mlx_array.java"), "");
    Path source = root.resolve("jmlx-core/src/main/java/sample/Example.java");
    Files.createDirectories(source.getParent());
    Files.writeString(
        source,
        """
        import se.alipsa.jmlx.ffi.*;
        class Example {
          void use(java.lang.foreign.MemorySegment segment) {
            se.alipsa.jmlx.ffi.mlx_array.ctx(segment);
          }
        }
        """);

    IllegalStateException failure =
        assertThrows(IllegalStateException.class, () -> MlxApiCallSites.verify(root));

    assertTrue(failure.getMessage().contains("mlx_array"));
    assertTrue(failure.getMessage().contains("star imports are not allowed"));
    assertTrue(failure.getMessage().contains("replace the star import with explicit"));
  }

  @Test
  void callSiteGuardRejectsMethodReferencesAndFullyQualifiedTypes() throws Exception {
    Path root = fixture(record("mlx_h.mlx_array_new", "implemented"));
    Files.writeString(
        root.resolve("jmlx-ffi/src/main/generated/java/se/alipsa/jmlx/ffi/mlx_array.java"), "");
    Path source = root.resolve("jmlx-core/src/main/java/sample/Example.java");
    Files.createDirectories(source.getParent());
    Files.writeString(
        source,
        """
        import java.util.function.Supplier;
        import se.alipsa.jmlx.ffi.mlx_h;
        class Example {
          Supplier<Object> one = mlx_h::mlx_array_free;
          Supplier<Object> two = se.alipsa.jmlx.ffi.mlx_h::mlx_array_free;
          Supplier<Object> three = se.alipsa.jmlx.ffi.mlx_array::ctx;
        }
        """);

    IllegalStateException failure =
        assertThrows(IllegalStateException.class, () -> MlxApiCallSites.verify(root));

    assertTrue(failure.getMessage().contains("mlx_h.mlx_array_free"));
    assertTrue(failure.getMessage().contains("mlx_array"));
  }

  @Test
  void callSiteGuardFailsWhenAHandwrittenSourceCannotBeParsed() throws Exception {
    Path root = fixture("{\"records\":[]}");
    Path source = root.resolve("jmlx-core/src/main/java/sample/Broken.java");
    Files.createDirectories(source.getParent());
    Files.writeString(
        source,
        "import se.alipsa.jmlx.ffi.mlx_h; class Broken { void use() { mlx_h.mlx_array_free(); } }"
            + " }");

    IllegalStateException failure =
        assertThrows(IllegalStateException.class, () -> MlxApiCallSites.verify(root));

    assertTrue(failure.getMessage().contains("Unable to parse handwritten Java source"));
  }

  @Test
  void callSiteGuardRejectsAnInsufficientParserRuntime() throws Exception {
    Path root = fixture("{\"records\":[]}");

    IllegalStateException failure =
        assertThrows(
            IllegalStateException.class,
            () -> MlxApiCallSites.verify(root, Runtime.version().feature() + 1));

    assertTrue(failure.getMessage().contains("verifyMlxApiCallSites requires JDK >="));
  }

  @Test
  void callSiteGuardRejectsMappingsWithoutObservedUses() throws Exception {
    Path root = fixture(record("mlx_h.mlx_array_new", "implemented"));

    IllegalStateException failure =
        assertThrows(IllegalStateException.class, () -> MlxApiCallSites.verify(root));

    assertTrue(
        failure.getMessage().contains("remove the stale inventory mapping"), failure::getMessage);
    assertTrue(failure.getMessage().contains("MLX binding inventory guard failures"));
  }

  @Test
  void callSiteGuardAllowsDocumentationOnlyMappingsWithoutUses() throws Exception {
    Path root = fixture(record("mlx_h.mlx_array_new", "unsupported-by-runtime"));
    Path planned = fixture(record("mlx_h.mlx_array_new", "planned"));

    assertDoesNotThrow(() -> MlxApiCallSites.verify(root));
    assertDoesNotThrow(() -> MlxApiCallSites.verify(planned));
  }

  @Test
  void rejectsVariadicCandidatesThatDoNotMatchTheInvokerShape() throws Exception {
    Path root = fixture(record("mlx_h.mlx_array_new", "implemented"));
    Path binding = root.resolve("jmlx-ffi/src/main/generated/java/se/alipsa/jmlx/ffi/mlx_h.java");
    Files.writeString(
        binding, binding().replace("makeInvoker(MemoryLayout... layouts)", "makeInvoker()"));

    IllegalStateException failure =
        assertThrows(IllegalStateException.class, () -> MlxApiInventory.render(root));

    assertTrue(failure.getMessage().contains("downcall implementations matched"));
  }

  @Test
  void rejectsVariadicInvokerClassDeclarationChanges() throws Exception {
    Path root = fixture(record("mlx_h.mlx_array_new", "implemented"));
    Path binding = root.resolve("jmlx-ffi/src/main/generated/java/se/alipsa/jmlx/ffi/mlx_h.java");
    Files.writeString(
        binding,
        binding()
            .replace("public static class _mlx_error", "public static final class _mlx_error"));

    IllegalStateException failure =
        assertThrows(IllegalStateException.class, () -> MlxApiInventory.render(root));

    assertTrue(failure.getMessage().contains("Expected variadic invoker canary bindings missing"));
  }

  @Test
  void callSiteGuardRestrictsScopedMappingsToTheirExactSource() throws Exception {
    String mappings =
        "{\"records\":[{\"binding\":\"mlx_h.mlx_array_free\",\"category\":\"downcall\","
            + "\"status\":\"planned\",\"facadeOrReason\":\"probe\",\"tests\":\"test\","
            + "\"sources\":[\"jmlx-core/src/test/java/sample/Allowed.java\"]}]}";
    Path root = fixture(mappings);
    Path allowed = root.resolve("jmlx-core/src/test/java/sample/Allowed.java");
    Files.createDirectories(allowed.getParent());
    Files.writeString(allowed, "class Allowed {}");
    Path source = root.resolve("jmlx-core/src/main/java/sample/Disallowed.java");
    Files.createDirectories(source.getParent());
    Files.writeString(
        source,
        "import se.alipsa.jmlx.ffi.mlx_h; class Disallowed { void use() { mlx_h.mlx_array_free(); }"
            + " }");

    IllegalStateException failure =
        assertThrows(IllegalStateException.class, () -> MlxApiCallSites.verify(root));

    assertTrue(failure.getMessage().contains("mlx_h.mlx_array_free"));
  }

  @Test
  void rejectsStaleScopedMappingPath() throws Exception {
    String mappings =
        "{\"records\":[{\"binding\":\"mlx_h.mlx_array_free\",\"category\":\"downcall\","
            + "\"status\":\"planned\",\"facadeOrReason\":\"probe\",\"tests\":\"test\","
            + "\"sources\":[\"jmlx-core/src/test/java/sample/Missing.java\"]}]}";

    IllegalArgumentException failure =
        assertThrows(
            IllegalArgumentException.class, () -> MlxApiInventory.render(fixture(mappings)));

    assertTrue(failure.getMessage().contains("Stale inventory mapping source path"));
  }

  @Test
  void callSiteGuardExcludesGeneratedSourcesAndNamedJextractInfrastructure() throws Exception {
    Path root = fixture("{\"records\":[]}");
    Files.writeString(
        root.resolve("jmlx-ffi/src/main/generated/java/se/alipsa/jmlx/ffi/Generated.java"),
        "class Generated { void use() { mlx_h.mlx_array_free(); } }");
    Path source = root.resolve("jmlx-core/src/main/java/sample/Infrastructure.java");
    Files.createDirectories(source.getParent());
    Files.writeString(
        source,
        "import se.alipsa.jmlx.ffi.mlx_h; class Infrastructure { Object value = mlx_h.C_POINTER;"
            + " }");

    assertDoesNotThrow(() -> MlxApiCallSites.verify(root));
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
        "    public static class _mlx_error {",
        "        public static _mlx_error makeInvoker(MemoryLayout... layouts) {",
        "    public static class mlx_node_namer_new {",
        "        public static mlx_node_namer_new makeInvoker(MemoryLayout... layouts) {",
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
