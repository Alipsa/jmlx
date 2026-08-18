package se.alipsa.jmlx.core;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Round-trip tests only (req/plans/phase5-m1-plan.md Task 4): every fixture here is written by
 * {@link MLXIO} itself, so this suite proves the facade is internally self-consistent, not interop
 * with a real HF/llama.cpp checkpoint -- that is deferred to M3 (req/phase5-plan.md's Testing
 * approach).
 */
@EnabledIfNativeAvailable
class MLXIOTest {

  @Test
  void saveThenLoadSafetensorsRoundTripsTensorValues(@TempDir Path dir) {
    String file = dir.resolve("tensors.safetensors").toString();
    try (MLXScope saveScope = new MLXScope()) {
      Map<String, MLXArray> tensors = new LinkedHashMap<>();
      tensors.put("a", MLX.array(saveScope, new float[] {1f, 2f, 3f, 4f}, new int[] {2, 2}));
      tensors.put("b", MLX.array(saveScope, new float[] {5f, 6f}, new int[] {2}));
      MLXIO.saveSafetensors(file, tensors, Map.of());
    }
    try (MLXScope loadScope = new MLXScope()) {
      MLXIO.SafetensorsResult result = MLXIO.loadSafetensors(loadScope, file);
      assertEquals(Set.of("a", "b"), result.tensors().keySet());
      assertArrayEquals(new float[] {1f, 2f, 3f, 4f}, result.tensors().get("a").toFloatArray());
      assertArrayEquals(new int[] {2, 2}, result.tensors().get("a").shape());
      assertArrayEquals(new float[] {5f, 6f}, result.tensors().get("b").toFloatArray());
    }
  }

  @Test
  void saveThenLoadSafetensorsRoundTripsMetadata(@TempDir Path dir) {
    String file = dir.resolve("meta.safetensors").toString();
    Map<String, String> metadata = new LinkedHashMap<>();
    metadata.put("format", "pt");
    metadata.put("author", "jmlx");
    try (MLXScope saveScope = new MLXScope()) {
      Map<String, MLXArray> tensors =
          Map.of("w", MLX.array(saveScope, new float[] {1f}, new int[] {1}));
      MLXIO.saveSafetensors(file, tensors, metadata);
    }
    try (MLXScope loadScope = new MLXScope()) {
      MLXIO.SafetensorsResult result = MLXIO.loadSafetensors(loadScope, file);
      assertEquals(metadata, result.metadata());
    }
  }

  @Test
  void loadSafetensorsEmptyMetadataMapIsEmptyNotNull(@TempDir Path dir) {
    String file = dir.resolve("nometa.safetensors").toString();
    try (MLXScope saveScope = new MLXScope()) {
      Map<String, MLXArray> tensors =
          Map.of("w", MLX.array(saveScope, new float[] {1f}, new int[] {1}));
      MLXIO.saveSafetensors(file, tensors, Map.of());
    }
    try (MLXScope loadScope = new MLXScope()) {
      MLXIO.SafetensorsResult result = MLXIO.loadSafetensors(loadScope, file);
      assertEquals(Map.of(), result.metadata());
    }
  }

  @Test
  void saveThenLoadGgufRoundTripsTensorsAndAllThreeMetadataKinds(@TempDir Path dir) {
    String file = dir.resolve("model.gguf").toString();
    try (MLXScope saveScope = new MLXScope()) {
      Map<String, MLXArray> tensors =
          Map.of("weight", MLX.array(saveScope, new float[] {1f, 2f, 3f}, new int[] {3}));
      Map<String, MLXArray> metaArrays =
          Map.of("some.count", MLX.array(saveScope, new float[] {42f}, new int[] {1}));
      Map<String, String> metaStrings = Map.of("general.name", "test-model");
      Map<String, List<String>> metaVectorStrings =
          Map.of("tokenizer.vocab", List.of("<s>", "</s>", "hello"));
      MLXIO.saveGguf(file, tensors, metaArrays, metaStrings, metaVectorStrings);
    }
    try (MLXScope loadScope = new MLXScope()) {
      MLXIO.GgufResult result =
          MLXIO.loadGguf(
              loadScope,
              file,
              Set.of("some.count"),
              Set.of("general.name"),
              Set.of("tokenizer.vocab"));
      assertEquals(Set.of("weight"), result.tensors().keySet());
      assertArrayEquals(new float[] {1f, 2f, 3f}, result.tensors().get("weight").toFloatArray());
      assertEquals(Set.of("some.count"), result.metadataArrays().keySet());
      assertArrayEquals(
          new float[] {42f}, result.metadataArrays().get("some.count").toFloatArray());
      assertEquals("test-model", result.metadataStrings().get("general.name"));
      assertEquals(
          List.of("<s>", "</s>", "hello"), result.metadataVectorStrings().get("tokenizer.vocab"));
    }
  }

  @Test
  void loadGgufMetadataVectorStringPreservesOrder(@TempDir Path dir) {
    String file = dir.resolve("order.gguf").toString();
    List<String> values = List.of("zeta", "alpha", "mu", "beta");
    try (MLXScope saveScope = new MLXScope()) {
      Map<String, MLXArray> tensors =
          Map.of("w", MLX.array(saveScope, new float[] {1f}, new int[] {1}));
      MLXIO.saveGguf(file, tensors, Map.of(), Map.of(), Map.of("list", values));
    }
    try (MLXScope loadScope = new MLXScope()) {
      MLXIO.GgufResult result = MLXIO.loadGguf(loadScope, file, Set.of(), Set.of(), Set.of("list"));
      assertEquals(values, result.metadataVectorStrings().get("list"));
    }
  }

  @Test
  void loadGgufRequestedMetadataKeyAbsentIsOmittedNotThrown(@TempDir Path dir) {
    String file = dir.resolve("nometa.gguf").toString();
    try (MLXScope saveScope = new MLXScope()) {
      Map<String, MLXArray> tensors =
          Map.of("w", MLX.array(saveScope, new float[] {1f}, new int[] {1}));
      MLXIO.saveGguf(file, tensors, Map.of(), Map.of(), Map.of());
    }
    try (MLXScope loadScope = new MLXScope()) {
      MLXIO.GgufResult result =
          MLXIO.loadGguf(
              loadScope,
              file,
              Set.of("does.not.exist"),
              Set.of("nor.this"),
              Set.of("nor.this.either"));
      assertEquals(Map.of(), result.metadataArrays());
      assertEquals(Map.of(), result.metadataStrings());
      assertEquals(Map.of(), result.metadataVectorStrings());
    }
  }

  @Test
  void loadSafetensorsUnknownFileThrowsMLXException(@TempDir Path dir) {
    String file = dir.resolve("does-not-exist.safetensors").toString();
    try (MLXScope scope = new MLXScope()) {
      assertThrows(MLXException.class, () -> MLXIO.loadSafetensors(scope, file));
    }
  }

  @Test
  void loadSafetensorsAllocatesIntoTheGivenScope(@TempDir Path dir) {
    String file = dir.resolve("scoped.safetensors").toString();
    try (MLXScope saveScope = new MLXScope()) {
      Map<String, MLXArray> tensors =
          Map.of("w", MLX.array(saveScope, new float[] {1f}, new int[] {1}));
      MLXIO.saveSafetensors(file, tensors, Map.of());
    }
    try (MLXScope root = new MLXScope()) {
      MLXScope child = root.newChild();
      MLXIO.SafetensorsResult result = MLXIO.loadSafetensors(child, file);
      MLXArray w = result.tensors().get("w");
      child.close();
      assertThrows(IllegalStateException.class, w::shape);
    }
  }

  @Test
  void saveSafetensorsUnwritablePathThrowsMLXException(@TempDir Path dir) {
    String file = dir.resolve("missing-subdir").resolve("out.safetensors").toString();
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> tensors =
          Map.of("w", MLX.array(scope, new float[] {1f}, new int[] {1}));
      assertThrows(MLXException.class, () -> MLXIO.saveSafetensors(file, tensors, Map.of()));
    }
  }
}
