package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.core.MLXQuant;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;

/**
 * An MLX-style quantized checkpoint loads and computes what its dequantized weights would. The
 * reference is a float checkpoint built from {@code dequantize(quantize(w))}, so the comparison
 * isolates the quantized wiring (projections, fused slices, embedding lookup, tied head) from the
 * quantization error itself.
 */
@EnabledIfNativeAvailable
class QuantizedDecoderTest {
  private static final int GROUP = 32;
  private static final int BITS = 4;

  /** Writes {@code quantized/} and {@code reference/} from one random float checkpoint. */
  private static void build(Path root, String family, boolean tied) throws Exception {
    Path source = root.resolve("source");
    if (family.equals("llama")) {
      TinyCheckpoints.randomLlama(source, 11L, 2, false, tied);
    } else {
      TinyCheckpoints.randomQwen2(source, 11L);
    }
    quantize(root, source);
  }

  /** Builds {@code quantized/} and {@code reference/} under {@code root} from {@code source}. */
  private static void quantize(Path root, Path source) throws Exception {
    Path quantized = Files.createDirectories(root.resolve("quantized"));
    Path reference = Files.createDirectories(root.resolve("reference"));
    String config = Files.readString(source.resolve("config.json"));
    Files.writeString(reference.resolve("config.json"), config);
    Files.writeString(
        quantized.resolve("config.json"),
        config.substring(0, config.lastIndexOf('}'))
            + ",\"quantization\":{\"group_size\":"
            + GROUP
            + ",\"bits\":"
            + BITS
            + "}}");
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> floats =
          MLXIO.loadSafetensors(scope, source.resolve("model.safetensors").toString()).tensors();
      Map<String, MLXArray> packed = new LinkedHashMap<>();
      Map<String, MLXArray> dequantized = new LinkedHashMap<>();
      for (var entry : floats.entrySet()) {
        String key = entry.getKey();
        boolean quantize =
            key.endsWith("_proj.weight")
                || key.equals("model.embed_tokens.weight")
                || key.equals("lm_head.weight");
        if (!quantize) {
          packed.put(key, entry.getValue());
          dequantized.put(key, entry.getValue());
          continue;
        }
        String stem = key.substring(0, key.length() - ".weight".length());
        MLXArray[] q = MLXQuant.quantize(entry.getValue(), GROUP, BITS, "affine", null);
        packed.put(key, q[0]);
        packed.put(stem + ".scales", q[1]);
        packed.put(stem + ".biases", q[2]);
        dequantized.put(
            key, MLXQuant.dequantize(q[0], q[1], q[2], GROUP, BITS, "affine", null, DType.FLOAT32));
      }
      MLXIO.saveSafetensors(quantized.resolve("model.safetensors").toString(), packed, Map.of());
      MLXIO.saveSafetensors(
          reference.resolve("model.safetensors").toString(), dequantized, Map.of());
    }
  }

  private static float[] logits(MLXScope scope, Path directory, int[] prompt) throws IOException {
    DecoderModel model = (DecoderModel) TextGenerationModels.load(scope, directory);
    List<KVCache> caches = new ArrayList<>();
    for (int i = 0; i < model.config().numHiddenLayers(); i++) {
      caches.add(new KVCache(scope));
    }
    MLXArray ids = MLX.array(scope, prompt, new int[] {1, prompt.length});
    return model.forward(ids, caches).toFloatArray();
  }

  private static void assertClose(float[] expected, float[] actual, float tolerance) {
    assertEquals(expected.length, actual.length);
    float worst = 0;
    for (int i = 0; i < expected.length; i++) {
      worst = Math.max(worst, Math.abs(expected[i] - actual[i]));
    }
    assertTrue(worst <= tolerance, "max |difference| " + worst + " exceeds " + tolerance);
  }

  @Test
  void untiedLlamaMatchesItsDequantizedWeights(@TempDir Path dir) throws Exception {
    build(dir, "llama", false);
    int[] prompt = {1, 5, 9, 3, 7};
    try (MLXScope scope = new MLXScope()) {
      assertClose(
          logits(scope, dir.resolve("reference"), prompt),
          logits(scope, dir.resolve("quantized"), prompt),
          1e-3f);
    }
  }

  @Test
  void tiedHeadUsesTheQuantizedEmbeddingTable(@TempDir Path dir) throws Exception {
    build(dir, "llama", true);
    int[] prompt = {2, 4, 6, 8};
    try (MLXScope scope = new MLXScope()) {
      assertClose(
          logits(scope, dir.resolve("reference"), prompt),
          logits(scope, dir.resolve("quantized"), prompt),
          1e-3f);
    }
  }

  @Test
  void qwen2KeepsItsFloatProjectionBiases(@TempDir Path dir) throws Exception {
    build(dir, "qwen2", false);
    int[] prompt = {1, 2, 3};
    try (MLXScope scope = new MLXScope()) {
      assertClose(
          logits(scope, dir.resolve("reference"), prompt),
          logits(scope, dir.resolve("quantized"), prompt),
          1e-3f);
    }
  }

  @Test
  void greedyGenerationAgreesWithTheDequantizedReference(@TempDir Path dir) throws Exception {
    build(dir, "llama", false);
    try (MLXScope scope = new MLXScope()) {
      GenerationRequest request =
          new GenerationRequest(
              new int[] {1, 5, 9},
              GenerationConfig.greedyDefaults(8, Set.of()),
              CancellationToken.NONE);
      List<Integer> expected =
          TextGenerationModels.load(scope, dir.resolve("reference"))
              .generate(request, e -> {})
              .generatedTokenIds();
      List<Integer> actual =
          TextGenerationModels.load(scope, dir.resolve("quantized"))
              .generate(request, e -> {})
              .generatedTokenIds();
      assertEquals(expected, actual);
    }
  }

  @Test
  void gemmaScaledEmbeddingAndTiedHeadMatchTheDequantizedWeights(@TempDir Path dir)
      throws Exception {
    assertMatchesDequantizedReference(dir, "gemma");
  }

  @Test
  void mistralMatchesTheDequantizedWeights(@TempDir Path dir) throws Exception {
    assertMatchesDequantizedReference(dir, "mistral");
  }

  @Test
  void qwen2SyntheticCheckpointMatchesTheDequantizedWeights(@TempDir Path dir) throws Exception {
    assertMatchesDequantizedReference(dir, "qwen2");
  }

  private static void assertMatchesDequantizedReference(Path dir, String family) throws Exception {
    Path source = dir.resolve("source");
    Path checkpoint =
        Path.of(System.getProperty("jmlx.repository.root"))
            .resolve("tools/hf-reference/goldens/checkpoints")
            .resolve(family);
    Files.createDirectories(source);
    Files.copy(checkpoint.resolve("config.json"), source.resolve("config.json"));
    Files.copy(checkpoint.resolve("model.safetensors"), source.resolve("model.safetensors"));
    quantize(dir, source);
    int[] prompt = {1, 5, 9, 3, 7};
    try (MLXScope scope = new MLXScope()) {
      assertClose(
          logits(scope, dir.resolve("reference"), prompt),
          logits(scope, dir.resolve("quantized"), prompt),
          1e-3f);
    }
  }

  @Test
  void phi3FusedProjectionsSliceTheirPackedRows(@TempDir Path dir) throws Exception {
    // Phi-3 stores qkv_proj and gate_up_proj fused; each is split by rows, which for a packed
    // weight must slice scales and biases by the same rows.
    assertMatchesDequantizedReference(dir, "phi3");
  }

  @Test
  void missingScalesTensorIsNamed(@TempDir Path dir) throws Exception {
    build(dir, "llama", false);
    Path quantized = dir.resolve("quantized");
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> tensors =
          new LinkedHashMap<>(
              MLXIO
                  .loadSafetensors(scope, quantized.resolve("model.safetensors").toString())
                  .tensors());
      tensors.remove("model.layers.0.mlp.up_proj.scales");
      MLXIO.saveSafetensors(quantized.resolve("model.safetensors").toString(), tensors, Map.of());
      IllegalArgumentException error =
          assertThrows(
              IllegalArgumentException.class, () -> TextGenerationModels.load(scope, quantized));
      assertTrue(error.getMessage().contains("up_proj.scales"), error.getMessage());
    }
  }

  @Test
  void floatCheckpointStillRejectsStrayQuantizationTensors(@TempDir Path dir) throws Exception {
    build(dir, "llama", false);
    Path reference = dir.resolve("reference");
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> tensors =
          new LinkedHashMap<>(
              MLXIO
                  .loadSafetensors(scope, reference.resolve("model.safetensors").toString())
                  .tensors());
      tensors.put("model.layers.0.mlp.up_proj.scales", tensors.get("model.norm.weight"));
      MLXIO.saveSafetensors(reference.resolve("model.safetensors").toString(), tensors, Map.of());
      IllegalArgumentException error =
          assertThrows(
              IllegalArgumentException.class, () -> TextGenerationModels.load(scope, reference));
      assertTrue(error.getMessage().contains("unexpected tensor"), error.getMessage());
    }
  }

  @Test
  void unsupportedQuantizationDeclarationsAreRejected(@TempDir Path dir) throws Exception {
    Path config = dir.resolve("config.json");
    String base =
        "{\"model_type\":\"llama\",\"vocab_size\":8,\"hidden_size\":64,\"intermediate_size\":128,"
            + "\"num_hidden_layers\":1,\"num_attention_heads\":4,\"quantization\":%s}";
    Map<String, String> cases =
        Map.of(
            "{\"group_size\":64,\"bits\":4,\"model.layers.0.mlp\":{\"bits\":8}}", "per-layer",
            "{\"group_size\":48,\"bits\":4}", "group_size",
            "{\"group_size\":64,\"bits\":7}", "bits",
            "{\"quant_method\":\"gptq\",\"group_size\":64,\"bits\":4}", "gptq",
            "{\"group_size\":64,\"bits\":4,\"mode\":\"mxfp4\"}", "mxfp4");
    for (var entry : cases.entrySet()) {
      Files.writeString(config, base.formatted(entry.getKey()));
      IllegalArgumentException error =
          assertThrows(IllegalArgumentException.class, () -> DecoderConfig.fromFile(config));
      assertTrue(error.getMessage().contains(entry.getValue()), error.getMessage());
    }
    Files.writeString(config, base.formatted("{\"group_size\":64,\"bits\":4}"));
    assertEquals(64, DecoderConfig.fromFile(config).hiddenSize());
  }
}
