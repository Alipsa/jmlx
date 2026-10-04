package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

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
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;

/** Qwen3 loads through the common entry point with per-head QK normalization. */
@EnabledIfNativeAvailable
class Qwen3ModelTest {
  @Test
  void loadsQwen3AndGeneratesReproduciblyThroughTheCommonEntryPoint(@TempDir Path dir)
      throws Exception {
    TinyCheckpoints.randomQwen3(dir, 11L, 2, false, false);
    try (MLXScope modelScope = new MLXScope()) {
      DecoderModel model = (DecoderModel) TextGenerationModels.load(modelScope, dir);
      // tokenIds() is prompt plus generated: three prompt tokens, four generated.
      List<Integer> first = model.generate(new int[] {1, 5, 9}, 4, Set.of());
      List<Integer> second = model.generate(new int[] {1, 5, 9}, 4, Set.of());
      assertEquals(7, first.size());
      assertEquals(first, second);
      assertEquals(
          new ArrayList<>(first.subList(3, 7)),
          model
              .generate(
                  new GenerationRequest(
                      new int[] {1, 5, 9}, GenerationConfig.greedyDefaults(4, Set.of()),
                      CancellationToken.NONE),
                  ignored -> {})
              .generatedTokenIds());
    }
  }

  @Test
  void tiedQwen3CheckpointOmitsLmHead(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomQwen3(dir, 13L, 2, false, true);
    try (MLXScope modelScope = new MLXScope()) {
      DecoderModel model = (DecoderModel) TextGenerationModels.load(modelScope, dir);
      assertEquals(4, model.generate(new int[] {2, 4}, 2, Set.of()).size());
    }
  }

  @Test
  void attentionBiasVariantCarriesAllFourProjectionBiases(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomQwen3(dir, 17L, 2, true, false);
    try (MLXScope modelScope = new MLXScope()) {
      DecoderModel model = (DecoderModel) TextGenerationModels.load(modelScope, dir);
      assertEquals(4, model.generate(new int[] {3, 7}, 2, Set.of()).size());
    }
  }

  @Test
  void cachedDecodeAgreesWithFullForward(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomQwen3(dir, 19L, 2, false, false);
    int[] prompt = {1, 5, 9, 3, 7};
    try (MLXScope modelScope = new MLXScope()) {
      DecoderModel model = (DecoderModel) TextGenerationModels.load(modelScope, dir);
      MLXArray ids = MLX.array(modelScope, prompt, new int[] {1, prompt.length});
      float[] full = lastRowLogits(model, modelScope, ids, newCaches(model, modelScope));
      List<KVCache> caches = newCaches(model, modelScope);
      for (int t = 0; t < prompt.length - 1; t++) {
        model.forward(MLX.array(modelScope, new int[] {prompt[t]}, new int[] {1, 1}), caches);
      }
      float[] stepped = lastRowLogits(model, modelScope,
          MLX.array(modelScope, new int[] {prompt[prompt.length - 1]}, new int[] {1, 1}), caches);
      assertEquals(full.length, stepped.length);
      float worst = 0;
      for (int i = 0; i < full.length; i++) {
        worst = Math.max(worst, Math.abs(full[i] - stepped[i]));
      }
      assertTrue(worst <= 1e-3f, "max |difference| " + worst + " exceeds 1e-3");
    }
  }

  @Test
  void missingQkNormWeightIsRejectedByName(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomQwen3(dir, 23L, 2, false, false);
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> tensors =
          new LinkedHashMap<>(
              MLXIO.loadSafetensors(scope, dir.resolve("model.safetensors").toString()).tensors());
      tensors.remove("model.layers.0.self_attn.q_norm.weight");
      MLXIO.saveSafetensors(dir.resolve("model.safetensors").toString(), tensors, Map.of());
      IllegalArgumentException error =
          assertThrows(
              IllegalArgumentException.class, () -> TextGenerationModels.load(scope, dir));
      assertTrue(error.getMessage().contains("q_norm.weight"), error.getMessage());
    }
  }

  @Test
  void packedQkNormWeightIsRejected(@TempDir Path dir) throws Exception {
    // q_norm/k_norm are never quantized; a packed norm weight must fail at load.
    TinyCheckpoints.randomQwen3(dir, 29L, 2, false, false);
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> tensors =
          new LinkedHashMap<>(
              MLXIO.loadSafetensors(scope, dir.resolve("model.safetensors").toString()).tensors());
      tensors.put(
          "model.layers.0.self_attn.q_norm.weight",
          MLX.astype(MLX.zeros(scope, new int[] {32}, DType.FLOAT32), DType.UINT32));
      MLXIO.saveSafetensors(dir.resolve("model.safetensors").toString(), tensors, Map.of());
      IllegalArgumentException error =
          assertThrows(
              IllegalArgumentException.class, () -> TextGenerationModels.load(scope, dir));
      assertTrue(error.getMessage().contains("q_norm"), error.getMessage());
    }
  }

  @Test
  void qkNormScalesAreRejectedAsUnexpected(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomQwen3(dir, 31L, 2, false, false);
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> tensors =
          new LinkedHashMap<>(
              MLXIO.loadSafetensors(scope, dir.resolve("model.safetensors").toString()).tensors());
      tensors.put(
          "model.layers.0.self_attn.q_norm.scales", tensors.get("model.norm.weight"));
      MLXIO.saveSafetensors(dir.resolve("model.safetensors").toString(), tensors, Map.of());
      IllegalArgumentException error =
          assertThrows(
              IllegalArgumentException.class, () -> TextGenerationModels.load(scope, dir));
      assertTrue(error.getMessage().contains("unexpected tensor"), error.getMessage());
      assertTrue(error.getMessage().contains("q_norm.scales"), error.getMessage());
    }
  }

  @Test
  void typedQwenEntryPointStillRequiresQwen2(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomQwen3(dir, 37L, 2, false, false);
    try (MLXScope scope = new MLXScope()) {
      IllegalArgumentException error =
          assertThrows(IllegalArgumentException.class, () -> QwenModel.load(scope, dir));
      assertTrue(error.getMessage().contains("expected model_type qwen2"), error.getMessage());
    }
  }

  private static List<KVCache> newCaches(DecoderModel model, MLXScope scope) {
    List<KVCache> caches = new ArrayList<>();
    for (int i = 0; i < model.config().numHiddenLayers(); i++) {
      caches.add(new KVCache(scope));
    }
    return caches;
  }

  private static float[] lastRowLogits(
      DecoderModel model, MLXScope scope, MLXArray ids, List<KVCache> caches) {
    float[] logits = model.forward(ids, caches).toFloatArray();
    int vocab = logits.length / (ids.shape()[0] * ids.shape()[1]);
    return java.util.Arrays.copyOfRange(
        logits, (ids.shape()[0] * ids.shape()[1] - 1) * vocab, ids.shape()[0] * ids.shape()[1] * vocab);
  }
}
