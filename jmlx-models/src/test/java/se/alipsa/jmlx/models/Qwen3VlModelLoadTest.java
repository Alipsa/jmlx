package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.core.MLXQuant;
import se.alipsa.jmlx.core.MLXRandom;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.vision.RgbImage;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ObjectNode;

/**
 * Native load, dispatch and request contracts for the composite Qwen3-VL model on the tiny HF float
 * checkpoint and the MLX affine-quantized conversion of it: metadata and the dtype up-cast,
 * rejection paths that fire before pixels or native work, and deterministic greedy generation with
 * and without one image.
 */
@EnabledIfNativeAvailable
class Qwen3VlModelLoadTest {

  // The fixture sentinels (qwen3vl-tokens/tokenizer.json): image 10, video 11, start 12, end 13.
  private static final int IMAGE = 10;
  private static final int VIDEO = 11;
  private static final int START = 12;
  private static final int END = 13;

  /**
   * One deterministic 64x64 image: per-pixel linear functions of the coordinates. At the fixture's
   * geometry it resizes to a 4x4 patch grid, merging to four feature rows (expanded length 8).
   */
  private static RgbImage syntheticImage(int side) {
    byte[] pixels = new byte[side * side * 3];
    for (int y = 0; y < side; y++) {
      for (int x = 0; x < side; x++) {
        int i = (y * side + x) * 3;
        pixels[i] = (byte) ((x * 7 + y * 11) % 256);
        pixels[i + 1] = (byte) ((x * 13 + y * 5) % 256);
        pixels[i + 2] = (byte) ((x * 19 + y * 3) % 256);
      }
    }
    return new RgbImage(side, side, pixels);
  }

  private static GenerationRequest imageRequest(int maxNewTokens) {
    return new GenerationRequest(
            new int[] {START, IMAGE, END, 1, 2},
            GenerationConfig.greedyDefaults(maxNewTokens, Set.of()),
            CancellationToken.NONE)
        .withImages(List.of(syntheticImage(64)));
  }

  private static Path copyDirectory(Path source, Path target) throws IOException {
    Files.createDirectories(target);
    try (var files = Files.list(source)) {
      for (Path file : files.filter(Files::isRegularFile).toList()) {
        Files.copy(file, target.resolve(file.getFileName()));
      }
    }
    return target;
  }

  private static void rewriteJson(Path directory, String file, Consumer<ObjectNode> change)
      throws IOException {
    ObjectMapper mapper = new ObjectMapper();
    ObjectNode tree = (ObjectNode) mapper.readTree(directory.resolve(file).toFile());
    change.accept(tree);
    Files.writeString(directory.resolve(file), mapper.writeValueAsString(tree));
  }

  private static void assertLoadRejected(MLXScope scope, Path directory, String expected) {
    IllegalArgumentException failure =
        assertThrows(IllegalArgumentException.class, () -> Qwen3VlModel.load(scope, directory));
    assertTrue(
        failure.getMessage().contains(expected),
        () -> "expected message containing " + expected + ", got " + failure.getMessage());
  }

  private static Path tinyCheckpoint(Path dir, long seed) throws IOException {
    Path checkpoint = dir.resolve("checkpoint");
    TinyCheckpoints.randomQwen3Vl(checkpoint, seed);
    return checkpoint;
  }

  @Test
  void loadReportsMetadataTowerAccessorsAndFloat32Weights(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomQwen3Vl(dir, 41L);
    try (MLXScope root = new MLXScope()) {
      Qwen3VlModel model = Qwen3VlModel.load(root, dir);
      VisionLanguageMetadata metadata =
          assertInstanceOf(VisionLanguageMetadata.class, model.metadata());
      assertEquals("qwen3_vl", metadata.modelType());
      assertEquals(21, metadata.vocabSize());
      assertEquals(2, metadata.numHiddenLayers());
      assertEquals(2, metadata.numEncoderLayers());
      assertEquals(Set.of(InputModality.TEXT, InputModality.IMAGE), metadata.inputModalities());
      Qwen3VlDescriptor d = model.descriptorForTest();
      assertEquals(new Qwen3VlPromptTokens(IMAGE, -1, START, END), d.tokens());
      assertEquals(VIDEO, d.videoTokenId());
      assertEquals(256, d.contextLength());
      assertArrayEquals(new int[] {8, 4, 4}, d.mropeSection());
      Qwen3VlVisionTower tower = model.towerForTest();
      assertEquals(2, tower.depth());
      assertEquals(4, tower.numHeads());
      assertEquals(64, tower.hidden());
      assertEquals(16, tower.headDim());
      assertEquals(16, tower.gridSide());
      assertEquals(2, tower.spatialMergeSize());
      assertEquals(64, tower.outHiddenSize());
      assertArrayEquals(new int[] {1}, tower.deepstackIndexes());
      // The header-level preflight sees the tiny checkpoint's exact 63-tensor name set
      // (24 text + 39 visual).
      CheckpointLoader.Preflight preflight = CheckpointLoader.preflight(dir, d.tensorPlan());
      assertEquals(1, preflight.shards().size());
      assertEquals(63, preflight.tensorNames().size());
      // The BF16 storage is up-cast: every live parameter is FLOAT32.
      assertTrue(
          model.decoderForTest().parameters().values().stream()
              .allMatch(a -> a.dtype() == DType.FLOAT32));
      assertTrue(tower.parameters().values().stream().allMatch(a -> a.dtype() == DType.FLOAT32));
    }
  }

  @Test
  void pytorchFormatMetadataLoadsThroughDispatch(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomQwen3Vl(dir, 43L);
    try (MLXScope root = new MLXScope()) {
      Map<String, MLXArray> tensors =
          MLXIO.loadSafetensors(root, dir.resolve("model.safetensors").toString()).tensors();
      MLX.eval(tensors.values().toArray(MLXArray[]::new));
      MLXIO.saveSafetensors(
          dir.resolve("model.safetensors").toString(), tensors, Map.of("format", "pt"));
      TextGenerationModel model = TextGenerationModels.load(root, dir);
      assertInstanceOf(Qwen3VlModel.class, model);
      assertEquals(
          2,
          model
              .generate(textRequest(new int[] {1, 2, 3}, 2), ignored -> {})
              .generatedTokenIds()
              .size());
    }
  }

  @Test
  void wideAttentionLoadsAndRunsTextAndImagePrefillAndDecode(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomQwen3Vl(dir, 44L);
    rewriteJson(
        dir,
        "config.json",
        config -> ((ObjectNode) config.get("text_config")).put("num_attention_heads", 8));
    try (MLXScope root = new MLXScope()) {
      Map<String, MLXArray> tensors =
          new LinkedHashMap<>(
              MLXIO.loadSafetensors(root, dir.resolve("model.safetensors").toString()).tensors());
      // The fixture has hidden=64, head_dim=32: eight query heads give attention width 256.
      MLXRandom.seed(4401L);
      for (int layer = 0; layer < 2; layer++) {
        String prefix = "model.language_model.layers." + layer + ".self_attn.";
        tensors.put(
            prefix + "q_proj.weight",
            MLXRandom.normal(root, new int[] {256, 64}, DType.BFLOAT16, 0f, 0.02f));
        tensors.put(
            prefix + "o_proj.weight",
            MLXRandom.normal(root, new int[] {64, 256}, DType.BFLOAT16, 0f, 0.02f));
      }
      MLX.eval(tensors.values().toArray(MLXArray[]::new));
      MLXIO.saveSafetensors(
          dir.resolve("model.safetensors").toString(), tensors, Map.of("format", "pt"));
      Qwen3VlModel model = (Qwen3VlModel) TextGenerationModels.load(root, dir);
      for (GenerationRequest request :
          List.of(textRequest(new int[] {1, 2, 3}, 3), imageRequest(3))) {
        GenerationResult result = model.generate(request, ignored -> {});
        assertEquals(3, result.generatedTokenIds().size());
        assertEquals(FinishReason.MAX_TOKENS, result.finishReason());
      }
    }
  }

  @Test
  void dispatchLoadsTheCompositeAndRejectsSiblingTypes(@TempDir Path dir) throws Exception {
    Path checkpoint = tinyCheckpoint(dir, 42L);
    try (MLXScope root = new MLXScope()) {
      assertInstanceOf(Qwen3VlModel.class, TextGenerationModels.load(root, checkpoint));
    }
    for (String type : new String[] {"qwen3_vl_moe", "qwen3_vl_x"}) {
      Path bad = dir.resolve(type);
      Files.createDirectories(bad);
      Files.writeString(bad.resolve("config.json"), "{\"model_type\": \"" + type + "\"}");
      try (MLXScope scope = new MLXScope()) {
        IllegalArgumentException failure =
            assertThrows(
                IllegalArgumentException.class, () -> TextGenerationModels.load(scope, bad));
        assertEquals(
            "model_type '" + type + "' is unsupported: only the dense qwen3_vl architecture loads",
            failure.getMessage());
      }
    }
  }

  @Test
  void textOnlyGenerationIsDeterministicAcrossLoads(@TempDir Path dir) throws Exception {
    int[] prompt = {1, 2, 3};
    TinyCheckpoints.randomQwen3Vl(dir.resolve("a"), 7L);
    TinyCheckpoints.randomQwen3Vl(dir.resolve("b"), 7L);
    GenerationResult first;
    try (MLXScope root = new MLXScope()) {
      first =
          Qwen3VlModel.load(root, dir.resolve("a")).generate(textRequest(prompt, 3), ignored -> {});
    }
    GenerationResult second;
    try (MLXScope root = new MLXScope()) {
      second =
          Qwen3VlModel.load(root, dir.resolve("b")).generate(textRequest(prompt, 3), ignored -> {});
    }
    assertEquals(first, second);
    assertEquals(3, first.generatedTokenIds().size());
    assertEquals(FinishReason.MAX_TOKENS, first.finishReason());
    assertEquals(3, first.promptPositions());
    assertEquals(Arrays.stream(prompt).boxed().toList(), first.promptTokenIds());
  }

  private static GenerationRequest textRequest(int[] prompt, int maxNewTokens) {
    return new GenerationRequest(
        prompt, GenerationConfig.greedyDefaults(maxNewTokens, Set.of()), CancellationToken.NONE);
  }

  @Test
  void singleImagePrefillRunsTheTowerOnceAndGenerates(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomQwen3Vl(dir, 8L);
    try (MLXScope root = new MLXScope()) {
      Qwen3VlModel model = Qwen3VlModel.load(root, dir);
      List<String> executions = new ArrayList<>();
      model.setObserverForTest(executions::add);
      GenerationRequest request = imageRequest(2);
      GenerationResult result = model.generate(request, ignored -> {});
      assertEquals(8, result.promptPositions());
      assertEquals(2, result.generatedTokenIds().size());
      assertEquals(FinishReason.MAX_TOKENS, result.finishReason());
      assertEquals(
          Arrays.stream(request.promptTokenIds()).boxed().toList(), result.promptTokenIds());
      assertEquals(1, executions.stream().filter("preprocess"::equals).count());
      assertEquals(1, executions.stream().filter("vision"::equals).count());
      assertEquals(1, executions.stream().filter("prefill-closed"::equals).count());
    }
  }

  @Test
  void zeroTokenImageRequestSkipsPixelAndNativeWork(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomQwen3Vl(dir, 9L);
    Qwen3VlModel model;
    try (MLXScope root = new MLXScope()) {
      model = Qwen3VlModel.load(root, dir);
    }
    List<String> executions = new ArrayList<>();
    model.setObserverForTest(executions::add);
    GenerationResult empty = model.generate(imageRequest(0), ignored -> {});
    assertTrue(empty.generatedTokenIds().isEmpty());
    assertEquals(8, empty.promptPositions());
    assertTrue(executions.isEmpty(), "zero-step must skip pixel and native work");
  }

  @Test
  void slidingVideoAndOversizedPromptsAreRejected(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomQwen3Vl(dir, 10L);
    try (MLXScope root = new MLXScope()) {
      Qwen3VlModel model = Qwen3VlModel.load(root, dir);
      IllegalArgumentException failure =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  model.generate(
                      imageRequest(1).withCachePolicy(GenerationCachePolicy.slidingWindow(16)),
                      ignored -> {}));
      assertEquals("sliding image-prefill retention is not yet supported", failure.getMessage());
      // A declared video token is rejected anywhere in the prompt, image or text only.
      for (int[] prompt : new int[][] {{START, IMAGE, VIDEO, 1, 2}, {1, VIDEO, 2}}) {
        IllegalArgumentException videoFailure =
            assertThrows(
                IllegalArgumentException.class,
                () -> model.generate(textRequest(prompt, 1), ignored -> {}));
        assertTrue(
            videoFailure.getMessage().contains("video is not supported"),
            () -> "got " + videoFailure.getMessage());
      }
      // The fixture context is 256 positions; a longer text prompt is rejected by the plan.
      int[] tooLong = new int[300];
      Arrays.fill(tooLong, 3);
      failure =
          assertThrows(
              IllegalArgumentException.class,
              () -> model.generate(textRequest(tooLong, 1), ignored -> {}));
      assertEquals(
          "expanded prompt exceeds context length: expanded 300, unexpanded 300",
          failure.getMessage());
    }
  }

  @Test
  void capacityIsEnforcedBeforePixels(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomQwen3Vl(dir, 11L);
    try (MLXScope root = new MLXScope()) {
      Qwen3VlModel model = Qwen3VlModel.load(root, dir);
      List<String> executions = new ArrayList<>();
      model.setObserverForTest(executions::add);
      IllegalArgumentException failure =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  model.generate(
                      imageRequest(5).withCachePolicy(GenerationCachePolicy.full(3)),
                      ignored -> {}));
      assertEquals(
          "generation (expanded prompt 8 tokens, unexpanded 5 tokens) exceeds capacity",
          failure.getMessage());
      assertTrue(executions.isEmpty(), "capacity rejected before pixels");
    }
  }

  @Test
  void mlxAffineCheckpointLoadsEndToEnd(@TempDir Path dir) throws Exception {
    Path checkpoint = tinyCheckpoint(dir, 12L);
    Path mlx = copyDirectory(checkpoint, dir.resolve("mlx"));
    int group = 32;
    try (MLXScope root = new MLXScope()) {
      Map<String, MLXArray> loaded =
          MLXIO.loadSafetensors(root, checkpoint.resolve("model.safetensors").toString()).tensors();
      Map<String, MLXArray> tensors = new LinkedHashMap<>();
      for (var entry : loaded.entrySet()) {
        String key = entry.getKey();
        if (key.startsWith("model.language_model.")) {
          key = "language_model.model." + key.substring("model.language_model.".length());
        } else if (key.startsWith("model.visual.")) {
          key = "vision_tower." + key.substring("model.visual.".length());
        }
        tensors.put(key, entry.getValue());
      }
      // The converted artifact's conv arrives channels-last, not PyTorch OIHW.
      tensors.put(
          "vision_tower.patch_embed.proj.weight",
          MLXShape.transpose(
              tensors.remove("vision_tower.patch_embed.proj.weight"), new int[] {0, 2, 3, 4, 1}));
      // The 4-bit affine packing of every projection; the embedding table keeps float storage,
      // so the mixed float/packed layout loads through the same plan.
      for (String key : new ArrayList<>(tensors.keySet())) {
        if (!key.startsWith("language_model.") || !key.endsWith("_proj.weight")) {
          continue;
        }
        String stem = key.substring(0, key.length() - ".weight".length());
        MLXArray[] packed = MLXQuant.quantize(tensors.remove(key), group, 4, "affine", null);
        tensors.put(stem + ".weight", packed[0]);
        tensors.put(stem + ".scales", packed[1]);
        tensors.put(stem + ".biases", packed[2]);
      }
      MLXIO.saveSafetensors(
          mlx.resolve("model.safetensors").toString(), tensors, Map.of("format", "mlx"));
      rewriteJson(
          mlx,
          "config.json",
          config -> {
            ObjectNode quant = config.putObject("quantization");
            quant.put("bits", 4);
            quant.put("group_size", group);
            quant.put("mode", "affine");
          });
      Qwen3VlModel model = Qwen3VlModel.load(root, mlx);
      ArchitectureDescriptor.Quantization quantization =
          model.descriptorForTest().text().quantization();
      assertEquals(group, quantization.groupSize());
      assertEquals(4, quantization.bits());
      // The packed projections ride the packed decoder path as U32; the float embedding and the
      // all-float vision tower load up-cast to FLOAT32.
      assertTrue(
          model.decoderForTest().parameters().values().stream()
              .anyMatch(a -> a.dtype() == DType.UINT32));
      assertTrue(
          model.towerForTest().parameters().values().stream()
              .allMatch(a -> a.dtype() == DType.FLOAT32));
      GenerationResult result = model.generate(textRequest(new int[] {1, 2, 3}, 2), ignored -> {});
      assertEquals(2, result.generatedTokenIds().size());
      assertEquals(FinishReason.MAX_TOKENS, result.finishReason());
      assertEquals(Arrays.stream(new int[] {1, 2, 3}).boxed().toList(), result.promptTokenIds());
    }
  }

  @Test
  void processorConfigCrossChecksNameBothValues(@TempDir Path dir) throws Exception {
    Path checkpoint = tinyCheckpoint(dir, 13L);
    record Mismatch(String key, int value, String expected) {}

    for (Mismatch m :
        new Mismatch[] {
          new Mismatch("patch_size", 8, "patch_size 8 must equal vision_config patch_size 16"),
          new Mismatch(
              "temporal_patch_size",
              4,
              "temporal_patch_size 4 must equal vision_config temporal_patch_size 2"),
          new Mismatch(
              "merge_size", 4, "merge_size 4 must equal vision_config spatial_merge_size 2")
        }) {
      Path bad = copyDirectory(checkpoint, dir.resolve(m.key()));
      rewriteJson(bad, "preprocessor_config.json", config -> config.put(m.key(), m.value()));
      try (MLXScope root = new MLXScope()) {
        assertLoadRejected(root, bad, m.expected());
      }
    }
  }

  @Test
  void tokenizerSentinelMismatchIsRejected(@TempDir Path dir) throws Exception {
    Path checkpoint = tinyCheckpoint(dir, 15L);
    Path bad = copyDirectory(checkpoint, dir.resolve("sentinel"));
    rewriteJson(bad, "config.json", config -> config.put("image_token_id", 9));
    try (MLXScope root = new MLXScope()) {
      assertLoadRejected(
          root, bad, "checkpoint tokenizer does not match the config's prompt sentinels");
    }
  }

  @Test
  void tiedLmHeadAliasMustMatchTheTokenEmbeddings(@TempDir Path dir) throws Exception {
    Path checkpoint = tinyCheckpoint(dir, 16L);
    Path bad = copyDirectory(checkpoint, dir.resolve("lmhead"));
    try (MLXScope root = new MLXScope()) {
      Map<String, MLXArray> tensors =
          new LinkedHashMap<>(
              MLXIO
                  .loadSafetensors(root, checkpoint.resolve("model.safetensors").toString())
                  .tensors());
      // A distinct [vocab, hidden] head while the config ties the head to the embeddings.
      tensors.put("lm_head.weight", MLX.full(root, new int[] {21, 64}, 0.5f, DType.BFLOAT16));
      MLXIO.saveSafetensors(bad.resolve("model.safetensors").toString(), tensors, Map.of());
      assertLoadRejected(root, bad, "tied lm_head alias differs from token embeddings");
    }
  }

  @Test
  void wrongConvShapeIsRejected(@TempDir Path dir) throws Exception {
    Path checkpoint = tinyCheckpoint(dir, 17L);
    Path bad = copyDirectory(checkpoint, dir.resolve("conv"));
    try (MLXScope root = new MLXScope()) {
      Map<String, MLXArray> tensors =
          new LinkedHashMap<>(
              MLXIO
                  .loadSafetensors(root, checkpoint.resolve("model.safetensors").toString())
                  .tensors());
      // OIHW with 8x8 spatial extent instead of 16x16; the tower's transpose is observed in the
      // message (O,T,H,W,I).
      tensors.put(
          "model.visual.patch_embed.proj.weight",
          MLX.full(root, new int[] {64, 3, 2, 8, 8}, 0.5f, DType.BFLOAT16));
      MLXIO.saveSafetensors(bad.resolve("model.safetensors").toString(), tensors, Map.of());
      assertLoadRejected(
          root, bad, "checkpoint shape [64, 2, 8, 8, 3] differs from config [64, 2, 16, 16, 3]");
    }
  }
}
