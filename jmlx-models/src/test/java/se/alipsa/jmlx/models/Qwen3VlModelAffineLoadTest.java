package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
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
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.core.MLXQuant;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.vision.ImageDecoder;
import se.alipsa.jmlx.vision.RgbImage;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ObjectNode;

/**
 * The MLX-affine-quantized layout on the tiny WP6 Qwen3-VL checkpoint, packed in Java: every
 * eligible text weight (the projections, the embedding table and an explicit untied head) is
 * quantized 4-bit group-64 affine, the conv arrives channels-last, and the packed decoder path is
 * compared against the identical weights dequantized to float (the sidecars up-cast to FLOAT32
 * first, mirroring the loader, so the reconstruction is bit-identical to what the packed kernel
 * itself dequantizes) and reloaded through the same affine plan -- the comparison isolates the
 * packed kernel from the quantization error itself, since both models consume the same weights. The
 * rejection matrix covers the layout's provenance, naming and shape contracts.
 */
@EnabledIfNativeAvailable
class Qwen3VlModelAffineLoadTest {

  private static final int GROUP = 64;
  private static final int BITS = 4;

  /**
   * The elementwise bound for the packed-versus-dequantized comparison, the same 2e-3 as {@link
   * BatchStepEquivalenceTest}'s TF32-mode tolerance. Both models consume bit-identical weights (the
   * reference dequantizes from the sidecars up-cast to FLOAT32, as the packed kernel does), so the
   * difference is kernel noise only: measured at or below 5.7e-7 with MLX_ENABLE_TF32=0 (matmul
   * accumulation order) and at or below 1.09e-3 under this task's forced MLX_ENABLE_TF32=1, where
   * the reference's plain GEMMs run at TF32 precision. A wiring error (wrong weight, wrong layout,
   * missing layer) would differ by orders of magnitude more.
   */
  private static final float TOLERANCE = 2e-3f;

  static Path source() {
    return Path.of(
        System.getProperty("jmlx.repository.root"),
        "tools/hf-reference/goldens/checkpoints/qwen3vl-model");
  }

  static JsonNode golden() throws Exception {
    return JsonFiles.read(source().getParent().getParent().resolve("qwen3vl-model-wp6.json"));
  }

  /**
   * The committed checkpoint's tensors rekeyed to the affine layout: {@code model.language_model.*}
   * to {@code language_model.model.*}, {@code model.visual.*} to {@code vision_tower.*}, and the
   * patch-embed conv transposed to the converted artifact's channels-last storage (the committed HF
   * float checkpoint holds PyTorch OIHW).
   */
  static Map<String, MLXArray> affineTensors(MLXScope scope) {
    Map<String, MLXArray> loaded =
        MLXIO.loadSafetensors(scope, source().resolve("model.safetensors").toString()).tensors();
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
    tensors.put(
        "vision_tower.patch_embed.proj.weight",
        MLXShape.transpose(
            tensors.remove("vision_tower.patch_embed.proj.weight"), new int[] {0, 2, 3, 4, 1}));
    return tensors;
  }

  /** Every affine-layout text weight that may be packed: the projections and the embedding. */
  private static boolean packable(String key) {
    return key.startsWith("language_model.")
        && (key.endsWith("_proj.weight") || key.equals("language_model.model.embed_tokens.weight"));
  }

  private static MLXArray[] packed(MLXArray weight) {
    return MLXQuant.quantize(weight, GROUP, BITS, "affine", null);
  }

  /**
   * The float reconstruction of one packed triple: the sidecars are up-cast to FLOAT32 first,
   * exactly as the loader's own up-cast does before the packed kernel consumes them. Dequantization
   * runs in the sidecar dtype, and the raw packing's sidecars are BFLOAT16 (they follow the packed
   * weight's storage dtype), so reconstructing from the up-cast sidecars is bit-identical to the
   * weights the packed model itself dequantizes.
   */
  private static MLXArray dequantized(MLXArray[] packed) {
    return MLXQuant.dequantize(
        packed[0],
        MLX.astype(packed[1], DType.FLOAT32),
        MLX.astype(packed[2], DType.FLOAT32),
        GROUP,
        BITS,
        "affine",
        null,
        DType.FLOAT32);
  }

  /**
   * The affine store for one derived checkpoint: every packable text weight quantized (or, when
   * {@code dequantized}, replaced by the float reconstruction of the very same packing with no
   * sidecars), the norms kept float, the channels-last vision subtree untouched, and the explicit
   * untied head stored as a distinct copy of the embedding, packed like every other eligible
   * weight.
   */
  static Map<String, MLXArray> packedStore(Map<String, MLXArray> tensors, boolean dequantized) {
    Map<String, MLXArray> store = new LinkedHashMap<>();
    for (var entry : tensors.entrySet()) {
      String key = entry.getKey();
      if (packable(key)) {
        String stem = key.substring(0, key.length() - ".weight".length());
        MLXArray[] packed = packed(entry.getValue());
        if (dequantized) {
          store.put(key, dequantized(packed));
        } else {
          store.put(key, packed[0]);
          store.put(stem + ".scales", packed[1]);
          store.put(stem + ".biases", packed[2]);
        }
      } else {
        store.put(key, entry.getValue());
      }
    }
    MLXArray[] head =
        packed(MLX.astype(tensors.get("language_model.model.embed_tokens.weight"), DType.BFLOAT16));
    if (dequantized) {
      store.put("language_model.lm_head.weight", dequantized(head));
    } else {
      store.put("language_model.lm_head.weight", head[0]);
      store.put("language_model.lm_head.scales", head[1]);
      store.put("language_model.lm_head.biases", head[2]);
    }
    return store;
  }

  private static ObjectNode quantBlock(ObjectMapper mapper, int groupSize) {
    ObjectNode block = mapper.createObjectNode();
    block.put("bits", BITS);
    block.put("group_size", groupSize);
    block.put("mode", "affine");
    return block;
  }

  /** The derived checkpoint's config: the untied head plus both JSON-equal quantization blocks. */
  static Consumer<ObjectNode> affineConfig(int groupSize) {
    ObjectMapper mapper = new ObjectMapper();
    ObjectNode block = quantBlock(mapper, groupSize);
    return config -> {
      config.put("tie_word_embeddings", false);
      config.set("quantization", block);
      config.set("quantization_config", block.deepCopy());
    };
  }

  /**
   * Writes one derived checkpoint directory: the committed support files, the given store under a
   * single shard with the given metadata, and the config with the given change (null keeps the
   * committed config).
   */
  static Path writeCheckpoint(
      Path directory,
      Map<String, MLXArray> store,
      Map<String, String> metadata,
      Consumer<ObjectNode> config)
      throws IOException {
    Files.createDirectories(directory.resolve("images"));
    for (String file : new String[] {"config.json", "preprocessor_config.json", "tokenizer.json"}) {
      Files.copy(source().resolve(file), directory.resolve(file));
    }
    for (Path image : Files.list(source().resolve("images")).toList()) {
      Files.copy(image, directory.resolve("images").resolve(image.getFileName()));
    }
    MLXIO.saveSafetensors(directory.resolve("model.safetensors").toString(), store, metadata);
    if (config != null) {
      ObjectMapper mapper = new ObjectMapper();
      ObjectNode tree = (ObjectNode) mapper.readTree(directory.resolve("config.json").toFile());
      config.accept(tree);
      Files.writeString(directory.resolve("config.json"), mapper.writeValueAsString(tree));
    }
    return directory;
  }

  private static int[] ints(JsonNode node) {
    List<Integer> values = new ArrayList<>();
    node.forEach(v -> values.add(v.intValue()));
    return values.stream().mapToInt(Integer::intValue).toArray();
  }

  private static List<RgbImage> images(JsonNode c) throws Exception {
    List<RgbImage> images = new ArrayList<>();
    for (JsonNode image : c.required("images")) {
      images.add(ImageDecoder.decode(source().resolve("images").resolve(image.asString())));
    }
    return images;
  }

  private static float maxDiff(float[] a, float[] b) {
    float max = 0f;
    for (int i = 0; i < a.length; i++) {
      max = Math.max(max, Math.abs(a[i] - b[i]));
    }
    return max;
  }

  private static void assertLoadRejected(MLXScope scope, Path directory, String expected) {
    IllegalArgumentException failure =
        assertThrows(IllegalArgumentException.class, () -> Qwen3VlModel.load(scope, directory));
    assertTrue(
        failure.getMessage().contains(expected),
        () -> "expected message containing " + expected + ", got " + failure.getMessage());
  }

  @Test
  void packedAffineMatchesTheDequantizedReference(@TempDir Path dir) throws Exception {
    try (MLXScope root = new MLXScope()) {
      Qwen3VlModel packed =
          Qwen3VlModel.load(
              root,
              writeCheckpoint(
                  dir.resolve("quantized"),
                  packedStore(affineTensors(root), false),
                  Map.of("format", "mlx"),
                  affineConfig(GROUP)));
      ArchitectureDescriptor.Quantization quantization =
          packed.descriptorForTest().text().quantization();
      assertEquals(GROUP, quantization.groupSize());
      assertEquals(BITS, quantization.bits());
      // The packed model consumes the packed decoder path (U32 weights, float32 sidecars).
      assertTrue(
          packed.decoderForTest().parameters().values().stream()
              .anyMatch(a -> a.dtype() == DType.UINT32));
      // The reference runs the identical affine plan on float storage, so every parameter is
      // FLOAT32 (the all-float warning is its only cost, not a rejection).
      Qwen3VlModel dequantized =
          Qwen3VlModel.load(
              root,
              writeCheckpoint(
                  dir.resolve("reference"),
                  packedStore(affineTensors(root), true),
                  Map.of("format", "mlx"),
                  affineConfig(GROUP)));
      assertTrue(
          dequantized.decoderForTest().parameters().values().stream()
              .allMatch(a -> a.dtype() == DType.FLOAT32));
      assertTrue(
          dequantized.towerForTest().parameters().values().stream()
              .allMatch(a -> a.dtype() == DType.FLOAT32));
      float max = 0f;
      for (JsonNode c : golden().required("cases")) {
        int[] ids = ints(c.required("unexpanded_ids"));
        List<RgbImage> images = images(c);
        float[] packedLogits;
        try (MLXScope generation = root.newChild()) {
          packedLogits =
              c.has("token_id")
                  ? packed.decodeStepLogitsForTest(
                      generation, ids, images, c.required("token_id").asInt())
                  : packed.prefillLogitsForTest(generation, ids, images);
        }
        float[] referenceLogits;
        try (MLXScope generation = root.newChild()) {
          referenceLogits =
              c.has("token_id")
                  ? dequantized.decodeStepLogitsForTest(
                      generation, ids, images, c.required("token_id").asInt())
                  : dequantized.prefillLogitsForTest(generation, ids, images);
        }
        String name = c.required("name").asString();
        max = Math.max(max, maxDiff(packedLogits, referenceLogits));
        assertArrayEquals(
            referenceLogits, packedLogits, TOLERANCE, name + " (measured max diff " + max + ")");
      }
    }
  }

  @Test
  void missingQuantizationSidecarIsRejected(@TempDir Path dir) throws Exception {
    try (MLXScope root = new MLXScope()) {
      Map<String, MLXArray> store = packedStore(affineTensors(root), false);
      store.remove("language_model.model.layers.0.self_attn.q_proj.scales");
      Path bad =
          writeCheckpoint(
              dir.resolve("sidecar"), store, Map.of("format", "mlx"), affineConfig(GROUP));
      assertLoadRejected(
          root, bad, "checkpoint missing tensor 'model.layers.0.self_attn.q_proj.scales'");
    }
  }

  @Test
  void groupSizeMismatchIsRejected(@TempDir Path dir) throws Exception {
    try (MLXScope root = new MLXScope()) {
      // One packed tensor (group 64) among otherwise float storage, against a declared group of
      // 128: its sidecar's 256 / 64 = 4 columns are the only shape the loader can mismatch with
      // the declared 256 / 128 = 2, so the rejection is deterministic no matter which tensor the
      // loader visits first (the safetensors map order is not the writing order).
      Map<String, MLXArray> tensors = affineTensors(root);
      String stem = "language_model.model.layers.0.self_attn.q_proj";
      Map<String, MLXArray> store = packedStore(tensors, true);
      MLXArray[] packed = packed(tensors.get(stem + ".weight"));
      store.put(stem + ".weight", packed[0]);
      store.put(stem + ".scales", packed[1]);
      store.put(stem + ".biases", packed[2]);
      Path bad =
          writeCheckpoint(dir.resolve("group"), store, Map.of("format", "mlx"), affineConfig(128));
      assertLoadRejected(root, bad, "checkpoint shape [256, 4] differs from config [256, 2]");
    }
  }

  @Test
  void normQuantizationSidecarIsRejected(@TempDir Path dir) throws Exception {
    try (MLXScope root = new MLXScope()) {
      // The norms are float-only in the affine layout; the plan bars their sidecars by name.
      Map<String, MLXArray> store = packedStore(affineTensors(root), false);
      store.put(
          "language_model.model.layers.0.self_attn.q_norm.scales",
          MLX.full(root, new int[] {64}, 0.5f, DType.BFLOAT16));
      Path bad =
          writeCheckpoint(dir.resolve("norm"), store, Map.of("format", "mlx"), affineConfig(GROUP));
      assertLoadRejected(
          root, bad, "unexpected tensor 'language_model.model.layers.0.self_attn.q_norm.scales'");
    }
  }

  @Test
  void pytorchConvLayoutIsRejectedUnderMlxPlan(@TempDir Path dir) throws Exception {
    try (MLXScope root = new MLXScope()) {
      // The converted artifact's conv arrives channels-last; the PyTorch OIHW storage the HF
      // layout carries does not match the tower's expected shape.
      Map<String, MLXArray> tensors = affineTensors(root);
      tensors.put(
          "vision_tower.patch_embed.proj.weight",
          MLXShape.transpose(
              tensors.remove("vision_tower.patch_embed.proj.weight"), new int[] {0, 4, 1, 2, 3}));
      Path bad =
          writeCheckpoint(
              dir.resolve("conv"),
              packedStore(tensors, false),
              Map.of("format", "mlx"),
              affineConfig(GROUP));
      assertLoadRejected(
          root,
          bad,
          "checkpoint shape [256, 3, 2, 16, 16] differs from config [256, 2, 16, 16, 3]");
    }
  }

  @Test
  void hfTextKeysAreRejectedUnderMlxPlan(@TempDir Path dir) throws Exception {
    try (MLXScope root = new MLXScope()) {
      // The HF float layout's text names under the affine plan: none of the layout's required
      // names is present, so the plan rejects the whole name set before any dtype work.
      Map<String, MLXArray> store = packedStore(affineTensors(root), false);
      Map<String, MLXArray> rekeyed = new LinkedHashMap<>();
      for (var entry : store.entrySet()) {
        String key = entry.getKey();
        if (key.startsWith("language_model.model.")) {
          key = "model.language_model." + key.substring("language_model.model.".length());
        } else if (key.startsWith("language_model.lm_head.")) {
          key = "lm_head." + key.substring("language_model.lm_head.".length());
        }
        rekeyed.put(key, entry.getValue());
      }
      Path bad =
          writeCheckpoint(
              dir.resolve("hf-keys"), rekeyed, Map.of("format", "mlx"), affineConfig(GROUP));
      assertLoadRejected(root, bad, "missing tensor 'language_model.model.embed_tokens.weight'");
    }
  }

  @Test
  void mlxFormatWithoutQuantizationIsRejected(@TempDir Path dir) throws Exception {
    try (MLXScope root = new MLXScope()) {
      // The shard metadata alone cannot name the layout: the config declares no quantization
      // block at all, so the provenance gate rejects before any tensor is read.
      Path bad =
          writeCheckpoint(
              dir.resolve("no-quant"),
              packedStore(affineTensors(root), false),
              Map.of("format", "mlx"),
              null);
      assertLoadRejected(
          root,
          bad,
          "shard metadata format=mlx requires a config quantization block of exactly"
              + " {bits: 4, group_size: 32, 64 or 128, mode: \"affine\"}, but none is declared");
    }
  }

  @Test
  void packedTensorsAreRejectedUnderHfPlan(@TempDir Path dir) throws Exception {
    try (MLXScope root = new MLXScope()) {
      // The packed affine store without any format metadata or quantization declaration: the
      // loader resolves the HF float plan, whose names the affine layout does not belong to --
      // the plan check fires before the U32 dtype could be rejected.
      Path bad =
          writeCheckpoint(
              dir.resolve("hf-plan"), packedStore(affineTensors(root), false), Map.of(), null);
      assertLoadRejected(root, bad, "unexpected tensor 'language_model.lm_head.scales'");
    }
  }
}
