package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXQuant;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;
import tools.jackson.databind.JsonNode;

/**
 * Drives one family/variant through every capture the recorder defines, on the calling thread (the
 * scheduler capture runs on the scheduler's own worker thread by design). One fresh KV cache set
 * per capture, so no capture sees another's advanced positions.
 */
final class ExactBitsCapture {

  /** One generated row of the scheduler capture: full token IDs plus the recorded finish reason. */
  record SchedulerRow(List<Integer> tokenIds, String finishReason) {}

  /** Everything captured (or, when loading failed, the not-applicable reason) for one variant. */
  static final class Variant {
    final String name;
    final String family;
    boolean applicable = true;
    String reason = "";
    final Map<String, String> inputHashes = new TreeMap<>();
    final Map<String, String> derivedHashes = new TreeMap<>();
    final Map<String, float[]> floatCaptures = new LinkedHashMap<>();
    List<Integer> greedyIds;
    List<SchedulerRow> schedulerRows;

    Variant(String name, String family) {
      this.name = name;
      this.family = family;
    }
  }

  private ExactBitsCapture() {}

  /**
   * Captures every schedule entry for one variant. {@code checkpoint} is the directory the variant
   * loads from (the committed float checkpoint, or a deterministically derived quantized one). A
   * load failure marks the variant not applicable with the error message instead of failing the
   * whole run.
   */
  static Variant capture(
      String name, String family, Path checkpoint, Path goldenFile, JsonNode golden)
      throws Exception {
    Variant variant = new Variant(name, family);
    variant.inputHashes.putAll(sha256Files(checkpoint));
    variant.inputHashes.put(family + ".json", ExactBitsSpec.sha256(Files.readString(goldenFile)));
    int[] prompt = tokenIds(golden.path("prompt_ids"));
    List<Integer> decodeTokens = decodeTokens(golden.path("decode_steps"));
    try (MLXScope modelScope = new MLXScope()) {
      DecoderModel model;
      try {
        model = (DecoderModel) TextGenerationModels.load(modelScope, checkpoint);
      } catch (Exception e) {
        variant.applicable = false;
        variant.reason = e.getClass().getSimpleName() + ": " + e.getMessage();
        return variant;
      }
      int layers = model.config().numHiddenLayers();
      directCaptures(variant, model, modelScope, layers, prompt, decodeTokens);
      List<Integer> generated = model.generate(prompt, ExactBitsSpec.MAX_NEW_TOKENS, Set.of());
      variant.greedyIds = List.copyOf(generated.subList(prompt.length, generated.size()));
      batchCaptures(variant, model, modelScope, layers, prompt, decodeTokens);
      variant.schedulerRows = schedulerCapture(checkpoint, prompt);
    }
    return variant;
  }

  /**
   * Deterministically derives the quantized checkpoint of a committed float checkpoint with the
   * same group, bits and key set as QuantizedDecoderTest.quantize, writing it to {@code outDir} and
   * recording the SHA-256 of the derived files in {@code derivedHashes}. Refuses with {@link
   * IllegalStateException} a checkpoint whose config.json already carries a top-level {@code
   * quantization} key, since the derivation would splice a duplicate key into it.
   */
  static Path deriveQuantized(Path floatCheckpoint, Path outDir, Map<String, String> derivedHashes)
      throws Exception {
    JsonNode config = JsonFiles.read(floatCheckpoint.resolve("config.json"));
    if (config.isObject() && config.has("quantization")) {
      throw new IllegalStateException(
          "cannot derive a quantized checkpoint from "
              + floatCheckpoint
              + ": its config.json already declares a top-level \"quantization\" key that the "
              + "derivation would duplicate");
    }
    Path out = Files.createDirectories(outDir);
    String configText = Files.readString(floatCheckpoint.resolve("config.json"));
    Files.writeString(
        out.resolve("config.json"),
        configText.substring(0, configText.lastIndexOf('}'))
            + ",\"quantization\":{\"group_size\":"
            + ExactBitsSpec.QUANT_GROUP
            + ",\"bits\":"
            + ExactBitsSpec.QUANT_BITS
            + "}}");
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> floats =
          MLXIO
              .loadSafetensors(scope, floatCheckpoint.resolve("model.safetensors").toString())
              .tensors();
      Map<String, MLXArray> packed = new LinkedHashMap<>();
      for (Map.Entry<String, MLXArray> entry : floats.entrySet()) {
        String key = entry.getKey();
        boolean quantize =
            key.endsWith("_proj.weight")
                || key.equals("model.embed_tokens.weight")
                || key.equals("lm_head.weight");
        if (!quantize) {
          packed.put(key, entry.getValue());
          continue;
        }
        String stem = key.substring(0, key.length() - ".weight".length());
        MLXArray[] quantized =
            MLXQuant.quantize(
                entry.getValue(),
                ExactBitsSpec.QUANT_GROUP,
                ExactBitsSpec.QUANT_BITS,
                "affine",
                null);
        packed.put(key, quantized[0]);
        packed.put(stem + ".scales", quantized[1]);
        packed.put(stem + ".biases", quantized[2]);
      }
      MLXIO.saveSafetensors(out.resolve("model.safetensors").toString(), packed, Map.of());
    }
    derivedHashes.putAll(sha256Files(out));
    return out;
  }

  /** Prefill logits, then one cached-step forward per golden decode step, single row. */
  private static void directCaptures(
      Variant variant,
      DecoderModel model,
      MLXScope modelScope,
      int layers,
      int[] prompt,
      List<Integer> decodeTokens) {
    try (MLXScope inference = modelScope.newChild()) {
      List<KVCache> caches = newCaches(inference, layers);
      try (MLXScope activation = inference.newChild()) {
        variant.floatCaptures.put(
            "direct-prefill",
            model
                .forward(MLX.array(activation, prompt, new int[] {1, prompt.length}), caches)
                .toFloatArray());
      }
      for (int i = 0; i < decodeTokens.size(); i++) {
        int token = decodeTokens.get(i);
        try (MLXScope activation = inference.newChild()) {
          variant.floatCaptures.put(
              "decode-step-" + i,
              model
                  .forward(MLX.array(activation, new int[] {token}, new int[] {1, 1}), caches)
                  .toFloatArray());
        }
      }
    }
  }

  /**
   * Batched left-padded prefill and two 1-column batched decode steps on one cache set, then the
   * same prefill with the row-B embedding hook on a fresh cache set. In both record and verify
   * modes, asserts that the hook capture stays bit-identical to {@code batch-prefill} on row 0 and
   * differs on the hooked row, so a refactor that drops the hook fails loudly at record/verify time
   * instead of silently verifying against a numerically-dead recording.
   */
  private static void batchCaptures(
      Variant variant,
      DecoderModel model,
      MLXScope modelScope,
      int layers,
      int[] prompt,
      List<Integer> decodeTokens) {
    int[] rowB = ExactBitsSpec.ROW_B;
    int width = Math.max(prompt.length, rowB.length);
    int[] valid = {prompt.length, rowB.length};
    int[] padded = new int[2 * width];
    int[][] rows = {prompt, rowB};
    for (int row = 0; row < 2; row++) {
      int[] ids = rows[row];
      Arrays.fill(padded, row * width, row * width + (width - ids.length), ExactBitsSpec.PAD_FILL);
      System.arraycopy(ids, 0, padded, row * width + (width - ids.length), ids.length);
    }
    try (MLXScope inference = modelScope.newChild()) {
      List<KVCache> caches = newCaches(inference, layers);
      variant.floatCaptures.put(
          "batch-prefill", step(model, inference, caches, padded, valid, null));
      for (int stepIndex = 0; stepIndex < 2; stepIndex++) {
        int rowA =
            stepIndex < decodeTokens.size()
                ? decodeTokens.get(stepIndex)
                : ExactBitsSpec.BATCH_DECODE_TOKENS_B[2];
        int[] decode = {rowA, ExactBitsSpec.BATCH_DECODE_TOKENS_B[stepIndex]};
        variant.floatCaptures.put(
            "batch-decode-" + stepIndex,
            step(model, inference, caches, decode, new int[] {1, 1}, null));
      }
    }
    try (MLXScope inference = modelScope.newChild()) {
      List<KVCache> caches = newCaches(inference, layers);
      variant.floatCaptures.put(
          "batch-prefill-hook",
          step(model, inference, caches, padded, valid, ExactBitsCapture::hook));
    }
    assertHookRowActive(variant);
  }

  /**
   * Asserts the embedding hook actually changed bits in the captured step: {@code
   * batch-prefill-hook} must be bit-for-bit identical to {@code batch-prefill} on row 0 and must
   * differ on row {@link ExactBitsSpec#HOOK_ROW} in at least one element. Runs in both record and
   * verify mode, since it is a property of the candidate run itself, so a refactor that silently
   * drops the hook fails loudly at record/verify time instead of matching a recording made with a
   * numerically-dead hook.
   */
  private static void assertHookRowActive(Variant variant) {
    float[] prefill = variant.floatCaptures.get("batch-prefill");
    float[] hooked = variant.floatCaptures.get("batch-prefill-hook");
    int rowStride = prefill.length / 2;
    for (int i = 0; i < rowStride; i++) {
      int plainBits = Float.floatToIntBits(prefill[i]);
      int hookBits = Float.floatToIntBits(hooked[i]);
      if (plainBits != hookBits) {
        throw new AssertionError(
            variant.name
                + ": batch-prefill row 0 diverges from batch-prefill-hook at index "
                + i
                + ": plain bits 0x"
                + Integer.toHexString(plainBits)
                + ", hook bits 0x"
                + Integer.toHexString(hookBits));
      }
    }
    for (int i = rowStride; i < prefill.length; i++) {
      if (Float.floatToIntBits(prefill[i]) != Float.floatToIntBits(hooked[i])) {
        return;
      }
    }
    throw new AssertionError(
        variant.name
            + ": batch-prefill-hook row "
            + ExactBitsSpec.HOOK_ROW
            + " is bit-identical to batch-prefill: the embedding hook is numerically dead");
  }

  /**
   * End-to-end scheduler: two greedy requests of different lengths sharing one gated cohort;
   * asserts they actually formed one 2-row cohort before the scheduler is closed, so a split into
   * two 1-row cohorts (e.g. a GC pause past the gate window) fails the capture instead of silently
   * recording an unbatched run.
   */
  private static List<SchedulerRow> schedulerCapture(Path checkpoint, int[] prompt)
      throws Exception {
    BatchSchedulerConfig config =
        new BatchSchedulerConfig(
            ExactBitsSpec.SCHEDULER_BATCH,
            ExactBitsSpec.SCHEDULER_QUEUED,
            ExactBitsSpec.SCHEDULER_PROMPT_BUDGET,
            ExactBitsSpec.SCHEDULER_MAX_NEW);
    BatchGenerationScheduler.Hooks hooks =
        BatchGenerationScheduler.Hooks.NONE.withCohortGate(
            waiting -> waiting >= ExactBitsSpec.SCHEDULER_GATE_N,
            Duration.ofMillis(ExactBitsSpec.SCHEDULER_GATE_WAIT_MS));
    try (BatchGenerationScheduler scheduler =
        BatchGenerationScheduler.start(
            config, root -> TextGenerationModels.load(root, checkpoint), hooks)) {
      BatchRequestHandle rowA =
          scheduler.submit(SchedulerFixtures.greedy(prompt, ExactBitsSpec.MAX_NEW_TOKENS), e -> {});
      BatchRequestHandle rowB =
          scheduler.submit(
              SchedulerFixtures.greedy(ExactBitsSpec.ROW_B, ExactBitsSpec.MAX_NEW_TOKENS), e -> {});
      GenerationResult resultA = SchedulerFixtures.await(rowA);
      GenerationResult resultB = SchedulerFixtures.await(rowB);
      if (!scheduler.cohortSizes().equals(List.of(2))) {
        throw new AssertionError(
            "scheduler capture did not form one 2-row cohort: " + scheduler.cohortSizes());
      }
      return List.of(
          new SchedulerRow(resultA.tokenIds(), resultA.finishReason().name()),
          new SchedulerRow(resultB.tokenIds(), resultB.finishReason().name()));
    }
  }

  /** Runs one lazy batched step, evaluates through the model's boundary, returns raw bits. */
  private static float[] step(
      DecoderModel model,
      MLXScope run,
      List<KVCache> caches,
      int[] flatIds,
      int[] valid,
      DecoderModel.EmbeddingHook hook) {
    int batch = valid.length;
    int width = flatIds.length / batch;
    try (MLXScope activation = run.newChild()) {
      MLXArray ids = MLX.array(activation, flatIds, new int[] {batch, width});
      MLXArray logits = model.stepLogits(ids, caches, valid, hook);
      MLXArray[] arrays = new MLXArray[1 + 2 * caches.size()];
      arrays[0] = logits;
      System.arraycopy(DecoderModel.stepCacheArrays(caches), 0, arrays, 1, 2 * caches.size());
      model.stepBoundaryEvaluator().evaluate(arrays);
      return logits.toFloatArray();
    }
  }

  /** The recorded hook: adds a fixed tiny constant to every element of one row's embeddings. */
  private static MLXArray hook(MLXArray embedded) {
    int[] shape = embedded.shape();
    float[] addend = new float[shape[0]];
    addend[ExactBitsSpec.HOOK_ROW] = ExactBitsSpec.HOOK_CONSTANT;
    return MLXOps.add(embedded, MLX.array(embedded.scope(), addend, new int[] {shape[0], 1, 1}));
  }

  private static List<KVCache> newCaches(MLXScope scope, int layers) {
    List<KVCache> caches = new ArrayList<>();
    for (int i = 0; i < layers; i++) {
      caches.add(new KVCache(scope));
    }
    return caches;
  }

  static List<Integer> decodeTokens(JsonNode steps) {
    List<Integer> out = new ArrayList<>();
    for (JsonNode step : steps) {
      out.add(step.path("token_id").asInt());
    }
    return out;
  }

  static int[] tokenIds(JsonNode node) {
    int[] out = new int[node.size()];
    for (int i = 0; i < out.length; i++) {
      out[i] = node.get(i).asInt();
    }
    return out;
  }

  /** SHA-256 of every regular file in the directory, keyed by file name. */
  static Map<String, String> sha256Files(Path directory) throws IOException {
    Map<String, String> out = new TreeMap<>();
    try (var paths = Files.list(directory)) {
      for (Path file : paths.sorted().toList()) {
        if (Files.isRegularFile(file)) {
          out.put(file.getFileName().toString(), ExactBitsSpec.sha256(Files.readAllBytes(file)));
        }
      }
    }
    return out;
  }
}
