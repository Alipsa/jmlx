package se.alipsa.jmlx.examples;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXMemory;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.models.CancellationToken;
import se.alipsa.jmlx.models.DecoderModel;
import se.alipsa.jmlx.models.GenerationCachePolicy;
import se.alipsa.jmlx.models.GenerationConfig;
import se.alipsa.jmlx.models.GenerationRequest;
import se.alipsa.jmlx.models.GenerationResult;
import se.alipsa.jmlx.models.TextGenerationModel;
import se.alipsa.jmlx.models.TextGenerationModels;
import se.alipsa.jmlx.nn.KVCache;
import se.alipsa.jmlx.nn.KVCachePolicy;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/** Opt-in local-checkpoint decode and native-memory benchmark. */
public final class DecodeBenchmark {
  private DecodeBenchmark() {}

  /**
   * Runs with {@code checkpoint-dir output-prefix [prompt-csv] [tokens] [samples] [warmups]
   * [policy]}, or, when the first argument is {@code seam-comparison}, with {@code seam-comparison
   * qwen3vl-checkpoint-dir output-prefix [prompt-csv] [tokens] [samples] [warmups]} (the WP4 seam
   * one-off; the cache policy is fixed to full).
   *
   * @param args local checkpoint, output prefix, and optional benchmark parameters
   * @throws Exception if loading, native evaluation, hashing, or report writing fails
   */
  public static void main(String[] args) throws Exception {
    if (args.length > 0 && "seam-comparison".equals(args[0])) {
      seamComparison(args);
      return;
    }
    if (args.length < 2 || args.length > 7) {
      throw new IllegalArgumentException(
          "usage: DecodeBenchmark checkpoint-dir output-prefix [prompt-csv] [tokens] [samples]"
              + " [warmups] [full|sliding-from-model]");
    }
    Path checkpoint = Path.of(args[0]);
    Path output = Path.of(args[1]);
    int[] prompt = args.length > 2 ? parsePrompt(args[2]) : new int[] {1, 2};
    int tokens = args.length > 3 ? Integer.parseInt(args[3]) : 8;
    int samples = args.length > 4 ? Integer.parseInt(args[4]) : 3;
    int warmups = args.length > 5 ? Integer.parseInt(args[5]) : 1;
    String policyName = args.length > 6 ? args[6] : "full";
    GenerationCachePolicy policy =
        switch (policyName) {
          case "full" -> GenerationCachePolicy.full();
          case "sliding-from-model" -> GenerationCachePolicy.slidingWindowFromModel();
          default ->
              throw new IllegalArgumentException(
                  "unknown cache policy '" + policyName + "'; expected full|sliding-from-model");
        };
    if (tokens < 1 || samples < 1 || warmups < 0) {
      throw new IllegalArgumentException("tokens/samples must be positive and warmups nonnegative");
    }
    long beforeLoad = MLXMemory.activeBytes();
    MLXMemory.resetPeak();
    long loadStart = System.nanoTime();
    try (MLXScope modelScope = new MLXScope()) {
      DecoderModel model = (DecoderModel) TextGenerationModels.load(modelScope, checkpoint);
      final long coldLoadNanos = System.nanoTime() - loadStart;
      final long afterLoad = MLXMemory.activeBytes();
      final long coldPeak = MLXMemory.peakBytes();
      // Hash only after the cold-load timer: hashing beforehand would warm the OS page cache.
      final Map<String, String> checkpointHashes = checkpointHashes(checkpoint);
      JsonNode config = new ObjectMapper().readTree(checkpoint.resolve("config.json").toFile());
      KVCachePolicy resolvedPolicy = model.resolveCachePolicy(policy);
      List<Long> prefillNanos = new ArrayList<>();
      List<Long> decodeNanos = new ArrayList<>();
      List<Long> generationNanos = new ArrayList<>();
      List<Long> prefillPeak = new ArrayList<>();
      List<Long> decodePeak = new ArrayList<>();
      List<Long> activeBeforePrefill = new ArrayList<>();
      List<Long> activeAfterPrefill = new ArrayList<>();
      List<Long> activeBeforeDecode = new ArrayList<>();
      List<Long> activeAfterDecode = new ArrayList<>();
      List<Long> activeBeforeGeneration = new ArrayList<>();
      List<Long> activeAfterGeneration = new ArrayList<>();
      List<Long> peakGeneration = new ArrayList<>();
      List<List<Long>> activePerGeneratedToken = new ArrayList<>();
      for (int run = -warmups; run < samples; run++) {
        try (MLXScope inference = modelScope.newChild()) {
          List<KVCache> caches = new ArrayList<>();
          for (int layer = 0; layer < model.config().numHiddenLayers(); layer++) {
            caches.add(new KVCache(inference, resolvedPolicy));
          }
          try (MLXScope step = inference.newChild()) {
            MLXArray ids = MLX.array(step, prompt, new int[] {1, prompt.length});
            long before = MLXMemory.activeBytes();
            MLXMemory.resetPeak();
            long start = System.nanoTime();
            model.forward(ids, caches);
            long elapsed = System.nanoTime() - start;
            if (run >= 0) {
              prefillNanos.add(elapsed);
              activeBeforePrefill.add(before);
              activeAfterPrefill.add(MLXMemory.activeBytes());
              prefillPeak.add(MLXMemory.peakBytes());
            }
          }
          try (MLXScope step = inference.newChild()) {
            MLXArray ids = MLX.array(step, new int[] {prompt[prompt.length - 1]}, new int[] {1, 1});
            long before = MLXMemory.activeBytes();
            MLXMemory.resetPeak();
            long start = System.nanoTime();
            model.forward(ids, caches);
            long elapsed = System.nanoTime() - start;
            if (run >= 0) {
              decodeNanos.add(elapsed);
              activeBeforeDecode.add(before);
              activeAfterDecode.add(MLXMemory.activeBytes());
              decodePeak.add(MLXMemory.peakBytes());
            }
          }
        }
        GenerationRequest request =
            new GenerationRequest(
                    prompt,
                    GenerationConfig.greedyDefaults(tokens, Set.of()),
                    CancellationToken.NONE)
                .withCachePolicy(policy);
        MLXMemory.resetPeak();
        long before = MLXMemory.activeBytes();
        long start = System.nanoTime();
        model.generate(request, ignored -> {});
        long elapsed = System.nanoTime() - start;
        if (run >= 0) {
          generationNanos.add(elapsed);
          activeBeforeGeneration.add(before);
          activeAfterGeneration.add(MLXMemory.activeBytes());
          peakGeneration.add(MLXMemory.peakBytes());
          // Sample a separate repeat so memory-counter calls do not alter timed throughput.
          List<Long> intervalSamples = new ArrayList<>();
          model.generate(
              request,
              event -> {
                if (event.tokenId() != null) {
                  intervalSamples.add(MLXMemory.activeBytes());
                }
              });
          activePerGeneratedToken.add(intervalSamples);
        }
      }
      Map<String, Object> report = new LinkedHashMap<>();
      report.put("schema", "jmlx-decode-benchmark-1");
      report.put("checkpoint", reportedCheckpoint(checkpoint));
      report.put("checkpoint_sha256", checkpointHashes);
      report.put("commit", commit());
      report.put("git_status_porcelain", gitStatus());
      report.put("java", System.getProperty("java.version"));
      report.put("os", System.getProperty("os.name"));
      report.put("architecture", System.getProperty("os.arch"));
      report.put("device", device());
      Properties pins = nativePins();
      report.put("mlx_c_pin", pins.getProperty("mlxcCommit", "unspecified"));
      report.put("mlx_metal_pin", pins.getProperty("mlxMetalVersion", "unspecified"));
      report.put("model_type", model.config().modelType());
      report.put("model_revision", "local-checkpoint-sha256");
      report.put(
          "dtype", config.path("dtype").asText(config.path("torch_dtype").asText("unspecified")));
      report.put("checkpoint_sliding_window", config.path("sliding_window").asText("null"));
      report.put("cache_policy", policy.mode().name());
      report.put("prefill_logits", "all_positions");
      report.put("batch", 1);
      report.put("prompt_length", prompt.length);
      report.put("tokens", tokens);
      report.put("warmups", warmups);
      report.put("samples", samples);
      report.put("cold_load_definition", "first load in fresh JVM; observed OS page-cache state");
      report.put("checkpoint_hash_timing", "after timed cold load");
      report.put("cold_load_ns", coldLoadNanos);
      report.put("active_before_load_bytes", beforeLoad);
      report.put("active_after_load_bytes", afterLoad);
      report.put("cold_load_peak_active_bytes", coldPeak);
      report.put("prefill_ns", stats(prefillNanos));
      report.put("one_token_decode_ns", stats(decodeNanos));
      report.put("sustained_generation_ns", stats(generationNanos));
      report.put("active_before_prefill_bytes", activeBeforePrefill);
      report.put("active_after_prefill_bytes", activeAfterPrefill);
      report.put("prefill_peak_active_bytes", prefillPeak);
      report.put("active_before_decode_bytes", activeBeforeDecode);
      report.put("active_after_decode_bytes", activeAfterDecode);
      report.put("decode_peak_active_bytes", decodePeak);
      report.put("active_before_generation_bytes", activeBeforeGeneration);
      report.put("tokens_per_second_median", tokens * 1_000_000_000.0 / median(generationNanos));
      report.put("active_after_generation_bytes", activeAfterGeneration);
      report.put("peak_generation_active_bytes", peakGeneration);
      report.put("active_per_generated_token_bytes", activePerGeneratedToken);
      report.put("allocator_cached_bytes", MLXMemory.cachedBytes());
      Path parent = output.toAbsolutePath().getParent();
      Files.createDirectories(parent);
      new ObjectMapper()
          .writeValue(output.resolveSibling(output.getFileName() + ".json").toFile(), report);
      String markdown =
          "# Decode benchmark\n\nCheckpoint: `"
              + checkpoint
              + "`\n\n"
              + "Cold load: "
              + coldLoadNanos / 1_000_000.0
              + " ms\n\n"
              + "Median prefill: "
              + median(prefillNanos) / 1_000_000.0
              + " ms\n\n"
              + "Median one-token decode: "
              + median(decodeNanos) / 1_000_000.0
              + " ms\n\n"
              + "Median sustained throughput: "
              + (tokens * 1_000_000_000.0 / median(generationNanos))
              + " tokens/s\n\n"
              + "Peak active native bytes: "
              + peakGeneration.stream().mapToLong(Long::longValue).max().orElse(0)
              + "\n";
      Files.writeString(output.resolveSibling(output.getFileName() + ".md"), markdown);
    }
  }

  /** The text half's prefix inside the composite checkpoint. */
  private static final String LANGUAGE_MODEL_PREFIX = "model.language_model.";

  /**
   * The default text-only prompt for the seam comparison: 16 IDs of the 21-token fixture
   * vocabulary, none of them a vision sentinel (the actual sentinel IDs are read from the
   * checkpoint and the prompt is validated against them).
   */
  private static final int[] DEFAULT_SEAM_PROMPT = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 14, 15, 16, 17, 18, 19
  };

  /**
   * The WP4 seam one-off (plan §7, PR #57 review): the same text-only request runs through the old
   * implicit-causal decode path (a plain Qwen3-family decoder derived in Java from the qwen3_vl
   * checkpoint's text half, below) and through the WP4 seam ({@code Qwen3VlModel}'s text-only
   * {@code generate}, whose prefill and every decode step pass an explicit {@code [1,1,1,L]} BOOL
   * batched mask plus the request's multi-axis rotation). On text-only input the two sides share
   * bit-exact text weights and one and the same angle table (proven by {@code
   * Qwen3VlTextOnlyEquivalenceTest}), so the delta isolates the seam machinery. Both models load in
   * the one fresh JVM and the sides alternate which runs first on each iteration.
   *
   * @param args {@code seam-comparison qwen3vl-checkpoint-dir output-prefix [prompt-csv] [tokens]
   *     [samples] [warmups]}
   * @throws Exception if loading, native evaluation, hashing, or report writing fails
   */
  private static void seamComparison(String[] args) throws Exception {
    if (args.length < 3 || args.length > 7) {
      throw new IllegalArgumentException(
          "usage: DecodeBenchmark seam-comparison qwen3vl-checkpoint-dir output-prefix"
              + " [prompt-csv] [tokens] [samples] [warmups]");
    }
    Path checkpoint = Path.of(args[1]);
    final Path output = Path.of(args[2]);
    final int[] prompt = args.length > 3 ? parsePrompt(args[3]) : DEFAULT_SEAM_PROMPT;
    int tokens = args.length > 4 ? Integer.parseInt(args[4]) : 32;
    int samples = args.length > 5 ? Integer.parseInt(args[5]) : 7;
    int warmups = args.length > 6 ? Integer.parseInt(args[6]) : 2;
    if (tokens < 1 || samples < 1 || warmups < 0) {
      throw new IllegalArgumentException("tokens/samples must be positive and warmups nonnegative");
    }
    // The composite rejects sliding retention, so the comparison runs the full policy.
    final GenerationCachePolicy policy = GenerationCachePolicy.full();
    ObjectMapper mapper = new ObjectMapper();
    JsonNode config = mapper.readTree(checkpoint.resolve("config.json").toFile());
    if (!"qwen3_vl".equals(config.path("model_type").asString())) {
      throw new IllegalArgumentException(
          "seam-comparison expects a model_type qwen3_vl checkpoint, got '"
              + config.path("model_type").asString()
              + "'");
    }
    JsonNode text = config.path("text_config");
    int headDim = text.path("head_dim").intValue();
    int[] section = {
      text.path("rope_scaling").path("mrope_section").get(0).intValue(),
      text.path("rope_scaling").path("mrope_section").get(1).intValue(),
      text.path("rope_scaling").path("mrope_section").get(2).intValue()
    };
    // The rotation-width invariant: the multi-axis kernel rotates 2 * sum(mrope_section)
    // dimensions, which must equal the family's head_dim so the derived plain decode rotates
    // the same width.
    if (2 * (section[0] + section[1] + section[2]) != headDim) {
      throw new IllegalArgumentException(
          "2 * sum(mrope_section) must equal head_dim so the seam and plain rotation widths"
              + " match");
    }
    int vocab = text.path("vocab_size").intValue();
    List<Integer> sentinels =
        List.of(
            config.path("image_token_id").intValue(),
            config.path("video_token_id").intValue(),
            config.path("vision_start_token_id").intValue(),
            config.path("vision_end_token_id").intValue());
    for (int id : prompt) {
      if (id < 0 || id >= vocab || sentinels.contains(id)) {
        throw new IllegalArgumentException(
            "prompt token "
                + id
                + " must be in [0, "
                + vocab
                + ") and avoid the vision sentinels "
                + sentinels
                + "; seam-comparison runs a text-only prompt");
      }
    }
    Path plainDir =
        output.toAbsolutePath().resolveSibling(output.getFileName() + "-plain-baseline");
    // Derive and verify the plain-qwen3 baseline before any timer starts.
    Map<String, String> baselineTensorSha =
        derivePlainBaseline(checkpoint, plainDir, config, mapper);
    long beforeLoad = MLXMemory.activeBytes();
    MLXMemory.resetPeak();
    try (MLXScope modelScope = new MLXScope()) {
      // Load order is old baseline first, seam checkpoint second, in the one fresh JVM; the
      // baseline files were just derived, so their page cache is warm.
      long oldLoadStart = System.nanoTime();
      DecoderModel oldModel = (DecoderModel) TextGenerationModels.load(modelScope, plainDir);
      final long oldLoadNanos = System.nanoTime() - oldLoadStart;
      if (!"qwen3".equals(oldModel.config().modelType())) {
        throw new IllegalStateException(
            "derived baseline loaded as model_type '" + oldModel.config().modelType() + "'");
      }
      final long afterOldLoad = MLXMemory.activeBytes();
      final long oldLoadPeak = MLXMemory.peakBytes();
      MLXMemory.resetPeak();
      long seamLoadStart = System.nanoTime();
      TextGenerationModel seamModel = TextGenerationModels.load(modelScope, checkpoint);
      final long seamLoadNanos = System.nanoTime() - seamLoadStart;
      if (!"qwen3_vl".equals(seamModel.metadata().modelType())) {
        throw new IllegalStateException(
            "seam checkpoint loaded as model_type '" + seamModel.metadata().modelType() + "'");
      }
      final long afterLoad = MLXMemory.activeBytes();
      final long seamLoadPeak = MLXMemory.peakBytes();
      // Hash only after the timed loads, as in the direct mode: hashing beforehand would warm
      // the OS page cache of the committed checkpoint before its timed load.
      final Map<String, String> checkpointHashes = checkpointHashes(checkpoint);
      final Map<String, String> baselineHashes = checkpointHashes(plainDir);
      KVCachePolicy resolvedPolicy = oldModel.resolveCachePolicy(policy);
      GenerationRequest request =
          new GenerationRequest(
                  prompt, GenerationConfig.greedyDefaults(tokens, Set.of()), CancellationToken.NONE)
              .withCachePolicy(policy);
      List<Long> oldPrefill = new ArrayList<>();
      List<Long> oldDecode = new ArrayList<>();
      List<Long> oldGen = new ArrayList<>();
      List<Long> oldPrefillBefore = new ArrayList<>();
      List<Long> oldPrefillAfter = new ArrayList<>();
      List<Long> oldPrefillPeak = new ArrayList<>();
      List<Long> oldDecodeBefore = new ArrayList<>();
      List<Long> oldDecodeAfter = new ArrayList<>();
      List<Long> oldDecodePeak = new ArrayList<>();
      List<Long> oldGenBefore = new ArrayList<>();
      List<Long> oldGenAfter = new ArrayList<>();
      List<Long> oldGenPeak = new ArrayList<>();
      List<List<Long>> oldPerToken = new ArrayList<>();
      List<List<Integer>> oldGenerated = new ArrayList<>();
      List<Long> seamGen = new ArrayList<>();
      List<Long> seamGenBefore = new ArrayList<>();
      List<Long> seamGenAfter = new ArrayList<>();
      List<Long> seamGenPeak = new ArrayList<>();
      List<List<Long>> seamPerToken = new ArrayList<>();
      List<List<Integer>> seamGenerated = new ArrayList<>();
      for (int run = -warmups; run < samples; run++) {
        OldRun old;
        SeamRun seam;
        // Alternate which side goes first so warm-up, thermal and allocator-order effects are
        // not always charged to the same one.
        if ((run & 1) == 0) {
          old = runOld(oldModel, resolvedPolicy, prompt, request, modelScope);
          seam = runSeam(seamModel, request);
        } else {
          seam = runSeam(seamModel, request);
          old = runOld(oldModel, resolvedPolicy, prompt, request, modelScope);
        }
        if (run >= 0) {
          oldPrefill.add(old.prefillNanos());
          oldDecode.add(old.decodeNanos());
          oldGen.add(old.generationNanos());
          oldPrefillBefore.add(old.prefillBefore());
          oldPrefillAfter.add(old.prefillAfter());
          oldPrefillPeak.add(old.prefillPeak());
          oldDecodeBefore.add(old.decodeBefore());
          oldDecodeAfter.add(old.decodeAfter());
          oldDecodePeak.add(old.decodePeak());
          oldGenBefore.add(old.generationBefore());
          oldGenAfter.add(old.generationAfter());
          oldGenPeak.add(old.generationPeak());
          oldPerToken.add(old.perToken());
          oldGenerated.add(old.generated());
          seamGen.add(seam.generationNanos());
          seamGenBefore.add(seam.generationBefore());
          seamGenAfter.add(seam.generationAfter());
          seamGenPeak.add(seam.generationPeak());
          seamPerToken.add(seam.perToken());
          seamGenerated.add(seam.generated());
        }
      }
      Map<String, Object> report = new LinkedHashMap<>();
      report.put("schema", "jmlx-decode-benchmark-3");
      report.put("mode", "seam-comparison");
      report.put("checkpoint", reportedCheckpoint(checkpoint));
      report.put("checkpoint_sha256", checkpointHashes);
      report.put("baseline_checkpoint", reportedCheckpoint(plainDir));
      report.put("baseline_checkpoint_sha256", baselineHashes);
      report.put(
          "baseline_derivation",
          "in-Java derivation of the checkpoint's text half: the "
              + baselineTensorSha.size()
              + " tensors model.language_model.* rekeyed to model.* and stored as an exact"
              + " BF16-to-FLOAT32 up-cast, every model.visual.* tensor dropped, the tied head"
              + " kept tied (no lm_head), a config.json with model_type qwen3 and the text"
              + " dimensions without rope_scaling (theta and head_dim kept; 2 * sum(mrope_section)"
              + " == head_dim, so the plain rotation width equals the seam's), and"
              + " tokenizer.json copied byte-identical (the plain loader does not read it)");
      Map<String, Object> verification = new LinkedHashMap<>();
      verification.put(
          "method",
          "every written tensor's FLOAT32 bytes (re-read from the derived file at the header's"
              + " own offsets) are element-wise bit-equal to an independent field-decomposition"
              + " decode (sign/exponent/mantissa through double) of the source BF16 bytes, a"
              + " different construction from the bit-shift write; the per-tensor SHA-256 of the"
              + " written bytes is recorded below, and the loader's tensor-plan validation is a"
              + " third check at load");
      verification.put("tensor_sha256", baselineTensorSha);
      report.put("text_weight_verification", verification);
      report.put("commit", commit());
      report.put("git_status_porcelain", gitStatus());
      report.put("java", System.getProperty("java.version"));
      report.put("os", System.getProperty("os.name"));
      report.put("architecture", System.getProperty("os.arch"));
      report.put("device", device());
      Properties pins = nativePins();
      report.put("mlx_c_pin", pins.getProperty("mlxcCommit", "unspecified"));
      report.put("mlx_metal_pin", pins.getProperty("mlxMetalVersion", "unspecified"));
      report.put(
          "tf32_mode",
          "MLX_ENABLE_TF32 unset by the task (native default); both sides run in the one JVM,"
              + " so both ran in the same mode");
      report.put("cache_policy", policy.mode().name());
      report.put("batch", 1);
      report.put("prompt", Arrays.stream(prompt).boxed().toList());
      report.put("prompt_length", prompt.length);
      report.put("tokens", tokens);
      report.put("warmups", warmups);
      report.put("samples", samples);
      report.put(
          "load_note",
          "both models load in the one fresh JVM (old baseline first: its files were just"
              + " derived, so their page cache is warm; seam checkpoint second); hashing after"
              + " both timed loads, as in the direct mode");
      report.put("old_load_ns", oldLoadNanos);
      report.put("seam_load_ns", seamLoadNanos);
      report.put("active_before_load_bytes", beforeLoad);
      report.put("active_after_old_load_bytes", afterOldLoad);
      report.put("active_after_load_bytes", afterLoad);
      report.put("old_load_peak_active_bytes", oldLoadPeak);
      report.put("seam_load_peak_active_bytes", seamLoadPeak);
      report.put(
          "memory_note",
          "both models stay resident; per-run active bytes include the other side's weight"
              + " set, so compare each side's before/after deltas and peaks, not absolutes");
      Map<String, Object> oldSide = new LinkedHashMap<>();
      oldSide.put("model_type", "qwen3");
      oldSide.put("path", "implicit causal (mask == null)");
      oldSide.put("prefill_logits", "all_positions");
      oldSide.put("prefill_ns", stats(oldPrefill));
      oldSide.put("one_token_decode_ns", stats(oldDecode));
      oldSide.put("sustained_generation_ns", stats(oldGen));
      oldSide.put("sustained_generation_samples_ns", oldGen);
      oldSide.put("prefill_samples_ns", oldPrefill);
      oldSide.put("one_token_decode_samples_ns", oldDecode);
      oldSide.put("tokens_per_second_median", tokens * 1_000_000_000.0 / median(oldGen));
      oldSide.put("generated_token_ids", oldGenerated);
      oldSide.put("active_before_prefill_bytes", oldPrefillBefore);
      oldSide.put("active_after_prefill_bytes", oldPrefillAfter);
      oldSide.put("prefill_peak_active_bytes", oldPrefillPeak);
      oldSide.put("active_before_decode_bytes", oldDecodeBefore);
      oldSide.put("active_after_decode_bytes", oldDecodeAfter);
      oldSide.put("decode_peak_active_bytes", oldDecodePeak);
      oldSide.put("active_before_generation_bytes", oldGenBefore);
      oldSide.put("active_after_generation_bytes", oldGenAfter);
      oldSide.put("peak_generation_active_bytes", oldGenPeak);
      oldSide.put("active_per_generated_token_bytes", oldPerToken);
      report.put("old", oldSide);
      Map<String, Object> seamSide = new LinkedHashMap<>();
      seamSide.put("model_type", "qwen3_vl");
      seamSide.put(
          "path",
          "WP4 seam: explicit [1,1,1,L] BOOL batched mask plus per-step multi-axis rotation"
              + " prepare; generate only, with the prefill (6-arg stepLogits with the rotation)"
              + " and every decode step inside it");
      seamSide.put("sustained_generation_ns", stats(seamGen));
      seamSide.put("sustained_generation_samples_ns", seamGen);
      seamSide.put("tokens_per_second_median", tokens * 1_000_000_000.0 / median(seamGen));
      seamSide.put("generated_token_ids", seamGenerated);
      seamSide.put("active_before_generation_bytes", seamGenBefore);
      seamSide.put("active_after_generation_bytes", seamGenAfter);
      seamSide.put("peak_generation_active_bytes", seamGenPeak);
      seamSide.put("active_per_generated_token_bytes", seamPerToken);
      report.put("seam", seamSide);
      Map<String, Object> ratios = new LinkedHashMap<>();
      ratios.put("sustained_generation_ns_seam_over_old", seamOverOld(seamGen, oldGen));
      report.put("ratios", ratios);
      report.put(
          "ratio_method",
          "seam[i]/old[i] in iteration order, then sort ratios; median index n/2,"
              + " p10 floor(0.1*(n-1)), p90 ceil(0.9*(n-1)); spread is not a confidence interval");
      Map<String, Object> scopeNotes = new LinkedHashMap<>();
      scopeNotes.put(
          "old_prefill_ns",
          "public forward over the full prompt, all positions' logits (overstates real prefill"
              + " cost); fresh KV caches per run");
      scopeNotes.put(
          "old_one_token_decode_ns",
          "public forward of the prompt's last token on the same run's warmed caches: the"
              + " implicit-causal lower bound without sampling overhead");
      scopeNotes.put(
          "old_sustained_generation_ns",
          "generate: prefill plus the decode steps with the sampling and streaming machinery:"
              + " the apples-to-apples pair with the seam side");
      scopeNotes.put(
          "seam_sustained_generation_ns",
          "generate of the identical text-only request through Qwen3VlModel; the only per-step"
              + " difference versus the old side is the seam machinery (explicit mask, rotation"
              + " prepare, hook wiring), so the delta isolates it");
      report.put("scope_notes", scopeNotes);
      Path parent = output.toAbsolutePath().getParent();
      Files.createDirectories(parent);
      mapper.writeValue(output.resolveSibling(output.getFileName() + ".json").toFile(), report);
      String markdown =
          "# Seam comparison benchmark (WP4 seam vs implicit causal)\n\nCheckpoint: `"
              + checkpoint
              + "` (qwen3_vl)\n\nBaseline: `"
              + plainDir
              + "` (plain qwen3 derived in Java; text weights verified bit-exact)\n\n"
              + "Prompt: "
              + Arrays.toString(prompt)
              + ", "
              + tokens
              + " new tokens, "
              + warmups
              + " warmups, "
              + samples
              + " samples, "
              + policy.mode().name()
              + " cache, batch 1; the sides alternate which runs first each iteration.\n\n"
              + "Old side (plain qwen3, mask == null): median prefill "
              + median(oldPrefill) / 1_000_000.0
              + " ms, median one-token decode "
              + median(oldDecode) / 1_000_000.0
              + " ms, median sustained "
              + (tokens * 1_000_000_000.0 / median(oldGen))
              + " tokens/s\n\n"
              + "Seam side (Qwen3VlModel text-only, explicit [1,1,1,L] mask, generate only):"
              + " median sustained "
              + (tokens * 1_000_000_000.0 / median(seamGen))
              + " tokens/s\n\n"
              + "Paired sustained seam/old ratios (p10/p90 use floor/ceil indices; with 7 samples"
              + " these are min/max): "
              + ratios.get("sustained_generation_ns_seam_over_old")
              + "\n";
      Files.writeString(output.resolveSibling(output.getFileName() + ".md"), markdown);
    }
  }

  /** One old-side iteration: the fine-grained forward timers plus the sustained generate. */
  private record OldRun(
      long prefillNanos,
      long decodeNanos,
      long generationNanos,
      long prefillBefore,
      long prefillAfter,
      long prefillPeak,
      long decodeBefore,
      long decodeAfter,
      long decodePeak,
      long generationBefore,
      long generationAfter,
      long generationPeak,
      List<Long> perToken,
      List<Integer> generated) {}

  /** One seam-side iteration: the sustained text-only generate (prefill and decode inside). */
  private record SeamRun(
      long generationNanos,
      long generationBefore,
      long generationAfter,
      long generationPeak,
      List<Long> perToken,
      List<Integer> generated) {}

  /**
   * The old side's timed run: a fresh KV cache set, one full-prompt {@code forward} (all positions'
   * logits), one one-token {@code forward} on the warmed caches, then the sustained {@code
   * generate} with a separate repeat sampling active bytes per emitted token.
   */
  private static OldRun runOld(
      DecoderModel model,
      KVCachePolicy resolvedPolicy,
      int[] prompt,
      GenerationRequest request,
      MLXScope modelScope) {
    ForwardStepTimings prefill;
    ForwardStepTimings decode;
    try (MLXScope inference = modelScope.newChild()) {
      List<KVCache> caches = new ArrayList<>();
      for (int layer = 0; layer < model.config().numHiddenLayers(); layer++) {
        caches.add(new KVCache(inference, resolvedPolicy));
      }
      try (MLXScope step = inference.newChild()) {
        MLXArray ids = MLX.array(step, prompt, new int[] {1, prompt.length});
        prefill = timedForwardStep(model, ids, caches);
      }
      try (MLXScope step = inference.newChild()) {
        MLXArray ids = MLX.array(step, new int[] {prompt[prompt.length - 1]}, new int[] {1, 1});
        decode = timedForwardStep(model, ids, caches);
      }
    }
    MLXMemory.resetPeak();
    long generationBefore = MLXMemory.activeBytes();
    long start = System.nanoTime();
    GenerationResult result = model.generate(request, ignored -> {});
    long generationNanos = System.nanoTime() - start;
    long generationAfter = MLXMemory.activeBytes();
    long generationPeak = MLXMemory.peakBytes();
    // Sample a separate repeat so memory-counter calls do not alter timed throughput.
    List<Long> interval = new ArrayList<>();
    model.generate(
        request,
        event -> {
          if (event.tokenId() != null) {
            interval.add(MLXMemory.activeBytes());
          }
        });
    return new OldRun(
        prefill.nanos(),
        decode.nanos(),
        generationNanos,
        prefill.before(),
        prefill.after(),
        prefill.peak(),
        decode.before(),
        decode.after(),
        decode.peak(),
        generationBefore,
        generationAfter,
        generationPeak,
        interval,
        result.generatedTokenIds());
  }

  /** One timed forward step's captures: the elapsed nanos and the before/after/peak bytes. */
  private record ForwardStepTimings(long nanos, long before, long after, long peak) {}

  /**
   * Times one public {@code forward} step with the memory captures around it, in the original
   * order: active bytes before, peak reset, the timed forward, then the elapsed nanos, active bytes
   * after and the step's peak.
   */
  private static ForwardStepTimings timedForwardStep(
      DecoderModel model, MLXArray ids, List<KVCache> caches) {
    final long before = MLXMemory.activeBytes();
    MLXMemory.resetPeak();
    final long start = System.nanoTime();
    model.forward(ids, caches);
    final long nanos = System.nanoTime() - start;
    final long after = MLXMemory.activeBytes();
    final long peak = MLXMemory.peakBytes();
    return new ForwardStepTimings(nanos, before, after, peak);
  }

  /**
   * The seam side's timed run: the sustained text-only {@code generate} of the identical request
   * (the prefill and every decode step run through the WP4 seam inside it), plus a separate repeat
   * sampling active bytes per emitted token.
   */
  private static SeamRun runSeam(TextGenerationModel model, GenerationRequest request) {
    MLXMemory.resetPeak();
    long generationBefore = MLXMemory.activeBytes();
    long start = System.nanoTime();
    GenerationResult result = model.generate(request, ignored -> {});
    long generationNanos = System.nanoTime() - start;
    long generationAfter = MLXMemory.activeBytes();
    long generationPeak = MLXMemory.peakBytes();
    List<Long> interval = new ArrayList<>();
    model.generate(
        request,
        event -> {
          if (event.tokenId() != null) {
            interval.add(MLXMemory.activeBytes());
          }
        });
    return new SeamRun(
        generationNanos,
        generationBefore,
        generationAfter,
        generationPeak,
        interval,
        result.generatedTokenIds());
  }

  /** Percentiles of per-iteration seam/old ratios, preserving the alternating-run pairs. */
  static Map<String, Double> seamOverOld(List<Long> seam, List<Long> old) {
    if (seam.isEmpty() || seam.size() != old.size()) {
      throw new IllegalArgumentException("samples must be nonempty and paired");
    }
    List<Double> paired = new ArrayList<>();
    for (int i = 0; i < old.size(); i++) {
      if (old.get(i) <= 0 || seam.get(i) <= 0) {
        throw new IllegalArgumentException("sample durations must be positive");
      }
      paired.add(seam.get(i) / (double) old.get(i));
    }
    List<Double> sorted = paired.stream().sorted().toList();
    int n = sorted.size();
    Map<String, Double> ratios = new LinkedHashMap<>();
    ratios.put("median", sorted.get(n / 2));
    ratios.put("p10", sorted.get((int) Math.floor(0.1 * (n - 1))));
    ratios.put("p90", sorted.get((int) Math.ceil(0.9 * (n - 1))));
    return ratios;
  }

  /**
   * Derives the plain-qwen3 baseline checkpoint from the composite source into {@code plainDir}:
   * the text tensors {@code model.language_model.*} rekeyed to {@code model.*} and stored as an
   * exact BF16-to-FLOAT32 up-cast (the composite loader applies the same up-cast, so both sides
   * operate on bit-identical FLOAT32 weights), every {@code model.visual.*} tensor dropped, the
   * tied head kept tied (no {@code lm_head}). The fresh {@code config.json} declares {@code
   * model_type "qwen3"} with the text dimensions and no {@code rope_scaling} (theta and head_dim
   * kept, so the plain decode rotates the same width as the seam); {@code tokenizer.json} is copied
   * byte-identical, although the plain loader does not read it. The derivation is verified before
   * it returns: each written tensor's FLOAT32 bytes (re-read from the derived file at the written
   * header's own offsets) are compared element-wise and bit-exactly against an independent decode
   * of the source BF16 bytes, and the per-tensor SHA-256 of the written bytes is returned for the
   * report.
   *
   * @return the SHA-256 of each re-keyed tensor's written FLOAT32 bytes, in file order
   */
  private static Map<String, String> derivePlainBaseline(
      Path source, Path plainDir, JsonNode config, ObjectMapper mapper) throws Exception {
    JsonNode text = config.path("text_config");
    final int layers = text.path("num_hidden_layers").intValue();
    byte[] file = Files.readAllBytes(source.resolve("model.safetensors"));
    long headerLen = ByteBuffer.wrap(file, 0, 8).order(ByteOrder.LITTLE_ENDIAN).getLong();
    if (headerLen <= 0 || 8 + headerLen > file.length) {
      throw new IllegalArgumentException("malformed safetensors header");
    }
    final JsonNode header = mapper.readTree(Arrays.copyOfRange(file, 8, 8 + (int) headerLen));
    final int dataStart = 8 + (int) headerLen;
    Set<String> expected = new HashSet<>();
    expected.add("model.embed_tokens.weight");
    expected.add("model.norm.weight");
    for (int i = 0; i < layers; i++) {
      String layer = "model.layers." + i + ".";
      for (String part :
          new String[] {
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight"
          }) {
        expected.add(layer + part);
      }
    }
    List<String> keys = new ArrayList<>();
    for (var entry : header.properties()) {
      if (entry.getKey().startsWith(LANGUAGE_MODEL_PREFIX)) {
        keys.add(entry.getKey());
      }
    }
    if (keys.size() != expected.size()) {
      throw new IllegalArgumentException(
          "expected " + expected.size() + " text tensors, found " + keys.size());
    }
    Map<String, byte[]> written = new LinkedHashMap<>();
    Map<String, int[]> sourceRange = new LinkedHashMap<>();
    Map<String, List<Integer>> shapes = new LinkedHashMap<>();
    for (String key : keys) {
      String newKey = "model." + key.substring(LANGUAGE_MODEL_PREFIX.length());
      if (!expected.remove(newKey)) {
        throw new IllegalArgumentException("unexpected text checkpoint key '" + key + "'");
      }
      JsonNode tensor = header.get(key);
      if (!"BF16".equals(tensor.path("dtype").asString())) {
        throw new IllegalArgumentException(
            "text tensor '"
                + key
                + "' must be stored BF16, got '"
                + tensor.path("dtype").asString()
                + "'");
      }
      List<Integer> shape = new ArrayList<>();
      for (var dimension : tensor.path("shape")) {
        shape.add(dimension.intValue());
      }
      long elements = 1;
      for (int d : shape) {
        elements *= d;
      }
      int start = tensor.path("data_offsets").get(0).intValue();
      int end = tensor.path("data_offsets").get(1).intValue();
      if (end - start != 2 * elements || dataStart + end > file.length) {
        throw new IllegalArgumentException("tensor '" + key + "' offsets disagree with its shape");
      }
      written.put(newKey, bf16ToFloat32(file, dataStart + start, dataStart + end));
      sourceRange.put(newKey, new int[] {dataStart + start, end - start});
      shapes.put(newKey, shape);
    }
    if (!expected.isEmpty()) {
      throw new IllegalArgumentException("missing text tensors " + expected);
    }
    int total = written.values().stream().mapToInt(b -> b.length).sum();
    byte[] data = new byte[total];
    Map<String, Object> headerObject = new LinkedHashMap<>();
    int offset = 0;
    for (var entry : written.entrySet()) {
      byte[] value = entry.getValue();
      System.arraycopy(value, 0, data, offset, value.length);
      Map<String, Object> tensorHeader = new LinkedHashMap<>();
      tensorHeader.put("dtype", "F32");
      tensorHeader.put("shape", shapes.get(entry.getKey()));
      tensorHeader.put("data_offsets", List.of(offset, offset + value.length));
      headerObject.put(entry.getKey(), tensorHeader);
      offset += value.length;
    }
    byte[] headerBytes = mapper.writeValueAsBytes(headerObject);
    byte[] out = new byte[8 + headerBytes.length + data.length];
    ByteBuffer.wrap(out, 0, 8).order(ByteOrder.LITTLE_ENDIAN).putLong(headerBytes.length);
    System.arraycopy(headerBytes, 0, out, 8, headerBytes.length);
    System.arraycopy(data, 0, out, 8 + headerBytes.length, data.length);
    Files.createDirectories(plainDir);
    Files.write(plainDir.resolve("model.safetensors"), out);
    List<String> layerTypes = new ArrayList<>();
    for (int i = 0; i < layers; i++) {
      layerTypes.add("full_attention");
    }
    Map<String, Object> plainConfig = new LinkedHashMap<>();
    plainConfig.put("architectures", List.of("Qwen3ForCausalLM"));
    plainConfig.put("attention_bias", text.path("attention_bias").asBoolean(false));
    plainConfig.put("attention_dropout", 0.0);
    plainConfig.put("dtype", "float32");
    plainConfig.put("head_dim", text.path("head_dim").intValue());
    plainConfig.put("hidden_act", text.path("hidden_act").asString("silu"));
    plainConfig.put("hidden_size", text.path("hidden_size").intValue());
    plainConfig.put("initializer_range", 0.02);
    plainConfig.put("intermediate_size", text.path("intermediate_size").intValue());
    plainConfig.put("layer_types", layerTypes);
    plainConfig.put("max_position_embeddings", text.path("max_position_embeddings").intValue());
    plainConfig.put("max_window_layers", layers);
    plainConfig.put("model_type", "qwen3");
    plainConfig.put("num_attention_heads", text.path("num_attention_heads").intValue());
    plainConfig.put("num_hidden_layers", layers);
    plainConfig.put("num_key_value_heads", text.path("num_key_value_heads").intValue());
    plainConfig.put("rms_norm_eps", text.path("rms_norm_eps").asDouble(1e-6));
    plainConfig.put("rope_scaling", null);
    plainConfig.put("rope_theta", text.path("rope_theta").asDouble(10_000));
    plainConfig.put("sliding_window", null);
    plainConfig.put("tie_word_embeddings", config.path("tie_word_embeddings").asBoolean(true));
    plainConfig.put(
        "transformers_version",
        config.has("transformers_version")
            ? config.get("transformers_version").asString()
            : "4.57.6");
    plainConfig.put("use_cache", true);
    plainConfig.put("use_sliding_window", false);
    plainConfig.put("vocab_size", text.path("vocab_size").intValue());
    mapper.writeValue(plainDir.resolve("config.json").toFile(), plainConfig);
    Path tokenizer = source.resolve("tokenizer.json");
    if (Files.isRegularFile(tokenizer)) {
      Files.copy(
          tokenizer, plainDir.resolve("tokenizer.json"), StandardCopyOption.REPLACE_EXISTING);
    }
    // Verify the write: re-read the derived file, follow its own header offsets, and compare
    // every element bit-exactly against an independent decode of the source BF16 bytes.
    byte[] writtenFile = Files.readAllBytes(plainDir.resolve("model.safetensors"));
    long writtenHeaderLen =
        ByteBuffer.wrap(writtenFile, 0, 8).order(ByteOrder.LITTLE_ENDIAN).getLong();
    JsonNode writtenHeader =
        mapper.readTree(Arrays.copyOfRange(writtenFile, 8, 8 + (int) writtenHeaderLen));
    int writtenDataStart = 8 + (int) writtenHeaderLen;
    Map<String, String> tensorSha = new LinkedHashMap<>();
    for (var entry : written.entrySet()) {
      String key = entry.getKey();
      JsonNode tensor = writtenHeader.get(key);
      if (tensor == null) {
        throw new IllegalStateException("derived file lost tensor '" + key + "'");
      }
      int start = writtenDataStart + tensor.path("data_offsets").get(0).intValue();
      int end = writtenDataStart + tensor.path("data_offsets").get(1).intValue();
      int n = entry.getValue().length / 4;
      int[] range = sourceRange.get(key);
      for (int j = 0; j < n; j++) {
        float actual = float32At(writtenFile, start + 4 * j);
        float wanted = bf16ToFloatIndependent(file, range[0], j);
        if (Float.floatToIntBits(actual) != Float.floatToIntBits(wanted)) {
          throw new AssertionError(
              "derived tensor '" + key + "' element " + j + " differs from the source BF16");
        }
      }
      tensorSha.put(key, sha256(Arrays.copyOfRange(writtenFile, start, end)));
    }
    return tensorSha;
  }

  /**
   * The write path: each little-endian BF16 word expands to the little-endian FLOAT32 word with the
   * same sign, exponent and mantissa bits (BF16 is the upper half of FLOAT32).
   */
  private static byte[] bf16ToFloat32(byte[] src, int srcStart, int srcEnd) {
    int n = (srcEnd - srcStart) / 2;
    byte[] out = new byte[4 * n];
    for (int i = 0; i < n; i++) {
      int u16 = (src[srcStart + 2 * i] & 0xFF) | ((src[srcStart + 2 * i + 1] & 0xFF) << 8);
      int u32 = u16 << 16;
      out[4 * i] = (byte) u32;
      out[4 * i + 1] = (byte) (u32 >>> 8);
      out[4 * i + 2] = (byte) (u32 >>> 16);
      out[4 * i + 3] = (byte) (u32 >>> 24);
    }
    return out;
  }

  /**
   * The verification decode: one BF16 word's sign, exponent and mantissa fields evaluated to the
   * exact value through double (BF16 subnormals included) — a different construction from the
   * bit-shift write, whose exact FLOAT32 result must match bit-for-bit. Non-finite source words are
   * rejected: the fixture carries finite weights only.
   */
  private static float bf16ToFloatIndependent(byte[] src, int srcStart, int index) {
    int u = (src[srcStart + 2 * index] & 0xFF) | ((src[srcStart + 2 * index + 1] & 0xFF) << 8);
    int exponent = (u >>> 7) & 0xFF;
    int mantissa = u & 0x7F;
    double value;
    if (exponent == 0) {
      value = mantissa * Math.scalb(1.0, -133);
    } else if (exponent == 0xFF) {
      throw new IllegalArgumentException(
          "source BF16 word 0x"
              + Integer.toHexString(u)
              + " is infinite or NaN; the derivation supports finite weights only");
    } else {
      value = (1.0 + mantissa / 128.0) * Math.scalb(1.0, exponent - 127);
    }
    return (float) (((u >>> 15) & 1) == 0 ? value : -value);
  }

  /** Reads one little-endian FLOAT32 word at {@code start}. */
  private static float float32At(byte[] bytes, int start) {
    int u32 =
        (bytes[start] & 0xFF)
            | ((bytes[start + 1] & 0xFF) << 8)
            | ((bytes[start + 2] & 0xFF) << 16)
            | ((bytes[start + 3] & 0xFF) << 24);
    return Float.intBitsToFloat(u32);
  }

  private static String sha256(byte[] bytes) throws Exception {
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    return HexFormat.of().formatHex(digest.digest(bytes));
  }

  private static int[] parsePrompt(String value) {
    int[] prompt = Arrays.stream(value.split(",")).mapToInt(Integer::parseInt).toArray();
    if (prompt.length == 0) {
      throw new IllegalArgumentException("prompt must contain token IDs");
    }
    return prompt;
  }

  /**
   * Reads the staged runtime's pins from the configured library directory. Empty when none is
   * configured -- e.g. when NativeLoader extracted the library from the classpath instead -- in
   * which case the report says "unspecified"; the Gradle task always sets the path.
   */
  private static Properties nativePins() throws Exception {
    String dir = System.getProperty("jmlx.library.path");
    if (dir == null || dir.isBlank()) {
      dir = System.getenv("JMLX_LIBRARY_PATH");
    }
    Properties pins = new Properties();
    if (dir != null && !dir.isBlank()) {
      Path file = Path.of(dir).resolve("native-pin.properties");
      if (Files.isRegularFile(file)) {
        try (InputStream input = Files.newInputStream(file)) {
          pins.load(input);
        }
      }
    }
    return pins;
  }

  /**
   * The checkpoint path as recorded in the report: relative to the repository root when the
   * checkpoint lies inside it, so committed reports carry no machine-specific home directory. The
   * root comes from {@code jmlx.repository.root} (set by the Gradle task) and otherwise the
   * absolute path is kept. The normalized paths are compared first, so a symlink inside the
   * repository is recorded under its in-repo name rather than its target. Only when that does not
   * match are both resolved to real paths, because Gradle reports the project directory with
   * symlinks resolved while the checkpoint argument may reach it through one (on macOS, {@code
   * /tmp} is {@code /private/tmp}).
   */
  private static String reportedCheckpoint(Path checkpoint) {
    Path absolute = checkpoint.toAbsolutePath().normalize();
    String root = System.getProperty("jmlx.repository.root");
    if (root != null && !root.isBlank()) {
      Path repository = Path.of(root).toAbsolutePath().normalize();
      if (!absolute.startsWith(repository)) {
        try {
          absolute = absolute.toRealPath();
          repository = repository.toRealPath();
        } catch (IOException unresolvable) {
          // Keep the normalized paths; at worst the absolute path is reported.
        }
      }
      if (absolute.startsWith(repository)) {
        return repository.relativize(absolute).toString();
      }
    }
    return absolute.toString();
  }

  /**
   * The device label: {@code JMLX_BENCH_DEVICE} when set, otherwise the macOS chip name from {@code
   * sysctl}, otherwise "unspecified".
   */
  private static String device() {
    String configured = System.getenv("JMLX_BENCH_DEVICE");
    if (configured != null && !configured.isBlank()) {
      return configured;
    }
    try {
      Process process =
          new ProcessBuilder("sysctl", "-n", "machdep.cpu.brand_string")
              .redirectErrorStream(true)
              .start();
      String chip = new String(process.getInputStream().readAllBytes()).trim();
      return process.waitFor() == 0 && !chip.isEmpty() ? chip : "unspecified";
    } catch (IOException unavailable) {
      return "unspecified";
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      return "unspecified";
    }
  }

  private static String gitStatus() {
    try {
      Process process = new ProcessBuilder("git", "status", "--porcelain").start();
      String status = new String(process.getInputStream().readAllBytes());
      return process.waitFor() == 0 ? status : "unknown";
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      return "unknown";
    } catch (IOException unavailable) {
      return "unknown";
    }
  }

  private static String commit() {
    String ci = System.getenv("GITHUB_SHA");
    if (ci != null && !ci.isBlank()) {
      return ci;
    }
    try {
      Process process = new ProcessBuilder("git", "rev-parse", "HEAD").start();
      String revision = new String(process.getInputStream().readAllBytes()).trim();
      return process.waitFor() == 0 ? revision : "unknown";
    } catch (Exception unavailable) {
      return "unknown";
    }
  }

  private static Map<String, String> checkpointHashes(Path root) throws Exception {
    Map<String, String> hashes = new LinkedHashMap<>();
    try (var files = Files.walk(root)) {
      for (Path path : files.filter(Files::isRegularFile).sorted().toList()) {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream input = Files.newInputStream(path)) {
          byte[] chunk = new byte[65536];
          int read;
          while ((read = input.read(chunk)) != -1) {
            digest.update(chunk, 0, read);
          }
        }
        hashes.put(root.relativize(path).toString(), HexFormat.of().formatHex(digest.digest()));
      }
    }
    return hashes;
  }

  private static Map<String, Long> stats(List<Long> data) {
    List<Long> sorted = data.stream().sorted().toList();
    return Map.of(
        "median",
        median(sorted),
        "p10",
        sorted.get((int) Math.floor(.1 * (sorted.size() - 1))),
        "p90",
        sorted.get((int) Math.ceil(.9 * (sorted.size() - 1))));
  }

  private static long median(List<Long> data) {
    List<Long> sorted = data.stream().sorted().toList();
    return sorted.get(sorted.size() / 2);
  }
}
