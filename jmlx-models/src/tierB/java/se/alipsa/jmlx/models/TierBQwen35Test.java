package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.zip.ZipFile;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXMemory;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCachePolicy;
import se.alipsa.jmlx.tokenizer.ChatTemplateOptions;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.vision.ImageDecoder;
import se.alipsa.jmlx.vision.RgbImage;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/** Pinned real hybrid acceptance; the manifest fixes numerical/stability/resource bounds first. */
@EnabledIfNativeAvailable
class TierBQwen35Test {
  private static final ObjectMapper MAPPER = new ObjectMapper();
  private static final int TOKENS = 8;

  @org.junit.jupiter.api.Test
  void pinnedTextImageLogitsStableGreedyIdsAndDecodedOutput() throws Exception {
    final TierBReferences reference = new TierBReferences("qwen35-generation");
    final Path repository = Path.of(System.getProperty("jmlx.repository.root"));
    Path base = Path.of(System.getProperty("jmlx.qwen35.acceptance.output"));
    Files.createDirectories(base);
    Path output = Files.createTempDirectory(base, "run-");
    System.out.println("Qwen3.5 acceptance evidence: " + output);
    Map<String, Object> report = new LinkedHashMap<>();
    report.put("status", "in progress; failed assertions preserve partial measurements");
    report.put("started_utc", Instant.now().toString());
    report.put("git_commit", command("git", "rev-parse", "HEAD"));
    report.put("working_tree_dirty", !command("git", "status", "--porcelain").isBlank());
    report.put("runtime_pin", TierBReferences.runtimePin());
    report.put("mlx_enable_tf32", System.getenv("MLX_ENABLE_TF32"));
    report.put("epsilon", reference.epsilon);
    report.put("arithmetic_epsilon", reference.reference.required("arithmetic_epsilon").asDouble());
    report.put(
        "normalization_contribution_max_abs",
        reference.reference.required("normalization_contribution_max_abs").asDouble());
    report.put("margin", reference.margin);
    report.put("host_before", host());
    List<Map<String, Object>> cases = new ArrayList<>();
    report.put("cases", cases);
    Path reportFile = output.resolve("report.json");
    writeReport(reportFile, report);
    JsonNode logitsSpec = reference.reference.required("logits_file");
    Path archive = repository.resolve(logitsSpec.required("path").asString());
    assertSha(archive, logitsSpec.required("sha256").asString());
    JsonNode inputSpec = reference.reference.required("input_record");
    Path inputFile = repository.resolve(inputSpec.required("path").asString());
    assertSha(inputFile, inputSpec.required("sha256").asString());
    JsonNode sourceInputs = MAPPER.readTree(inputFile.toFile());
    HfTokenizer tokenizer = HfTokenizer.fromDirectory(reference.directory);
    Path imageFile =
        repository.resolve(
            reference.manifest.required("images").get(0).required("path").asString());
    assertSha(
        imageFile, reference.manifest.required("images").get(0).required("sha256").asString());
    RgbImage image = ImageDecoder.decode(imageFile);
    try (TierBFootprintSampler footprint = new TierBFootprintSampler();
        MLXScope root = new MLXScope()) {
      long loadStart = System.nanoTime();
      Qwen35Model model = (Qwen35Model) TextGenerationModels.load(root, reference.directory);
      report.put("load_wall_ms", (System.nanoTime() - loadStart) / 1_000_000.0);
      reference.model(model.metadata());
      assertEquals("qwen3_5", model.metadata().modelType());
      assertEquals(
          Set.of(InputModality.TEXT, InputModality.IMAGE), model.metadata().inputModalities());
      for (JsonNode item : reference.reference.required("cases")) {
        final String name = item.required("name").asString();
        boolean visual = item.required("image").asBoolean();
        List<RgbImage> images = visual ? List.of(image) : List.of();
        String content =
            item.path("user_content")
                .asString(
                    visual
                        ? "<|vision_start|><|image_pad|><|vision_end|>Describe this image briefly."
                        : "What is two plus two? Answer briefly.");
        GenerationRequest request =
            GenerationRequest.chat(
                    tokenizer,
                    List.of(Map.of("role", "user", "content", content)),
                    new ChatTemplateOptions("", true, Map.of("enable_thinking", false)),
                    GenerationConfig.greedyDefaults(TOKENS, Set.of()),
                    CancellationToken.NONE)
                .withImages(images);
        assertEquals(item.required("rendered_prompt").asString(), request.promptText());
        assertEquals(
            TierBReferences.ids(item.required("original_ids")),
            java.util.Arrays.stream(request.promptTokenIds()).boxed().toList());
        Qwen3VlPromptPlan plan = model.plan(request.promptTokenIds(), images);
        assertEquals(
            TierBReferences.ids(item.required("expanded_ids")),
            java.util.Arrays.stream(plan.expandedIds()).boxed().toList());
        assertEquals(item.required("position_delta").asInt(), plan.positionDelta());
        assertArrayEquals(ints(item.required("positions").get(0)), plan.temporalPositions());
        assertArrayEquals(ints(item.required("positions").get(1)), plan.heightPositions());
        assertArrayEquals(ints(item.required("positions").get(2)), plan.widthPositions());
        if (visual) {
          verifyPixels(model, image, sourceInputs, name, inputFile.getParent());
        }
        Map<String, Object> measured = new LinkedHashMap<>();
        measured.put("name", name);
        measured.put("expanded_positions", plan.expandedLength());
        measured.put("position_delta", plan.positionDelta());
        cases.add(measured);
        List<Double> warm = new ArrayList<>();
        for (int sample = -1; sample < 3; sample++) {
          long start = System.nanoTime();
          GenerationResult result = model.generate(request, ignored -> {});
          double elapsed = (System.nanoTime() - start) / 1_000_000.0;
          if (sample < 0) {
            measured.put("first_generation_after_load_or_previous_case_ms", elapsed);
            measured.put("generated_ids", result.generatedTokenIds());
            measured.put("decoded", result.generatedText());
          } else {
            warm.add(elapsed);
          }
          writeReport(reportFile, report);
          assertEquals(
              TierBReferences.ids(item.required("greedy_ids")), result.generatedTokenIds());
          assertEquals(item.required("expected_text").asString(), result.generatedText());
          assertEquals(FinishReason.MAX_TOKENS, result.finishReason());
          assertEquals(plan.expandedLength(), result.promptPositions());
        }
        measured.put("warmed_generation_samples_ms", warm);
        measured.put(
            "timing_scope",
            "generate from an already rendered/tokenized request and RgbImage; 8 tokens, EOS"
                + " stopping disabled; case order and host caches retained");
        List<Map<String, Object>> phases = new ArrayList<>();
        measured.put("phases", phases);
        try (MLXScope generation = root.newChild();
            Qwen35Decoder.Session session =
                model.decoderForTest().newSession(generation, KVCachePolicy.full())) {
          for (int step = 0; step < item.required("logits").size(); step++) {
            String phase = item.required("logits").get(step).asString();
            float[] expected = readVector(archive, phase, model.metadata().vocabSize());
            try (MLXScope activation = generation.newChild()) {
              MLXArray logits;
              if (step == 0) {
                MLXArray ids =
                    MLX.array(activation, plan.expandedIds(), new int[] {1, plan.expandedLength()});
                var rotation =
                    model
                        .descriptorForTest()
                        .text()
                        .rope()
                        .prepare(
                            activation,
                            1,
                            plan.expandedLength(),
                            new int[][] {
                              plan.temporalPositions(),
                              plan.heightPositions(),
                              plan.widthPositions()
                            });
                MLXArray rows =
                    images.isEmpty()
                        ? null
                        : model.features(CancellationToken.NONE, images, activation);
                logits =
                    session.build(
                        ids,
                        rotation,
                        true,
                        rows == null
                            ? null
                            : embedded ->
                                VisualFeatureInjection.replaceImageRows(
                                    embedded,
                                    MLXShape.reshape(
                                        rows,
                                        new int[] {
                                          1,
                                          plan.totalFeatureRows(),
                                          model.descriptorForTest().text().hidden()
                                        }),
                                    plan.featureDestinations()));
              } else {
                int token = item.required("greedy_ids").get(step - 1).asInt();
                int position = plan.ropePositionAt(session.position());
                MLXArray ids = MLX.array(activation, new int[] {token}, new int[] {1, 1});
                logits =
                    session.build(
                        ids,
                        model
                            .descriptorForTest()
                            .text()
                            .rope()
                            .prepare(
                                activation, 1, 1, new int[][] {{position}, {position}, {position}}),
                        true);
              }
              session.evaluateAndCommit(logits, StepBoundaryEvaluator.NATIVE);
              float[] actual = logits.toFloatArray();
              final double error = maximumError(expected, actual);
              Path vector = output.resolve(phase + ".f32");
              ByteBuffer bytes =
                  ByteBuffer.allocate(actual.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
              bytes.asFloatBuffer().put(actual);
              Files.write(vector, bytes.array());
              Map<String, Object> result = new LinkedHashMap<>();
              result.put("phase", phase);
              result.put("maximum_stock_error", error);
              result.put("java_gap", gap(actual));
              result.put("reference_gap", gap(expected));
              result.put("java_greedy", best(actual));
              result.put("reference_greedy", best(expected));
              result.put("actual_vector_file", vector.getFileName().toString());
              result.put("actual_vector_sha256", sha(vector));
              phases.add(result);
              writeReport(reportFile, report);
              reference.compare(expected, actual);
              reference.stable(gap(expected));
              reference.stable(gap(actual));
              assertEquals(best(expected), best(actual));
            }
          }
        }
        writeReport(reportFile, report);
      }
      report.put("native_peak_bytes", MLXMemory.peakBytes());
      report.put("sampled_peak_phys_footprint_bytes", footprint.maximum());
      report.put("footprint_sample_interval_ms", 20);
      JsonNode limits = reference.manifest.required("resource_limits");
      writeReport(reportFile, report);
      assertTrue(MLXMemory.peakBytes() <= limits.required("native_peak_bytes").asLong());
      assertTrue(footprint.maximum() <= limits.required("sampled_phys_footprint_bytes").asLong());
      reference.report();
    }
    report.put("host_after", host());
    report.put(
        "status", "passed numerical, stability, exact IDs/decoded output and resource gates");
    report.put("finished_utc", Instant.now().toString());
    writeReport(reportFile, report);
  }

  private static void verifyPixels(
      Qwen35Model model, RgbImage image, JsonNode inputs, String name, Path directory)
      throws Exception {
    for (JsonNode source : inputs.required("cases")) {
      if (source.required("name").asString().equals(name)) {
        Path file = directory.resolve(source.required("pixel_values_file").asString());
        assertSha(file, source.required("pixel_values_sha256").asString());
        var actual = model.processorForTest().process(image);
        assertArrayEquals(ints(source.required("image_grid_thw").get(0)), actual.gridThw());
        assertArrayEquals(
            readNpy(Files.readAllBytes(file), actual.rawPatchRows(), actual.patchStride()),
            actual.values());
        return;
      }
    }
    throw new AssertionError("missing image input provenance: " + name);
  }

  private static float[] readVector(Path archive, String name, int vocabulary) throws Exception {
    try (ZipFile zip = new ZipFile(archive.toFile())) {
      var entry = zip.getEntry(name + ".npy");
      assertTrue(entry != null, name);
      try (var stream = zip.getInputStream(entry)) {
        return readNpy(stream.readAllBytes(), 1, vocabulary);
      }
    }
  }

  private static float[] readNpy(byte[] bytes, int rows, int columns) {
    ByteBuffer input = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    byte[] magic = new byte[6];
    input.get(magic);
    assertArrayEquals(new byte[] {(byte) 0x93, 'N', 'U', 'M', 'P', 'Y'}, magic);
    int version = Byte.toUnsignedInt(input.get());
    assertEquals(0, input.get());
    assertTrue(version == 1 || version == 2);
    int length = version == 1 ? Short.toUnsignedInt(input.getShort()) : input.getInt();
    assertTrue(length > 0 && length <= input.remaining());
    byte[] headerBytes = new byte[length];
    input.get(headerBytes);
    String header = new String(headerBytes, StandardCharsets.US_ASCII);
    assertTrue(
        header.contains("'descr': '<f4'") && header.contains("'fortran_order': False"), header);
    assertTrue(
        header.matches("(?s).*'shape': \\(\\s*" + rows + "\\s*,\\s*" + columns + "\\s*,?\\).*"),
        header);
    int count = Math.multiplyExact(rows, columns);
    assertEquals(Math.multiplyExact(count, Float.BYTES), input.remaining());
    float[] result = new float[count];
    input.asFloatBuffer().get(result);
    return result;
  }

  private static double maximumError(float[] expected, float[] actual) {
    assertEquals(expected.length, actual.length);
    double result = 0;
    for (int i = 0; i < actual.length; i++) {
      assertTrue(Float.isFinite(actual[i]));
      result = Math.max(result, Math.abs((double) expected[i] - actual[i]));
    }
    return result;
  }

  private static int best(float[] values) {
    int best = 0;
    for (int i = 1; i < values.length; i++) {
      if (values[i] > values[best]) {
        best = i;
      }
    }
    return best;
  }

  private static double gap(float[] values) {
    int best = best(values);
    float second = Float.NEGATIVE_INFINITY;
    for (int i = 0; i < values.length; i++) {
      if (i != best) {
        second = Math.max(second, values[i]);
      }
    }
    return (double) values[best] - second;
  }

  private static int[] ints(JsonNode values) {
    return TierBReferences.ids(values).stream().mapToInt(Integer::intValue).toArray();
  }

  private static void assertSha(Path file, String expected) throws Exception {
    assertEquals(expected, sha(file), file.toString());
  }

  private static String sha(Path file) throws Exception {
    MessageDigest hash = MessageDigest.getInstance("SHA-256");
    try (var stream = Files.newInputStream(file)) {
      byte[] bytes = new byte[1024 * 1024];
      int count;
      while ((count = stream.read(bytes)) != -1) {
        hash.update(bytes, 0, count);
      }
    }
    return HexFormat.of().formatHex(hash.digest());
  }

  private static void writeReport(Path file, Map<String, Object> report) throws Exception {
    MAPPER.writerWithDefaultPrettyPrinter().writeValue(file.toFile(), report);
  }

  private static Map<String, String> host() throws Exception {
    return Map.of(
        "observed_utc",
        Instant.now().toString(),
        "os",
        System.getProperty("os.version"),
        "architecture",
        System.getProperty("os.arch"),
        "cpu",
        command("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"),
        "uptime",
        command("uptime"),
        "vm_stat",
        command("vm_stat"),
        "thermal",
        command("pmset", "-g", "therm"),
        "process_cpu_observation",
        command("ps", "-axo", "pid,pcpu,comm"));
  }

  private static String command(String... arguments) throws Exception {
    Process process = new ProcessBuilder(arguments).redirectErrorStream(true).start();
    String result =
        new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8).trim();
    assertEquals(0, process.waitFor(), String.join(" ", arguments));
    return result;
  }
}
