package smoke;

import java.io.IOException;
import java.io.InputStream;
import java.io.Reader;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.CodeSource;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import se.alipsa.jmlx.models.BatchGenerationScheduler;
import se.alipsa.jmlx.models.BatchRequestHandle;
import se.alipsa.jmlx.models.BatchSchedulerConfig;
import se.alipsa.jmlx.models.CancellationToken;
import se.alipsa.jmlx.models.FinishReason;
import se.alipsa.jmlx.models.GenerationConfig;
import se.alipsa.jmlx.models.GenerationRequest;
import se.alipsa.jmlx.models.GenerationResult;
import se.alipsa.jmlx.models.TextGenerationModels;
import se.alipsa.jmlx.tokenizer.ChatTemplateOptions;
import se.alipsa.jmlx.tokenizer.HfTokenizer;

/**
 * Release smoke: runs against the published jars only, with no checkout classpath and no native
 * path override. Fails with a message on the first violated expectation; with {@code
 * -PsmokeRecord=true} it (re)writes the committed golden instead of comparing against it.
 */
public final class ReleaseSmoke {
  private static final long SEED = 42;
  private static final float TEMPERATURE = 0.8f;
  private static final int NEW_TOKENS = 12;
  private static final int SECOND_NEW_TOKENS = 6;
  private static final List<Map<String, Object>> MESSAGES =
      List.of(
          Map.of("role", "system", "content", "be brief"),
          Map.of("role", "user", "content", "the fox and the hen"));
  private static final List<Map<String, Object>> SECOND_MESSAGES =
      List.of(Map.of("role", "user", "content", "once in a while"));
  private static final List<String> PINNED_FILES =
      List.of(
          "libmlxc.dylib",
          "libmlx.dylib",
          "libjaccl.dylib",
          "mlx.metallib",
          "native-pin.properties");
  private static final Pattern SNAPSHOT_STAMP = Pattern.compile("-\\d{8}\\.\\d{6}-\\d+");

  private ReleaseSmoke() {}

  /**
   * Runs the smoke.
   *
   * @param args unused
   * @throws Exception on any failed expectation
   */
  public static void main(String[] args) throws Exception {
    Path modelDir = Path.of(requireProperty("smoke.model.dir"));
    Path cache = Path.of(requireProperty("jmlx.native.cache.path"));
    boolean record = Boolean.parseBoolean(requireProperty("smoke.record"));

    check(System.getProperty("jmlx.library.path") == null, "jmlx.library.path must not be set");
    check(System.getenv("JMLX_LIBRARY_PATH") == null, "JMLX_LIBRARY_PATH must not be set");
    check(isEmptyDirectory(cache), "the native cache must be empty before the first load: " + cache);

    verifyJarProvenance();

    HfTokenizer tokenizer = HfTokenizer.fromDirectory(modelDir);
    String configJson = Files.readString(modelDir.resolve("config.json"));
    int vocabSize = intField(configJson, "vocab_size");
    int eos = intField(configJson, "eos_token_id");
    check(
        tokenizer.vocabSize() == vocabSize,
        "tokenizer ids 0.." + (tokenizer.vocabSize() - 1) + " != checkpoint vocab_size " + vocabSize);
    check(
        tokenizer.eosTokenId("</s>").orElseThrow() == eos
            && tokenizer.bosTokenId("<s>").orElseThrow() == intField(configJson, "bos_token_id"),
        "tokenizer BOS/EOS do not match the checkpoint");
    Set<Integer> eosIds = Set.of(eos);

    GenerationRequest first = request(tokenizer, MESSAGES, sampled(eosIds));
    GenerationRequest second = request(tokenizer, SECOND_MESSAGES, greedy(eosIds));
    List<String> deltas = new CopyOnWriteArrayList<>();
    BatchGenerationResult run;
    // maxBatch 2: the two requests are a bounded batch. close() stops admission and the worker
    // closes the model and its scope.
    try (BatchGenerationScheduler scheduler =
        BatchGenerationScheduler.start(
            new BatchSchedulerConfig(2, 4, 4096, 64),
            scope -> TextGenerationModels.load(scope, modelDir))) {
      BatchRequestHandle a =
          scheduler.submit(
              first,
              event -> {
                if (event.textDelta() != null) {
                  deltas.add(event.textDelta());
                }
              });
      BatchRequestHandle b = scheduler.submit(second, event -> {});
      run =
          new BatchGenerationResult(
              a.stage().toCompletableFuture().join(), b.stage().toCompletableFuture().join());
    }

    verifyExtraction(cache);
    GenerationResult firstResult = run.first();
    GenerationResult secondResult = run.second();
    check(
        secondResult.generatedTokenIds().size() == SECOND_NEW_TOKENS
            || secondResult.finishReason() == FinishReason.EOS,
        "second request: " + secondResult);
    check(!deltas.isEmpty() && deltas.stream().anyMatch(d -> !d.isEmpty()), "no non-empty delta");
    for (int id : firstResult.generatedTokenIds()) {
      boolean special = id <= 2;
      check(
          special || !tokenizer.decode(List.of(id), false).isEmpty(),
          "generated id " + id + " decodes to nothing");
    }

    Path goldenPath = Path.of(requireProperty("smoke.golden"));
    Properties actual = golden(firstResult, deltas, nativePin(cache), first);
    if (record) {
      try (Writer out = Files.newBufferedWriter(goldenPath)) {
        store(actual, out);
      }
      System.out.println("recorded " + goldenPath);
    } else {
      Properties expected = new Properties();
      try (Reader in = Files.newBufferedReader(goldenPath)) {
        expected.load(in);
      }
      check(
          expected.stringPropertyNames().equals(actual.stringPropertyNames()),
          "golden keys differ: " + expected.stringPropertyNames() + " vs " + actual.stringPropertyNames());
      for (String key : expected.stringPropertyNames()) {
        check(
            expected.getProperty(key).equals(actual.getProperty(key)),
            "golden mismatch for " + key + ": expected [" + expected.getProperty(key)
                + "] actual [" + actual.getProperty(key) + "]");
      }
    }
    System.out.println("release smoke OK: " + firstResult.generatedTokenIds() + " " + deltas);
  }

  private record BatchGenerationResult(GenerationResult first, GenerationResult second) {}

  private static GenerationConfig sampled(Set<Integer> eos) {
    return GenerationConfig.samplingDefaults(NEW_TOKENS, SEED, TEMPERATURE, eos);
  }

  private static GenerationConfig greedy(Set<Integer> eos) {
    return GenerationConfig.greedyDefaults(SECOND_NEW_TOKENS, eos);
  }

  private static GenerationRequest request(
      HfTokenizer tokenizer, List<Map<String, Object>> messages, GenerationConfig config) {
    return GenerationRequest.chat(
        tokenizer, messages, new ChatTemplateOptions("", true, Map.of()), config,
        CancellationToken.NONE);
  }

  private static Properties golden(
      GenerationResult result, List<String> deltas, Properties pin, GenerationRequest request) {
    Properties out = new Properties();
    out.setProperty("seed", Long.toString(SEED));
    out.setProperty("policy", "sampling temperature=" + TEMPERATURE + " maxNewTokens=" + NEW_TOKENS);
    out.setProperty("batch", "maxBatch=2 requests=2 companion=greedy:" + SECOND_NEW_TOKENS);
    out.setProperty("native.mlxMetalVersion", pin.getProperty("mlxMetalVersion"));
    out.setProperty("native.mlxcCommit", pin.getProperty("mlxcCommit"));
    out.setProperty("promptIds", result.promptTokenIds().toString());
    out.setProperty("generatedIds", result.generatedTokenIds().toString());
    out.setProperty("finishReason", result.finishReason().name());
    out.setProperty("deltaCount", Integer.toString(deltas.size()));
    for (int i = 0; i < deltas.size(); i++) {
      out.setProperty("delta." + i, deltas.get(i));
    }
    return out;
  }

  /** Writes keys in sorted order so a re-recorded golden produces a stable diff. */
  private static void store(Properties source, Writer out) throws IOException {
    out.write("# release-smoke golden: regenerate with -PsmokeRecord=true\n");
    for (String key : new java.util.TreeSet<>(source.stringPropertyNames())) {
      Properties one = new Properties();
      one.setProperty(key, source.getProperty(key));
      java.io.StringWriter line = new java.io.StringWriter();
      one.store(line, null);
      line.toString().lines().filter(l -> !l.startsWith("#")).forEach(l -> {
        try {
          out.write(l + "\n");
        } catch (IOException e) {
          throw new java.io.UncheckedIOException(e);
        }
      });
    }
  }

  /**
   * Every jmlx class must come from a jar in the resolved dependency cache: not from a checkout
   * build directory, and, in CI mode, not from {@code mavenLocal}.
   */
  private static void verifyJarProvenance() throws Exception {
    Path checkout = Path.of(requireProperty("smoke.checkout.root")).toAbsolutePath().normalize();
    Map<String, String> expected = expectedVersions();
    List<Class<?>> probes =
        List.of(
            BatchGenerationScheduler.class,
            se.alipsa.jmlx.core.MLX.class,
            // jmlx-ffi is runtime-only for a consumer (jmlx-core's `implementation` dependency).
            Class.forName("se.alipsa.jmlx.ffi.NativeLoader"),
            HfTokenizer.class,
            se.alipsa.jmlx.jinja.Template.class);
    for (Class<?> probe : probes) {
      CodeSource source = probe.getProtectionDomain().getCodeSource();
      check(source != null, probe + " has no code source");
      Path jar = Path.of(java.net.URI.create(source.getLocation().toString())).toAbsolutePath();
      check(jar.toString().endsWith(".jar"), probe + " is not loaded from a jar: " + jar);
      Path normalized = jar.normalize();
      // The smoke's own tree lives inside the checkout, so exclude it; any other path under the
      // checkout (build/libs, build/classes) is a leak.
      check(
          !normalized.startsWith(checkout) || normalized.startsWith(checkout.resolve("tools/release-smoke")),
          probe + " came from the checkout: " + jar);
      check(!jar.toString().contains("/.m2/"), probe + " came from mavenLocal: " + jar);
      String artifact = jar.getFileName().toString();
      String module = expected.keySet().stream().filter(artifact::startsWith).findFirst().orElse(null);
      check(module != null, "unexpected jar " + artifact);
      String version = SNAPSHOT_STAMP.matcher(artifact).replaceAll("-SNAPSHOT");
      check(
          version.equals(module + "-" + expected.get(module) + ".jar"),
          artifact + " is not the expected version " + expected.get(module));
    }
  }

  private static Map<String, String> expectedVersions() {
    Map<String, String> out = new java.util.LinkedHashMap<>();
    for (String pair : requireProperty("smoke.expected.versions").split(",")) {
      String[] kv = pair.split("=");
      out.put(kv[0], kv[1]);
    }
    return out;
  }

  /** The pinned files were freshly extracted from the packaged jar into the disposable cache. */
  private static void verifyExtraction(Path cache) throws IOException {
    check(Files.isDirectory(cache), "no native extraction under " + cache);
    List<Path> dirs;
    try (var stream = Files.list(cache)) {
      dirs = stream.filter(p -> Files.isDirectory(p) && !p.getFileName().toString().startsWith(".")).toList();
    }
    check(dirs.size() == 1, "expected exactly one extracted pin directory, got " + dirs);
    for (String name : PINNED_FILES) {
      check(Files.isRegularFile(dirs.getFirst().resolve(name)), "extraction lacks " + name);
    }
  }

  private static Properties nativePin(Path cache) throws IOException {
    Properties pin = new Properties();
    try (var stream = Files.list(cache)) {
      Path dir = stream.filter(p -> Files.isDirectory(p) && !p.getFileName().toString().startsWith(".")).findFirst().orElseThrow();
      try (InputStream in = Files.newInputStream(dir.resolve("native-pin.properties"))) {
        pin.load(in);
      }
    }
    return pin;
  }

  private static boolean isEmptyDirectory(Path dir) throws IOException {
    if (!Files.exists(dir)) {
      return true;
    }
    try (var stream = Files.list(dir)) {
      return stream.findAny().isEmpty();
    }
  }

  private static int intField(String json, String field) {
    Matcher matcher = Pattern.compile("\"" + field + "\"\\s*:\\s*(\\d+)").matcher(json);
    check(matcher.find(), "config.json lacks " + field);
    return Integer.parseInt(matcher.group(1));
  }

  private static String requireProperty(String name) {
    String value = System.getProperty(name);
    check(value != null && !value.isBlank(), "missing system property " + name);
    return value;
  }

  private static void check(boolean condition, String message) {
    if (!condition) {
      throw new IllegalStateException("release smoke FAILED: " + message);
    }
  }
}
