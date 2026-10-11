package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;
import se.alipsa.jmlx.tokenizer.ChatTemplateOptions;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.vision.ImageDecoder;
import se.alipsa.jmlx.vision.RgbImage;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/** Opt-in pinned real image/text references, stable greedy IDs and measured memory. */
@EnabledIfNativeAvailable
class TierBVisionLanguageTest {
  @Test
  void expandedPromptCachedLogitsAndStableGreedyIds() throws Exception {
    TierBReferences reference = new TierBReferences("image-text-generation");
    final HfTokenizer tokenizer = HfTokenizer.fromDirectory(reference.directory);
    Path repository = Path.of(System.getProperty("jmlx.repository.root"));
    var logitsSpec = reference.reference.required("logits_file");
    Path logitsFile = repository.resolve(logitsSpec.required("path").asString());
    assertEquals(
        logitsSpec.required("sha256").asString(),
        HexFormat.of()
            .formatHex(
                MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(logitsFile))));
    JsonNode logitsGolden = JsonFiles.read(logitsFile).required("cases");
    assertEquals(reference.reference.required("cases").size(), logitsGolden.size());
    List<RgbImage> images = new ArrayList<>();
    for (JsonNode spec : reference.manifest.required("images")) {
      Path file = repository.resolve(spec.required("path").asString());
      assertEquals(
          spec.required("sha256").asString(),
          HexFormat.of()
              .formatHex(MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(file))));
      assertTrue(!spec.required("license").asString().isBlank());
      images.add(ImageDecoder.decode(file));
    }
    try (RssSampler rss = new RssSampler();
        MLXScope root = new MLXScope()) {
      SmolVlmModel model = (SmolVlmModel) TextGenerationModels.load(root, reference.directory);
      reference.model(model.metadata());
      int caseIndex = 0;
      for (JsonNode input : reference.reference.required("cases")) {
        JsonNode expectedLogits = logitsGolden.get(caseIndex++);
        assertEquals(input.required("target_ids").size(), expectedLogits.size());
        List<Map<String, Object>> messages = new ArrayList<>();
        ObjectMapper mapper = new ObjectMapper();
        for (JsonNode message : input.required("messages")) {
          @SuppressWarnings("unchecked")
          Map<String, Object> converted = mapper.convertValue(message, Map.class);
          messages.add(converted);
        }
        List<RgbImage> selected = new ArrayList<>();
        for (JsonNode index : input.required("image_indices")) {
          selected.add(images.get(index.asInt()));
        }
        assertEquals(
            input.required("rendered_prompt").asString(),
            tokenizer.renderChat(messages, ChatTemplateOptions.defaults(true)));
        GenerationRequest request =
            GenerationRequest.chat(
                    tokenizer,
                    messages,
                    ChatTemplateOptions.defaults(true),
                    GenerationConfig.greedyDefaults(
                        input.required("max_new_tokens").asInt(),
                        Set.of(tokenizer.eosTokenId().orElseThrow())),
                    CancellationToken.NONE)
                .withImages(selected);
        assertEquals(
            TierBReferences.ids(input.required("unexpanded_ids")),
            Arrays.stream(request.promptTokenIds()).boxed().toList());
        SmolVlmPromptPlan plan = model.plan(request.promptTokenIds(), request.images());
        assertEquals(
            TierBReferences.ids(input.required("expanded_ids")),
            Arrays.stream(plan.expandedIds()).boxed().toList());
        try (MLXScope state = root.newChild()) {
          List<KVCache> caches = new ArrayList<>();
          for (int i = 0; i < model.decoderForTest().config().numHiddenLayers(); i++) {
            caches.add(new KVCache(state));
          }
          for (int step = 0; step < expectedLogits.size(); step++) {
            try (MLXScope activation = state.newChild()) {
              MLXArray logits;
              if (step == 0) {
                MLXArray features = model.features(request, activation);
                logits =
                    model
                        .decoderForTest()
                        .stepLogits(
                            MLX.array(
                                activation,
                                plan.expandedIds(),
                                new int[] {1, plan.expandedLength()}),
                            caches,
                            new int[] {plan.expandedLength()},
                            embedded -> SmolVlmModel.merge(embedded, features, plan));
              } else {
                int token = input.required("target_ids").get(step - 1).asInt();
                logits =
                    model
                        .decoderForTest()
                        .stepLogits(
                            MLX.array(activation, new int[] {token}, new int[] {1, 1}),
                            caches,
                            new int[] {1},
                            null);
              }
              float[] actual = logits.toFloatArray();
              reference.compare(expectedLogits.get(step), actual);
              JsonNode expected = expectedLogits.get(step);
              int best = 0;
              int runnerUp = 1;
              if (expected.get(runnerUp).asDouble() > expected.get(best).asDouble()) {
                best = 1;
                runnerUp = 0;
              }
              for (int i = 2; i < expected.size(); i++) {
                if (expected.get(i).asDouble() > expected.get(best).asDouble()) {
                  runnerUp = best;
                  best = i;
                } else if (expected.get(i).asDouble() > expected.get(runnerUp).asDouble()) {
                  runnerUp = i;
                }
              }
              double gap = expected.get(best).asDouble() - expected.get(runnerUp).asDouble();
              assertEquals(input.required("gaps").get(step).asDouble(), gap, 1e-6);
              assertEquals(
                  List.of(best, runnerUp),
                  TierBReferences.ids(input.required("top_two_ids").get(step)));
              reference.stable(gap);
              int winner = 0;
              for (int i = 1; i < actual.length; i++) {
                if (actual[i] > actual[winner]) {
                  winner = i;
                }
              }
              assertEquals(input.required("target_ids").get(step).asInt(), winner);
              MLX.eval(DecoderModel.stepCacheArrays(caches));
            }
          }
        }
        GenerationResult result = model.generate(request, ignored -> {});
        assertEquals(TierBReferences.ids(input.required("target_ids")), result.generatedTokenIds());
        assertEquals(
            TierBReferences.ids(input.required("unexpanded_ids")), result.promptTokenIds());
        assertEquals(plan.expandedLength(), result.promptPositions());
        assertEquals(
            FinishReason.valueOf(input.required("finish_reason").asString()),
            result.finishReason());
      }
      reference.report();
      System.out.println("Tier B sampled_peak_rss_kib=" + rss.maximum());
    }
  }

  /** Samples JVM RSS while inference runs; this thread never calls MLX. */
  private static final class RssSampler implements AutoCloseable {
    private volatile boolean running = true;
    private volatile long maximum;
    private volatile Exception failure;
    private final Thread worker;

    RssSampler() {
      worker =
          new Thread(
              () -> {
                try {
                  while (running) {
                    sample();
                    Thread.sleep(20);
                  }
                } catch (Exception exception) {
                  failure = exception;
                }
              },
              "tier-b-rss-sampler");
      worker.start();
    }

    long maximum() throws java.io.IOException, InterruptedException {
      sample();
      return maximum;
    }

    private synchronized void sample() throws java.io.IOException, InterruptedException {
      Process process =
          new ProcessBuilder("ps", "-o", "rss=", "-p", Long.toString(ProcessHandle.current().pid()))
              .start();
      long rss =
          Long.parseLong(
              new String(
                      process.getInputStream().readAllBytes(),
                      java.nio.charset.StandardCharsets.UTF_8)
                  .trim());
      if (process.waitFor() != 0) {
        throw new IllegalStateException("RSS sampling failed");
      }
      maximum = Math.max(maximum, rss);
    }

    @Override
    public void close() throws InterruptedException {
      running = false;
      worker.join();
      assertEquals(null, failure, "RSS sampler failure");
      assertTrue(maximum > 0);
    }
  }
}
