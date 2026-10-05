package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLXMemory;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

@EnabledIfNativeAvailable
class Seq2SeqMemoryTest {
  @Test
  void tokenizerFailureReleasesRequestState(
      @org.junit.jupiter.api.io.TempDir java.nio.file.Path directory) throws Exception {
    java.nio.file.Path file = directory.resolve("tokenizer.json");
    java.nio.file.Files.writeString(
        file,
        """
        {"version":"1.0","truncation":null,"padding":null,"added_tokens":[],
         "normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false},
         "post_processor":null,"decoder":{"type":"ByteLevel"},
         "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
                  "vocab":{"p":5,"q":7,"r":9,"e":1,"z":31},"merges":[]}}
        """);
    var tokenizer = se.alipsa.jmlx.tokenizer.HfTokenizer.fromFile(file);
    try (MLXScope scope = new MLXScope()) {
      T5Model model = T5Model.load(scope, Seq2SeqGenerationTest.checkpoint());
      model.generate(Seq2SeqGenerationTest.request(3), ignored -> {});
      final long baseline = MLXMemory.activeBytes();
      var request =
          GenerationRequest.text(
              tokenizer,
              "pqre",
              PromptSpecialTokens.OMIT,
              GenerationConfig.greedyDefaults(3, Set.of()),
              CancellationToken.NONE);
      for (int i = 0; i < 10; i++) {
        var failure =
            assertThrows(
                GenerationAbortedException.class, () -> model.generate(request, ignored -> {}));
        org.junit.jupiter.api.Assertions.assertEquals("output decoder", failure.stage());
        org.junit.jupiter.api.Assertions.assertEquals(List.of(17), failure.generatedTokenIds());
        assertTrue(MLXMemory.activeBytes() <= baseline + 65536);
      }
    }
  }

  @Test
  void requestsAndFailuresReturnToWarmBaseline() throws Exception {
    try (MLXScope scope = new MLXScope()) {
      T5Model model = T5Model.load(scope, Seq2SeqGenerationTest.checkpoint());
      for (int i = 0; i < 4; i++) {
        model.generate(Seq2SeqGenerationTest.request(8), ignored -> {});
      }
      final long baseline = MLXMemory.activeBytes();
      List<Long> samples = new ArrayList<>();
      for (int i = 0; i < 20; i++) {
        int[] source = i % 2 == 0 ? new int[] {5, 7, 9, 1} : new int[] {5, 7, 9, 11, 13, 15, 17, 1};
        GenerationRequest request =
            new GenerationRequest(
                source, GenerationConfig.greedyDefaults(8, Set.of()), CancellationToken.NONE);
        model.generate(request, ignored -> {});
        model.generate(request.withCachePolicy(GenerationCachePolicy.full(8)), ignored -> {});
        model.generate(Seq2SeqGenerationTest.request(0), ignored -> {});
        model.generate(new GenerationRequest(source, request.config(), () -> true), ignored -> {});
        assertThrows(
            GenerationAbortedException.class,
            () ->
                model.generate(
                    request,
                    event -> {
                      if (event.tokenId() != null) {
                        throw new IllegalStateException("listener");
                      }
                    }));
        samples.add(MLXMemory.activeBytes());
      }
      model.projectionObserver(
          (layer, cache) -> {
            throw new se.alipsa.jmlx.core.MLXException("injected projection boundary failure");
          });
      assertThrows(
          se.alipsa.jmlx.core.MLXException.class,
          () -> model.generate(Seq2SeqGenerationTest.request(3), ignored -> {}));
      model.projectionObserver((layer, cache) -> {});
      samples.add(MLXMemory.activeBytes());
      long maximum = samples.stream().mapToLong(Long::longValue).max().orElseThrow();
      long growth = samples.getLast() - samples.getFirst();
      // Tiny cache: 2 layers * 4 heads * d_kv=3 * K/V * float32 = 192 bytes/position.
      // Warm fixed margin is 64 KiB; repeated request scopes must have no positive retention slope.
      assertTrue(
          maximum <= baseline + 65536,
          "request retention above warm baseline: " + (maximum - baseline));
      assertTrue(growth <= 4096, "late request growth: " + growth);
      System.out.println(
          "seq2seq memory: baseline=" + baseline + ", max=" + maximum + ", growth=" + growth);
    }
  }

  @Test
  void sourceAndTargetWidthsHaveDerivedPeakBudget() throws Exception {
    try (MLXScope scope = new MLXScope()) {
      T5Model model = T5Model.load(scope, Seq2SeqGenerationTest.checkpoint());
      model.generate(Seq2SeqGenerationTest.request(4), ignored -> {});
      final long baseline = MLXMemory.activeBytes();
      for (int[] widths : List.of(new int[] {32, 1}, new int[] {4, 32}, new int[] {32, 32})) {
        int[] source = new int[widths[0]];
        java.util.Arrays.fill(source, 5);
        MLXMemory.resetPeak();
        model.generate(
            new GenerationRequest(
                    source,
                    GenerationConfig.greedyDefaults(widths[1], Set.of()),
                    CancellationToken.NONE)
                .withCachePolicy(GenerationCachePolicy.full(widths[1])),
            ignored -> {});
        long retained = 192L * (widths[0] + widths[1]) + widths[0] * 16L * 4;
        long peak = MLXMemory.peakBytes() - baseline;
        // Measured peak delta 302464 bytes at retained 8384 bytes for source 32/target 1.
        // Thirty-two times retained cache/output covers attention/FFN intermediates here;
        // 64 KiB is the independent fixed warm allocator margin for this tiny checkpoint.
        assertTrue(peak <= 32 * retained + 65536, "peak=" + peak + ", retained=" + retained);
        assertTrue(MLXMemory.activeBytes() <= baseline + 65536);
        System.out.println(
            "seq2seq source="
                + widths[0]
                + ", target="
                + widths[1]
                + ", retained="
                + retained
                + ", peak_delta="
                + peak);
      }
    }
  }
}
