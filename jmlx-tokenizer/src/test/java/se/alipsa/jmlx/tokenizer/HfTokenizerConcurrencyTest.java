package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;

/**
 * One {@link HfTokenizer} shared by many threads gives every thread the single-threaded answer for
 * encoding, decoding, incremental decoding and chat rendering. The batch scheduler's dispatcher and
 * callers' submit threads rely on this.
 */
class HfTokenizerConcurrencyTest {
  private static final int WORKERS = 16;
  private static final int ROUNDS = 40;
  private static final List<String> TEXTS =
      List.of(
          "Hello, world!",
          "The quick brown fox",
          "  spaces\n",
          "Unicode: åäö 日本語 😀",
          "<|im_start|>user\nhi<|im_end|>",
          "");

  @TempDir Path directory;

  private HfTokenizer tokenizer() throws IOException {
    Files.copy(
        resource("qwen2.5-0.5b-instruct.tokenizer.json"), directory.resolve("tokenizer.json"));
    Files.copy(
        resource("qwen2.5-instruct-chat-template.jinja"), directory.resolve("chat_template.jinja"));
    return HfTokenizer.fromDirectory(directory);
  }

  private Path resource(String name) {
    try {
      return Path.of(getClass().getResource(name).toURI());
    } catch (java.net.URISyntaxException e) {
      throw new IllegalStateException(e);
    }
  }

  /** Everything one worker computes for a text, reduced to comparable values. */
  private record Outcome(
      List<Integer> ids, String decoded, String streamed, String chat, List<Integer> offsets) {}

  private static Outcome compute(HfTokenizer tokenizer, String text, int salt) {
    List<Integer> ids = tokenizer.encode(text, false);
    TokenizerEncoding encoding = tokenizer.encodeWithDefaults(text, false);
    IncrementalTokenDecoder decoder = tokenizer.newIncrementalDecoder(false);
    StringBuilder streamed = new StringBuilder();
    for (int id : ids) {
      streamed.append(decoder.append(id));
    }
    streamed.append(decoder.finish());
    String chat =
        tokenizer.renderChat(
            List.of(
                Map.of("role", "system", "content", "salt " + salt),
                Map.of("role", "user", "content", text)),
            new ChatTemplateOptions("", true, Map.of()));
    return new Outcome(
        ids,
        tokenizer.decode(ids, false),
        streamed.toString(),
        chat,
        encoding.offsets().stream().map(TokenOffset::startByte).toList());
  }

  @Test
  @Timeout(value = 60, unit = TimeUnit.SECONDS, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
  void sharedTokenizerMatchesSingleThreadedReferenceUnderContention() throws Exception {
    HfTokenizer tokenizer = tokenizer();
    List<Outcome> reference = new ArrayList<>();
    for (int salt = 0; salt < TEXTS.size(); salt++) {
      reference.add(compute(tokenizer, TEXTS.get(salt), salt));
    }
    ExecutorService pool = Executors.newFixedThreadPool(WORKERS);
    try {
      CountDownLatch start = new CountDownLatch(1);
      List<Future<?>> futures = new ArrayList<>();
      for (int worker = 0; worker < WORKERS; worker++) {
        final int offset = worker;
        futures.add(
            pool.submit(
                () -> {
                  start.await();
                  for (int round = 0; round < ROUNDS; round++) {
                    int index = (round + offset) % TEXTS.size();
                    assertEquals(
                        reference.get(index),
                        compute(tokenizer, TEXTS.get(index), index),
                        "worker " + offset + " round " + round);
                  }
                  return null;
                }));
      }
      start.countDown();
      for (Future<?> future : futures) {
        future.get(45, TimeUnit.SECONDS);
      }
    } finally {
      pool.shutdownNow();
    }
  }
}
