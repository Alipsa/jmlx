package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Gate 0 probe (b), an <em>evidence</em> probe that never gates: two threads each load a model and
 * generate concurrently on the shared default stream. A pass proves nothing, since a race can pass
 * here and crash later, so "at most one MLX-using thread at a time per process" stays the
 * documented rule. It can crash its JVM, so it runs in its own forked task ({@code
 * concurrentStreamProbe}) outside {@code check}.
 */
@EnabledIfNativeAvailable
class ConcurrentStreamProbeTest {
  @Test
  void twoThreadsGenerateConcurrently(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomLlama(dir, 1, 2, false, false);
    CountDownLatch go = new CountDownLatch(1);
    AtomicReference<List<Integer>> first = new AtomicReference<>();
    AtomicReference<List<Integer>> second = new AtomicReference<>();
    AtomicReference<Throwable> failure = new AtomicReference<>();
    Thread a = racer(dir, go, first, failure, "probe-a");
    Thread b = racer(dir, go, second, failure, "probe-b");
    a.start();
    b.start();
    go.countDown();
    a.join();
    b.join();
    assertEquals(null, failure.get(), () -> "a racing thread failed: " + failure.get());
    assertEquals(first.get(), second.get(), "same model and prompt must agree");
  }

  private static Thread racer(
      Path dir,
      CountDownLatch go,
      AtomicReference<List<Integer>> out,
      AtomicReference<Throwable> failure,
      String name) {
    return new Thread(
        () -> {
          try (MLXScope root = new MLXScope()) {
            DecoderModel model = LlamaModel.load(root, dir);
            go.await();
            out.set(model.generate(new int[] {1, 2, 3}, 16, Set.of()));
          } catch (Throwable t) {
            failure.compareAndSet(null, t);
          }
        },
        name);
  }
}
