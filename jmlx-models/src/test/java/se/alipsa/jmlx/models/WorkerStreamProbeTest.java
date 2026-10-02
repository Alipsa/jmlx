package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Gate 0 probe (a): MLX streams are thread-bound, so a worker thread must run on a stream it owns.
 * Before per-thread streams (each root {@code MLXScope} carries its owner thread's stream) this
 * failed with "There is no Stream(gpu, 0) in current thread" for every thread but the first to use
 * MLX, which also broke close-then-restart. Runs in its own forked JVM ({@code threadProbeTest}) so
 * the "main thread used MLX first" setup is genuinely first in the process.
 */
@EnabledIfNativeAvailable
class WorkerStreamProbeTest {
  @Test
  void workerThreadLoadsAndGeneratesAfterStreamInitializedElsewhere(@TempDir Path dir)
      throws Exception {
    TinyCheckpoints.randomLlama(dir, 1, 2, false, false);
    // Initialize the default stream on this (the "main") thread, and nothing else.
    try (MLXScope scope = new MLXScope()) {
      MLXArray probe = MLX.array(scope, new float[] {1f, 2f}, new int[] {2});
      MLX.eval(probe);
    }

    AtomicReference<List<Integer>> generated = new AtomicReference<>();
    AtomicReference<Throwable> failure = new AtomicReference<>();
    Thread worker =
        new Thread(
            () -> {
              try (MLXScope root = new MLXScope()) {
                DecoderModel model = LlamaModel.load(root, dir);
                generated.set(model.generate(new int[] {1, 2, 3}, 4, Set.of()));
              } catch (Throwable t) {
                failure.set(t);
              }
            },
            "worker-stream-probe");
    worker.start();
    worker.join();

    assertEquals(null, failure.get(), () -> "worker failed: " + failure.get());
    assertEquals(7, generated.get().size(), "prompt (3) plus 4 generated tokens");
    assertFalse(generated.get().isEmpty());
  }

  /** Close-then-restart shape: two successive workers, neither of them the first MLX thread. */
  @Test
  void successiveWorkersEachGetTheirOwnStream(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomLlama(dir, 1, 2, false, false);
    List<Integer> first = runWorker(dir, "worker-stream-probe-1");
    List<Integer> second = runWorker(dir, "worker-stream-probe-2");
    assertEquals(first, second, "same model and prompt must agree across workers");
    assertNotEquals(0, first.size());
  }

  private static List<Integer> runWorker(Path dir, String name) throws Exception {
    AtomicReference<List<Integer>> out = new AtomicReference<>();
    AtomicReference<Throwable> failure = new AtomicReference<>();
    Thread worker =
        new Thread(
            () -> {
              try (MLXScope root = new MLXScope()) {
                out.set(LlamaModel.load(root, dir).generate(new int[] {1, 2, 3}, 4, Set.of()));
              } catch (Throwable t) {
                failure.set(t);
              }
            },
            name);
    worker.start();
    worker.join();
    assertEquals(null, failure.get(), () -> name + " failed: " + failure.get());
    return out.get();
  }
}
