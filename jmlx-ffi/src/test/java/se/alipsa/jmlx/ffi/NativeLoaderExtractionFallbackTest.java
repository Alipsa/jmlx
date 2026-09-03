package se.alipsa.jmlx.ffi;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

/**
 * Runs ONLY via the {@code :jmlx-ffi:extractionFallbackTest} Gradle task, which forks its own JVM
 * with {@code jmlx.library.path} blanked, {@code JMLX_LIBRARY_PATH} removed, and the real {@code
 * se.alipsa:jmlx-native-macos-arm64} jar on its classpath (see that task in {@code
 * jmlx-ffi/build.gradle}) -- forcing {@link NativeLoader#ensureLoaded()} onto {@link
 * ClasspathNativeExtractor}'s fallback path. Excluded from the default {@code test} task for the
 * same reason {@code NativeLoaderMissingMetallibTest} is: {@code ensureLoaded()} caches its
 * outcome, so this must not share a JVM with a test that takes the explicit-path branch.
 *
 * <p>{@link ClasspathNativeExtractorTest} already covers the extraction mechanics with tiny fake
 * fixtures; this is the one place that proves the real ~180MB artifact's contents actually let MLX
 * load and evaluate a GPU kernel, which forces MLX to open the colocated {@code mlx.metallib}.
 */
class NativeLoaderExtractionFallbackTest {

  @Test
  void ensureLoadedSucceedsViaClasspathExtraction() throws Exception {
    assertEquals("", System.getProperty("jmlx.library.path"));
    Path cacheRoot = Path.of(System.getProperty("jmlx.native.cache.path"));
    NativeLoader.ensureLoaded();
    try (var entries = Files.list(cacheRoot)) {
      Path extracted = entries.filter(Files::isDirectory).findFirst().orElseThrow();
      assertTrue(Files.isRegularFile(extracted.resolve("libmlxc.dylib")));
      assertTrue(Files.isRegularFile(extracted.resolve("mlx.metallib")));
      assertTrue(Files.isRegularFile(extracted.resolve("native-pin.properties")));
    }
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment dev = mlx_h.mlx_device_new(arena);
      assertEquals(0, mlx_h.mlx_get_default_device(dev));

      MemorySegment typeOut = arena.allocate(ValueLayout.JAVA_INT);
      assertEquals(0, mlx_h.mlx_device_get_type(typeOut, dev));
      int type = typeOut.get(ValueLayout.JAVA_INT, 0);
      assertTrue(type == 0 || type == 1, "unexpected device type: " + type);

      assertEquals(0, mlx_h.mlx_device_free(dev));

      // A device query does not open mlx.metallib. Force MLX onto the GPU and evaluate a genuine
      // kernel so this fallback test proves the extracted flat layout includes the metallib MLX
      // locates lazily beside libmlx.dylib.
      MemorySegment gpu = mlx_h.mlx_device_new_type(arena, mlx_h.MLX_GPU(), 0);
      assertEquals(0, mlx_h.mlx_set_default_device(gpu));
      MemorySegment stream = mlx_h.mlx_default_gpu_stream_new(arena);
      MemorySegment left = mlx_h.mlx_array_new_float32(arena, 1.0f);
      MemorySegment right = mlx_h.mlx_array_new_float32(arena, 2.0f);
      MemorySegment sum = mlx_h.mlx_array_new(arena);
      assertEquals(0, mlx_h.mlx_add(sum, left, right, stream));
      assertEquals(0, mlx_h.mlx_array_eval(sum));
      assertEquals(0, mlx_h.mlx_array_free(sum));
      assertEquals(0, mlx_h.mlx_array_free(right));
      assertEquals(0, mlx_h.mlx_array_free(left));
    }
  }
}
