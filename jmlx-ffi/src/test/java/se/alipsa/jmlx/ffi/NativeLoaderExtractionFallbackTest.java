package se.alipsa.jmlx.ffi;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
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
 * load and answer a real downcall (same shape as {@link NativeLoaderSmokeTest}).
 */
class NativeLoaderExtractionFallbackTest {

  @Test
  void ensureLoadedSucceedsViaClasspathExtraction() {
    NativeLoader.ensureLoaded();
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment dev = mlx_h.mlx_device_new(arena);
      assertEquals(0, mlx_h.mlx_get_default_device(dev));

      MemorySegment typeOut = arena.allocate(ValueLayout.JAVA_INT);
      assertEquals(0, mlx_h.mlx_device_get_type(typeOut, dev));
      int type = typeOut.get(ValueLayout.JAVA_INT, 0);
      assertTrue(type == 0 || type == 1, "unexpected device type: " + type);

      assertEquals(0, mlx_h.mlx_device_free(dev));
    }
  }
}
