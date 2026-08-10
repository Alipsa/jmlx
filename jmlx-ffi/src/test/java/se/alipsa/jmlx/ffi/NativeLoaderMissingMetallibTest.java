package se.alipsa.jmlx.ffi;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/**
 * See req/initial-plan.md, Testing approach, "Loader guard".
 *
 * <p>
 * Runs ONLY via the {@code :jmlx-ffi:loaderGuardTest} Gradle task, which forks its own JVM with
 * {@code jmlx.library.path} pointed at a disposable copy of {@code native/install/lib} with {@code mlx.metallib}
 * removed -- see that task in {@code jmlx-ffi/build.gradle}. Excluded from the default {@code test} task:
 * {@code NativeLoader.ensureLoaded()} caches its outcome, so if this ran in the same JVM as
 * {@code NativeLoaderSmokeTest} (in either order), the result would depend on which test happened to run first rather
 * than on whether the metallib is actually present. This class also never mutates the real staging directory -- it only
 * ever reads whatever directory {@code jmlx.library.path} points it at.
 */
class NativeLoaderMissingMetallibTest {

  @Test
  void missingMetallibFailsFastNamingTheBootstrapScript() {
    IllegalStateException e = assertThrows(IllegalStateException.class, NativeLoader::ensureLoaded);
    assertTrue(e.getMessage().contains("bootstrap-native.sh"), e.getMessage());
  }
}
