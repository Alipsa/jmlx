package se.alipsa.jmlx.ffi;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import java.util.stream.Stream;

/**
 * Extracts the bundled MLX/mlx-c binaries from the {@code se.alipsa:jmlx-native-macos-arm64}
 * artifact's classpath resources to a directory on disk, for {@link NativeLoader}'s fallback when
 * neither {@code jmlx.library.path} nor {@code JMLX_LIBRARY_PATH} is set.
 *
 * <p>Extraction is cached on disk, keyed by a hash of the packaged {@code native-pin.properties}
 * (the exact upstream mlx-metal/mlx-c pin baked into the jar), so a JVM only pays the ~180MB
 * extraction cost once per pin rather than on every process start.
 */
final class ClasspathNativeExtractor {

  private ClasspathNativeExtractor() {}

  /**
   * The 4 files {@link NativeLoader} actually needs, colocated -- dyld resolves them by directory.
   */
  private static final List<String> BINARY_FILES =
      List.of("libmlxc.dylib", "libmlx.dylib", "libjaccl.dylib", "mlx.metallib");

  private static final String PIN_FILE = "native-pin.properties";

  private static final String RESOURCE_ROOT = "se/alipsa/jmlx/native/macos-aarch64";

  /** Only macOS/Apple Silicon has a published native artifact today. */
  static boolean isSupportedPlatform(String osName, String osArch) {
    return "Mac OS X".equals(osName) && "aarch64".equals(osArch);
  }

  static boolean isSupportedPlatform() {
    return isSupportedPlatform(System.getProperty("os.name"), System.getProperty("os.arch"));
  }

  /**
   * Extracts the bundled binaries to the default per-pin cache directory under {@code
   * ~/Library/Caches/se.alipsa.jmlx/native} (overridable via the {@code jmlx.native.cache.path}
   * system property), or an empty {@link Optional} if {@code jmlx-native-macos-arm64} isn't on the
   * classpath at all.
   */
  static Optional<Path> extractIfAvailable() {
    return extractIfAvailable(
        ClasspathNativeExtractor.class.getClassLoader(), RESOURCE_ROOT, defaultCacheRoot());
  }

  /**
   * Test seam: extracts using an arbitrary {@code loader}/{@code resourceRoot}/{@code cacheRoot}
   * rather than the real classpath resources and the real (multi-hundred-MB) cache directory.
   */
  static Optional<Path> extractIfAvailable(
      ClassLoader loader, String resourceRoot, Path cacheRoot) {
    if (loader.getResource(resourceRoot + "/mlx.metallib") == null) {
      return Optional.empty(); // the native jar simply isn't a dependency
    }
    try {
      Path target = cacheRoot.resolve(cacheKeyFor(loader, resourceRoot));
      if (isComplete(target)) {
        return Optional.of(target);
      }
      return Optional.of(extractAtomically(loader, resourceRoot, cacheRoot, target));
    } catch (IOException e) {
      throw new IllegalStateException(
          "failed to extract bundled native libraries from the classpath", e);
    }
  }

  private static Path defaultCacheRoot() {
    String override = System.getProperty("jmlx.native.cache.path");
    if (override != null && !override.isBlank()) {
      return Path.of(override);
    }
    return Path.of(
        System.getProperty("user.home"), "Library", "Caches", "se.alipsa.jmlx", "native");
  }

  private static String cacheKeyFor(ClassLoader loader, String resourceRoot) throws IOException {
    try (InputStream in = loader.getResourceAsStream(resourceRoot + "/" + PIN_FILE)) {
      if (in == null) {
        throw new IOException(PIN_FILE + " not found on classpath under " + resourceRoot);
      }
      return sha256Hex(in.readAllBytes());
    }
  }

  private static boolean isComplete(Path dir) {
    return Files.isDirectory(dir)
        && BINARY_FILES.stream().allMatch(name -> isNonEmptyFile(dir.resolve(name)));
  }

  private static boolean isNonEmptyFile(Path path) {
    try {
      return Files.isRegularFile(path) && Files.size(path) > 0;
    } catch (IOException e) {
      return false;
    }
  }

  /**
   * Copies every bundled file into a sibling temp directory (same filesystem as {@code cacheRoot},
   * required for the rename below to actually be atomic), then atomically renames it to {@code
   * target}. A losing rename -- another thread or process finished extracting the same cache key
   * first -- is reported by the platform as {@link FileAlreadyExistsException} on some filesystems
   * and a directory-not-empty {@link IOException} on others (renaming onto a non-empty directory is
   * {@code ENOTEMPTY}, not {@code EEXIST}, under POSIX rename() semantics); rather than
   * pattern-match exception types across platforms, any failed rename is treated as a lost race --
   * and the loser's temp copy discarded -- as long as {@code target} is a complete extraction
   * afterward (contents for a given cache key are always identical), and rethrown otherwise.
   */
  private static Path extractAtomically(
      ClassLoader loader, String resourceRoot, Path cacheRoot, Path target) throws IOException {
    Files.createDirectories(cacheRoot);
    Path tmp = cacheRoot.resolve(".tmp-" + UUID.randomUUID());
    Files.createDirectories(tmp);
    try {
      for (String name : BINARY_FILES) {
        copyResource(loader, resourceRoot + "/" + name, tmp.resolve(name));
      }
      copyResource(loader, resourceRoot + "/" + PIN_FILE, tmp.resolve(PIN_FILE));
      try {
        Files.move(tmp, target, StandardCopyOption.ATOMIC_MOVE);
      } catch (IOException e) {
        if (!isComplete(target)) {
          throw e;
        }
        deleteRecursively(tmp);
      }
    } catch (IOException e) {
      deleteRecursively(tmp);
      throw e;
    }
    return target;
  }

  private static void copyResource(ClassLoader loader, String resourcePath, Path target)
      throws IOException {
    try (InputStream in = loader.getResourceAsStream(resourcePath)) {
      if (in == null) {
        throw new IOException("resource not found on classpath: " + resourcePath);
      }
      Files.copy(in, target);
    }
  }

  private static void deleteRecursively(Path dir) throws IOException {
    if (!Files.exists(dir)) {
      return;
    }
    try (Stream<Path> walk = Files.walk(dir)) {
      walk.sorted(Comparator.reverseOrder())
          .forEach(
              p -> {
                try {
                  Files.delete(p);
                } catch (IOException e) {
                  throw new UncheckedIOException(e);
                }
              });
    } catch (UncheckedIOException e) {
      throw e.getCause();
    }
  }

  private static String sha256Hex(byte[] bytes) {
    try {
      byte[] hash = MessageDigest.getInstance("SHA-256").digest(bytes);
      StringBuilder sb = new StringBuilder(hash.length * 2);
      for (byte b : hash) {
        sb.append(String.format("%02x", b));
      }
      return sb.toString();
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException("SHA-256 not available", e);
    }
  }
}
