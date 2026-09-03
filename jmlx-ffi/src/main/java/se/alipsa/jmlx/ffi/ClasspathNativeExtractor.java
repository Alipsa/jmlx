package se.alipsa.jmlx.ffi;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.time.Instant;
import java.util.Comparator;
import java.util.HexFormat;
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
 * plus each binary's size (see {@link #cacheKeyFor}), so a JVM only pays the ~180MB extraction cost
 * once per pin rather than on every process start.
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

  /** A leftover {@code .tmp-*} sibling older than this is assumed abandoned, not in-progress. */
  private static final Duration STALE_TMP_AGE = Duration.ofHours(1);

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

  /**
   * Hashes {@code native-pin.properties} together with each binary's byte size. Mixing in sizes
   * (cheap: {@link URL#openConnection()} reports a jar entry's or file's length without reading its
   * content) catches the case where a contributor rebuilds the binaries locally against an
   * unchanged pin (e.g. re-running {@code bootstrap-native.sh} against a modified mlx-c checkout
   * without bumping {@code native-pin.properties}) and republishes to {@code mavenLocal()} --
   * without this, such a rebuild would collide with a stale cache entry from before. This is
   * deliberately not a full content hash: reading and hashing the ~180MB of binaries on every JVM
   * start would defeat the point of caching them at all, and a same-size rebuild still collides --
   * an accepted trade-off given how rarely this scenario occurs outside active development of this
   * repo itself.
   */
  private static String cacheKeyFor(ClassLoader loader, String resourceRoot) throws IOException {
    MessageDigest digest = sha256();
    try (InputStream in = loader.getResourceAsStream(resourceRoot + "/" + PIN_FILE)) {
      if (in == null) {
        throw new IOException(PIN_FILE + " not found on classpath under " + resourceRoot);
      }
      digest.update(in.readAllBytes());
    }
    for (String name : BINARY_FILES) {
      URL resource = loader.getResource(resourceRoot + "/" + name);
      if (resource == null) {
        throw new IOException("resource not found on classpath: " + resourceRoot + "/" + name);
      }
      long size = resource.openConnection().getContentLengthLong();
      if (size < 0) {
        throw new IOException("could not determine size of classpath resource: " + resource);
      }
      digest.update(Long.toString(size).getBytes(StandardCharsets.UTF_8));
    }
    return HexFormat.of().formatHex(digest.digest());
  }

  private static MessageDigest sha256() {
    try {
      return MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException("SHA-256 not available", e);
    }
  }

  private static boolean isComplete(Path dir) {
    return Files.isDirectory(dir)
        && isNonEmptyFile(dir.resolve(PIN_FILE))
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
   * required for the rename below to actually be atomic), then hands off to {@link
   * #moveIntoPlace(Path, Path)}.
   */
  private static Path extractAtomically(
      ClassLoader loader, String resourceRoot, Path cacheRoot, Path target) throws IOException {
    Files.createDirectories(cacheRoot);
    sweepStaleTempDirs(cacheRoot);
    Path tmp = cacheRoot.resolve(".tmp-" + UUID.randomUUID());
    Files.createDirectories(tmp);
    try {
      for (String name : BINARY_FILES) {
        copyResource(loader, resourceRoot + "/" + name, tmp.resolve(name));
      }
      copyResource(loader, resourceRoot + "/" + PIN_FILE, tmp.resolve(PIN_FILE));
      moveIntoPlace(tmp, target);
    } catch (IOException e) {
      // Best-effort: a failed cleanup of our own temp copy must never mask the real extraction
      // failure being reported below.
      deleteQuietly(tmp);
      throw e;
    }
    return target;
  }

  /**
   * Renames {@code tmp} onto {@code target}. A losing rename -- another thread or process finished
   * extracting the same cache key first -- is reported by the platform as {@link
   * java.nio.file.FileAlreadyExistsException} on some filesystems and a directory-not-empty {@link
   * IOException} on others (renaming onto a non-empty directory is {@code ENOTEMPTY}, not {@code
   * EEXIST}, under POSIX rename() semantics); rather than pattern-match exception types across
   * platforms, any failed rename is first checked against {@link #isComplete}: a complete {@code
   * target} means a race was lost fairly (contents for a given cache key are always identical), so
   * the loser's temp copy is simply discarded. An *incomplete* target -- left over from an
   * interrupted extraction, a half-deleted cache-cleaner pass, or disk exhaustion -- is not a race
   * that was lost; without clearing it, every subsequent JVM start would hit the same rename
   * failure forever. It's cleared and the rename retried once before giving up.
   */
  private static void moveIntoPlace(Path tmp, Path target) throws IOException {
    try {
      Files.move(tmp, target, StandardCopyOption.ATOMIC_MOVE);
    } catch (IOException first) {
      if (isComplete(target)) {
        deleteQuietly(tmp);
        return;
      }
      deleteQuietly(target);
      try {
        Files.move(tmp, target, StandardCopyOption.ATOMIC_MOVE);
      } catch (IOException second) {
        if (isComplete(target)) {
          deleteQuietly(tmp);
          return;
        }
        throw second;
      }
    }
  }

  /**
   * A JVM killed mid-extraction (SIGKILL, OOM, power loss) leaves its {@code .tmp-*} directory
   * behind forever otherwise. Best-effort and age-gated: an in-progress sibling from a
   * concurrently-racing call must never be swept out from under it.
   */
  private static void sweepStaleTempDirs(Path cacheRoot) {
    try (Stream<Path> entries = Files.list(cacheRoot)) {
      entries
          .filter(p -> p.getFileName().toString().startsWith(".tmp-"))
          .filter(ClasspathNativeExtractor::isStale)
          .forEach(ClasspathNativeExtractor::deleteQuietly);
    } catch (IOException ignored) {
      // A failed sweep must never block extraction itself.
    }
  }

  private static boolean isStale(Path dir) {
    try {
      Instant lastModified = Files.getLastModifiedTime(dir).toInstant();
      return lastModified.isBefore(Instant.now().minus(STALE_TMP_AGE));
    } catch (IOException e) {
      return false;
    }
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

  /** A leftover temp or stale target directory is never worth failing a successful load over. */
  private static void deleteQuietly(Path dir) {
    try {
      deleteRecursively(dir);
    } catch (IOException ignored) {
      // best-effort, see caller-side comments
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
}
