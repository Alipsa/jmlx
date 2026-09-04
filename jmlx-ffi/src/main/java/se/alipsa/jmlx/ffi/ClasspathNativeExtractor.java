package se.alipsa.jmlx.ffi;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.channels.OverlappingFileLockException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.time.Instant;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.stream.Stream;

/**
 * Extracts the bundled MLX/mlx-c binaries from the {@code se.alipsa:jmlx-native-macos-arm64}
 * artifact's classpath resources to a directory on disk, for {@link NativeLoader}'s fallback when
 * neither {@code jmlx.library.path} nor {@code JMLX_LIBRARY_PATH} is set.
 *
 * <p>Extraction is cached on disk, keyed by a hash of the packaged {@code native-pin.properties}
 * plus each binary's size (see {@link #cacheDescriptorFor}), so a JVM only pays the ~180MB
 * extraction cost once per pin rather than on every process start. Entries are retained rather than
 * automatically evicted: another running JVM can still lazily open an older pin's metallib.
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

  private static final String NATIVE_ARTIFACT_CLASS_RESOURCE =
      "se/alipsa/jmlx/nativelib/macosarm64/NativeArtifact.class";

  /** A leftover {@code .tmp-*} sibling older than this is assumed abandoned, not in-progress. */
  private static final Duration STALE_TMP_AGE = Duration.ofHours(1);

  /** A peer JVM that never releases the cache lock must not hang native loading forever. */
  private static final Duration LOCK_TIMEOUT = Duration.ofMinutes(5);

  /** Prevents overlapping cache-wide file locks from concurrent extraction calls in this JVM. */
  private static final Object EXTRACTION_LOCK = new Object();

  /** Only macOS/Apple Silicon has a published native artifact today. */
  static boolean isSupportedPlatform(String osName, String osArch) {
    return "Mac OS X".equals(osName) && "aarch64".equals(osArch);
  }

  static boolean isSupportedPlatform() {
    return isSupportedPlatform(System.getProperty("os.name"), System.getProperty("os.arch"));
  }

  /**
   * Extracts the bundled binaries to the default per-pin cache directory under {@code
   * ~/Library/Application Support/se.alipsa.jmlx/native} (overridable via the {@code
   * jmlx.native.cache.path} system property), or an empty {@link Optional} if {@code
   * jmlx-native-macos-arm64} isn't on the classpath at all. This is durable application data rather
   * than {@code ~/Library/Caches}: MLX may not open {@code mlx.metallib} until the first kernel
   * launch, after the dylibs have already been loaded.
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
      if (RESOURCE_ROOT.equals(resourceRoot)
          && loader.getResource(NATIVE_ARTIFACT_CLASS_RESOURCE) != null) {
        throw new NativeExtractionException(
            "jmlx-native-macos-arm64 is on the classpath but contains no native payload. "
                + "Re-run scripts/bootstrap-native.sh before building or publishing that artifact, "
                + "or bypass it with -Djmlx.library.path=<directory> or "
                + "JMLX_LIBRARY_PATH=<directory>",
            new IOException("native artifact has no " + RESOURCE_ROOT + "/mlx.metallib resource"));
      }
      return Optional.empty(); // the native jar simply isn't a dependency
    }
    CacheDescriptor cacheDescriptor;
    try {
      cacheDescriptor = cacheDescriptorFor(loader, resourceRoot);
    } catch (IOException e) {
      throw new NativeExtractionException(
          "bundled native artifact is incomplete or unreadable on the classpath", e);
    }
    Path target = cacheRoot.resolve(cacheDescriptor.key());
    if (isComplete(target, cacheDescriptor.expectedSizes())) {
      return Optional.of(target);
    }
    try {
      return Optional.of(
          extractAtomically(loader, resourceRoot, cacheRoot, target, cacheDescriptor));
    } catch (IOException e) {
      throw new NativeExtractionException(
          "failed to extract bundled native libraries from the classpath. Configure a writable "
              + "cache with -Djmlx.native.cache.path=<directory>, or bypass extraction with "
              + "-Djmlx.library.path=<directory> or JMLX_LIBRARY_PATH=<directory>",
          e);
    } catch (OverlappingFileLockException e) {
      throw new NativeExtractionException(
          "native extraction cache is already locked in this JVM. Configure a separate writable "
              + "cache with -Djmlx.native.cache.path=<directory>, or bypass extraction with "
              + "-Djmlx.library.path=<directory> or JMLX_LIBRARY_PATH=<directory>",
          new IOException("overlapping lock for " + cacheRoot.resolve(".extraction.lock"), e));
    }
  }

  private static Path defaultCacheRoot() {
    String override = System.getProperty("jmlx.native.cache.path");
    if (override != null && !override.isBlank()) {
      return Path.of(override);
    }
    return Path.of(
        System.getProperty("user.home"),
        "Library",
        "Application Support",
        "se.alipsa.jmlx",
        "native");
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
  private static CacheDescriptor cacheDescriptorFor(ClassLoader loader, String resourceRoot)
      throws IOException {
    MessageDigest digest = sha256();
    Map<String, Long> expectedSizes = new LinkedHashMap<>();
    try (InputStream in = loader.getResourceAsStream(resourceRoot + "/" + PIN_FILE)) {
      if (in == null) {
        throw new IOException(PIN_FILE + " not found on classpath under " + resourceRoot);
      }
      byte[] pin = in.readAllBytes();
      if (pin.length == 0) {
        throw new IOException(PIN_FILE + " is empty on the classpath under " + resourceRoot);
      }
      digest.update(pin);
      expectedSizes.put(PIN_FILE, (long) pin.length);
    }
    for (String name : BINARY_FILES) {
      URL resource = loader.getResource(resourceRoot + "/" + name);
      if (resource == null) {
        throw new IOException("resource not found on classpath: " + resourceRoot + "/" + name);
      }
      long size = resourceSize(resource);
      if (size <= 0) {
        throw new IOException("classpath resource is missing or empty: " + resource);
      }
      digest.update(ByteBuffer.allocate(Long.BYTES).putLong(size).array());
      expectedSizes.put(name, size);
    }
    return new CacheDescriptor(
        HexFormat.of().formatHex(digest.digest()), Map.copyOf(expectedSizes));
  }

  /**
   * Finds a resource size without opening and retaining a {@code file:} URL connection. If another
   * URL handler cannot report a content length, count its stream as a compatibility fallback for
   * nested jars and custom classpaths.
   */
  private static long resourceSize(URL resource) throws IOException {
    if ("file".equals(resource.getProtocol())) {
      try {
        return Files.size(Path.of(resource.toURI()));
      } catch (URISyntaxException e) {
        throw new IOException("invalid file resource URL: " + resource, e);
      }
    }
    long reportedSize = resource.openConnection().getContentLengthLong();
    if (reportedSize >= 0) {
      return reportedSize;
    }
    try (InputStream in = resource.openStream()) {
      long size = 0;
      byte[] buffer = new byte[8192];
      for (int read; (read = in.read(buffer)) != -1; ) {
        size += read;
      }
      return size;
    }
  }

  private static MessageDigest sha256() {
    try {
      return MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException("SHA-256 not available", e);
    }
  }

  private static boolean isComplete(Path dir, Map<String, Long> expectedSizes) {
    return Files.isDirectory(dir, LinkOption.NOFOLLOW_LINKS)
        && expectedSizes.entrySet().stream()
            .allMatch(entry -> hasExpectedSize(dir.resolve(entry.getKey()), entry.getValue()));
  }

  private static boolean hasExpectedSize(Path path, long expectedSize) {
    try {
      return expectedSize > 0 && Files.isRegularFile(path) && Files.size(path) == expectedSize;
    } catch (IOException e) {
      return false;
    }
  }

  /**
   * Acquires a cache-wide cross-process lock, rechecks the target under that lock, then copies
   * every bundled file into a sibling temp directory (same filesystem as {@code cacheRoot},
   * required for the rename below to actually be atomic). This also serializes stale-temp cleanup:
   * a temp name does not encode its pin, so a per-pin lock could not protect a slow extraction of
   * another pin.
   */
  private static Path extractAtomically(
      ClassLoader loader,
      String resourceRoot,
      Path cacheRoot,
      Path target,
      CacheDescriptor cacheDescriptor)
      throws IOException {
    Files.createDirectories(cacheRoot);
    Path lockFile = cacheRoot.resolve(".extraction.lock");
    synchronized (EXTRACTION_LOCK) {
      // POSIX closes all fcntl locks this process owns for a file when any descriptor for that file
      // closes. Opening and closing the channel inside this monitor prevents one local caller from
      // accidentally releasing another local caller's lock after it has acquired it.
      try (FileChannel channel =
          FileChannel.open(lockFile, StandardOpenOption.CREATE, StandardOpenOption.WRITE)) {
        try (FileLock ignored = acquireLock(channel, lockFile)) {
          sweepStaleTempDirs(cacheRoot);
          if (isComplete(target, cacheDescriptor.expectedSizes())) {
            return target;
          }
          Path tmp = cacheRoot.resolve(".tmp-" + UUID.randomUUID());
          Files.createDirectories(tmp);
          try {
            for (String name : BINARY_FILES) {
              copyResource(loader, resourceRoot + "/" + name, tmp.resolve(name));
            }
            copyResource(loader, resourceRoot + "/" + PIN_FILE, tmp.resolve(PIN_FILE));
            moveIntoPlaceLocked(tmp, target, cacheDescriptor.expectedSizes());
          } catch (IOException e) {
            // Best-effort: a failed cleanup of our own temp copy must never mask the real
            // extraction failure being reported below.
            deleteQuietly(tmp);
            throw e;
          }
        }
      }
    }
    return target;
  }

  private static FileLock acquireLock(FileChannel channel, Path lockFile) throws IOException {
    Instant deadline = Instant.now().plus(LOCK_TIMEOUT);
    while (Instant.now().isBefore(deadline)) {
      FileLock lock = channel.tryLock();
      if (lock != null) {
        return lock;
      }
      try {
        Thread.sleep(100);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new IOException(
            "interrupted while waiting for native extraction lock: " + lockFile, e);
      }
    }
    throw new IOException("timed out waiting for native extraction lock: " + lockFile);
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
   * that was lost. Its files are repaired in place from {@code tmp}, preserving the directory path
   * so a JVM that already loaded its dylibs can still lazily open its metallib.
   */
  private static void moveIntoPlaceLocked(Path tmp, Path target, Map<String, Long> expectedSizes)
      throws IOException {
    try {
      Files.move(tmp, target, StandardCopyOption.ATOMIC_MOVE);
    } catch (IOException first) {
      if (isComplete(target, expectedSizes)) {
        deleteQuietly(tmp);
        return;
      }
      if (!Files.isDirectory(target, LinkOption.NOFOLLOW_LINKS)) {
        // A stale file or symlink at the cache-key path cannot be repaired in place. We hold the
        // cache-wide lock, so remove it and retry the move just as we repair incomplete folders.
        try {
          Files.deleteIfExists(target);
          Files.move(tmp, target, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException retry) {
          first.addSuppressed(retry);
          throw first;
        }
        return;
      }
      repairInPlace(tmp, target);
      deleteQuietly(tmp);
    }
  }

  /** Restores an incomplete cache entry without removing its directory from a live JVM's path. */
  private static void repairInPlace(Path tmp, Path target) throws IOException {
    sweepStaleRepairFiles(target);
    try (Stream<Path> files = Files.list(tmp)) {
      for (Path source : files.toList()) {
        Path destination = target.resolve(source.getFileName());
        Path staged = target.resolve(".tmp-repair-" + UUID.randomUUID());
        try {
          Files.copy(source, staged);
          Files.move(
              staged,
              destination,
              StandardCopyOption.ATOMIC_MOVE,
              StandardCopyOption.REPLACE_EXISTING);
        } finally {
          Files.deleteIfExists(staged);
        }
      }
    }
  }

  /** Removes abandoned same-directory repair staging files while the cache-wide lock is held. */
  private static void sweepStaleRepairFiles(Path target) throws IOException {
    try (Stream<Path> entries = Files.list(target)) {
      for (Path entry :
          entries.filter(p -> p.getFileName().toString().startsWith(".tmp-repair-")).toList()) {
        deleteQuietly(entry);
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

  /** Identifies a classpath-native-artifact failure for {@link NativeLoader}'s cached cause. */
  public static final class NativeExtractionException extends IllegalStateException {
    NativeExtractionException(String message, IOException cause) {
      super(message, cause);
    }
  }

  private record CacheDescriptor(String key, Map<String, Long> expectedSizes) {}
}
