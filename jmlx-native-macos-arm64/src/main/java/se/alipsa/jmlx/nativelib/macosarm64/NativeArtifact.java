package se.alipsa.jmlx.nativelib.macosarm64;

import java.io.IOException;
import java.io.InputStream;
import java.util.Objects;
import java.util.Properties;

/**
 * Metadata for the macOS/Apple-Silicon native-runtime artifact.
 *
 * <p>This class does not load MLX. Its constants describe the classpath resources packaged beside
 * it, and {@link #pin()} reads the native versions that produced those resources. Native loading is
 * intentionally owned by {@code jmlx-ffi}, so applications can depend on this artifact only at
 * runtime.
 */
public final class NativeArtifact {

  /** Artifact platform directory below {@link #RESOURCE_ROOT}. */
  public static final String PLATFORM = "macos-aarch64";

  /** Classpath directory containing the native dylibs, metallib, and pin metadata. */
  public static final String RESOURCE_ROOT = "se/alipsa/jmlx/native/" + PLATFORM;

  /** Resource name containing the native build pins. */
  public static final String PIN_RESOURCE = RESOURCE_ROOT + "/native-pin.properties";

  private NativeArtifact() {}

  /**
   * Reads the MLX and mlx-c revisions used to build this artifact.
   *
   * @return the independent native build pins packaged by this artifact
   * @throws IllegalStateException if this local development jar was built without staged native
   *     resources, or its packaged pin metadata is malformed
   */
  public static NativePin pin() {
    try (InputStream input =
        NativeArtifact.class.getClassLoader().getResourceAsStream(PIN_RESOURCE)) {
      if (input == null) {
        throw new IllegalStateException("native pin resource not found: " + PIN_RESOURCE);
      }
      Properties properties = new Properties();
      properties.load(input);
      return new NativePin(
          requiredProperty(properties, "mlxMetalVersion"),
          requiredProperty(properties, "mlxcCommit"));
    } catch (IOException e) {
      throw new IllegalStateException("failed to read native pin resource: " + PIN_RESOURCE, e);
    }
  }

  private static String requiredProperty(Properties properties, String key) {
    String value = properties.getProperty(key);
    if (value == null || value.isBlank()) {
      throw new IllegalStateException("native pin resource is missing a non-blank " + key);
    }
    return value;
  }

  /**
   * The independent MLX wheel version and mlx-c source revision packaged by this artifact.
   *
   * @param mlxMetalVersion packaged MLX wheel version
   * @param mlxcCommit packaged mlx-c source commit
   */
  public record NativePin(String mlxMetalVersion, String mlxcCommit) {
    /** Validates that both packaged pin values are present. */
    public NativePin {
      Objects.requireNonNull(mlxMetalVersion, "mlxMetalVersion");
      Objects.requireNonNull(mlxcCommit, "mlxcCommit");
    }
  }
}
