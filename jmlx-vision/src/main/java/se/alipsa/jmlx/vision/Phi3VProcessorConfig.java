package se.alipsa.jmlx.vision;

import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/**
 * Validated Phi-3.5-Vision image processor configuration, parsed from a Hugging Face {@code
 * preprocessor_config.json}, for the bundled trust-remote-code {@code Phi3VImageProcessor}
 * preprocessing path that {@link Phi3VImageProcessor} ports (findings §6, {@code
 * req/plans/phase7-3c-reference-findings.md}).
 *
 * <p>Parsing is strict: the root must be a JSON object, every key must be a recognized processor
 * option, required options must be present with the expected type, and malformed values are
 * rejected with a descriptive {@link IllegalArgumentException} that names the offending key — the
 * same contract as {@link Qwen3VlProcessorConfig}, with the message prefix identifying this config
 * as the Phi-3.5-Vision one.
 *
 * <p>The reference {@code __init__} reads exactly {@code num_crops} (required here: the token
 * formula and the scale loop are defined in terms of it), {@code image_mean}, {@code image_std} and
 * {@code do_convert_rgb}; the legacy {@code num_img_tokens} key is accepted and recorded via {@link
 * #numImgTokens()} (the reference processor never reads it; the model cross-checks it against the
 * checkpoint's {@code img_processor} block when present). The decode policy is drop-alpha +
 * EXIF-ignore: the reference {@code convert("RGB")} step drops alpha, never composites, and the
 * processor never calls {@code exif_transpose} (findings §6), so a config with {@code
 * do_convert_rgb: false} is rejected at parse time because the ported decode path always delivers
 * RGB.
 */
public final class Phi3VProcessorConfig {
  private static final ObjectMapper MAPPER = new ObjectMapper();

  private static final Set<String> KNOWN_KEYS =
      Set.of(
          "auto_map",
          "do_convert_rgb",
          "image_mean",
          "image_processor_type",
          "image_std",
          "num_crops",
          "num_img_tokens",
          "processor_class");

  /**
   * The full-dump form of transformers' {@code preprocessor_config.json} can carry the image
   * processor's complete field set instead of the minimal effective subset. Every key in this set
   * is inert for the planned pipeline: each must be JSON {@code null} when present — a non-null
   * value would be an active processor setting this port does not implement, and is rejected rather
   * than ignored (the same contract as {@link Qwen3VlProcessorConfig}).
   */
  private static final Set<String> INERT_NULL_KEYS =
      Set.of(
          "crop_size",
          "device",
          "disable_grouping",
          "do_center_crop",
          "do_pad",
          "input_data_format",
          "pad_size",
          "return_tensors");

  private static final String[] REQUIRED_KEYS = {"image_mean", "image_std", "num_crops"};

  private final int numCrops;
  private final Integer numImgTokens;
  private final float[] imageMean;
  private final float[] imageStd;
  private final String processorClass;

  private Phi3VProcessorConfig(
      int numCrops,
      Integer numImgTokens,
      float[] imageMean,
      float[] imageStd,
      String processorClass) {
    // Shared semantic validation for both construction paths: the JSON path has already checked
    // types, the builder has not checked anything, so both reach the same guards here.
    if (numCrops <= 0) {
      throw new IllegalArgumentException("numCrops must be positive: " + numCrops);
    }
    if (imageMean == null || imageMean.length != 3) {
      throw new IllegalArgumentException("imageMean must be an array of exactly three values");
    }
    for (float m : imageMean) {
      if (!Float.isFinite(m)) {
        throw new IllegalArgumentException("imageMean values must be finite: " + m);
      }
    }
    if (imageStd == null || imageStd.length != 3) {
      throw new IllegalArgumentException("imageStd must be an array of exactly three values");
    }
    for (float s : imageStd) {
      if (!Float.isFinite(s) || s <= 0.0f) {
        throw new IllegalArgumentException("imageStd values must be finite and positive: " + s);
      }
    }
    this.numCrops = numCrops;
    this.numImgTokens = numImgTokens;
    this.imageMean = imageMean.clone();
    this.imageStd = imageStd.clone();
    this.processorClass = processorClass;
  }

  /**
   * Parses and validates a {@code preprocessor_config.json} document.
   *
   * @param json UTF-8 JSON bytes of the processor configuration
   * @return the validated configuration
   * @throws IllegalArgumentException if the document is not valid JSON, has unknown or missing
   *     keys, malformed values, a non-object {@code auto_map}, a wrong {@code
   *     image_processor_type}, or {@code do_convert_rgb: false}
   */
  public static Phi3VProcessorConfig fromJson(byte[] json) {
    if (json == null) {
      throw new IllegalArgumentException("json must not be null");
    }
    JsonNode root;
    try {
      root = MAPPER.readTree(json);
    } catch (JacksonException e) {
      throw new IllegalArgumentException(
          "phi3v preprocessor config is not valid JSON: " + e.getMessage(), e);
    }
    return fromNode(root);
  }

  private static Phi3VProcessorConfig fromNode(JsonNode root) {
    if (!root.isObject()) {
      throw new IllegalArgumentException("phi3v preprocessor config root must be a JSON object");
    }
    Set<String> keys = new LinkedHashSet<>();
    for (Map.Entry<String, JsonNode> entry : root.properties()) {
      keys.add(entry.getKey());
    }
    for (String key : keys) {
      if (!KNOWN_KEYS.contains(key)) {
        if (INERT_NULL_KEYS.contains(key)) {
          if (!root.path(key).isNull()) {
            throw new IllegalArgumentException(
                "phi3v preprocessor config full-dump key '"
                    + key
                    + "' must be null; a non-null value is an active processor setting this port"
                    + " does not implement");
          }
          continue;
        }
        throw new IllegalArgumentException(
            "phi3v preprocessor config has unrecognized key '" + key + "'");
      }
    }
    for (String key : REQUIRED_KEYS) {
      if (!keys.contains(key)) {
        throw new IllegalArgumentException(
            "phi3v preprocessor config is missing required key '" + key + "'");
      }
    }
    if (!root.path("image_processor_type").isString()
        || !"Phi3VImageProcessor".equals(root.path("image_processor_type").asString())) {
      throw new IllegalArgumentException(
          "phi3v preprocessor config image_processor_type must be 'Phi3VImageProcessor'");
    }
    JsonNode autoMap = root.path("auto_map");
    if (keys.contains("auto_map") && !autoMap.isNull() && !autoMap.isObject()) {
      throw new IllegalArgumentException("auto_map must be a JSON object");
    }
    final float[] imageMean = floatVector(root, "image_mean");
    final float[] imageStd = floatVector(root, "image_std");
    String processorClass = null;
    if (keys.contains("processor_class")) {
      JsonNode processorClassNode = root.path("processor_class");
      if (!processorClassNode.isString()) {
        throw new IllegalArgumentException("processor_class must be a JSON string");
      }
      processorClass = processorClassNode.asString();
    }
    boolean doConvertRgb = optionalBool(root, "do_convert_rgb", true);
    if (!doConvertRgb) {
      throw new IllegalArgumentException(
          "do_convert_rgb must be true: the ported decoder always converts decoded input to "
              + "RGB, so a config that skips the conversion would be silently ignored");
    }
    int numCrops = positiveInt(root, "num_crops");
    // The legacy per-image-token-count hint: accepted and recorded (the reference processor never
    // reads it from the image-processor config; the model cross-checks it against config.json).
    Integer numImgTokens =
        keys.contains("num_img_tokens") && !root.path("num_img_tokens").isNull()
            ? positiveInt(root, "num_img_tokens")
            : null;
    return new Phi3VProcessorConfig(numCrops, numImgTokens, imageMean, imageStd, processorClass);
  }

  private static int positiveInt(JsonNode root, String key) {
    JsonNode node = root.path(key);
    if (!node.isIntegralNumber()) {
      throw new IllegalArgumentException(key + " must be an integer");
    }
    if (!node.canConvertToInt()) {
      throw new IllegalArgumentException(
          key + " must fit in a signed 32-bit integer: " + node.asString());
    }
    int value = node.asInt();
    if (value <= 0) {
      throw new IllegalArgumentException(key + " must be positive: " + value);
    }
    return value;
  }

  private static boolean optionalBool(JsonNode root, String key, boolean defaultValue) {
    if (!root.has(key)) {
      return defaultValue;
    }
    JsonNode node = root.path(key);
    if (!node.isBoolean()) {
      throw new IllegalArgumentException(key + " must be a JSON boolean");
    }
    return node.asBoolean();
  }

  /** Exactly three finite numbers; the reference casts them to float32, as does this port. */
  private static float[] floatVector(JsonNode root, String key) {
    JsonNode node = root.path(key);
    if (!node.isArray() || node.size() != 3) {
      throw new IllegalArgumentException(key + " must be an array of exactly three numbers");
    }
    float[] out = new float[3];
    for (int i = 0; i < 3; i++) {
      JsonNode value = node.get(i);
      if (!value.isNumber()) {
        throw new IllegalArgumentException(key + "[" + i + "] must be a number");
      }
      double d = value.asDouble();
      if (!Double.isFinite(d)) {
        throw new IllegalArgumentException(key + "[" + i + "] must be finite: " + d);
      }
      out[i] = (float) d;
    }
    return out;
  }

  /** A programmatic builder for configurations that do not come from a pinned artifact JSON. */
  public static Builder builder() {
    return new Builder();
  }

  /** Programmatic construction; see {@link #builder()}. */
  public static final class Builder {
    private int numCrops;
    private float[] imageMean;
    private float[] imageStd;
    private String processorClass;

    private Builder() {}

    /** The required positive {@code num_crops} (the HD scale loop's budget and the slot count). */
    public Builder numCrops(int numCrops) {
      this.numCrops = numCrops;
      return this;
    }

    /** The three per-channel float32 means; copied. */
    public Builder imageMean(float[] imageMean) {
      if (imageMean == null) {
        throw new IllegalArgumentException("imageMean must not be null");
      }
      this.imageMean = imageMean.clone();
      return this;
    }

    /** The three per-channel float32 standard deviations (each positive); copied. */
    public Builder imageStd(float[] imageStd) {
      if (imageStd == null) {
        throw new IllegalArgumentException("imageStd must not be null");
      }
      this.imageStd = imageStd.clone();
      return this;
    }

    /** The optional {@code processor_class} metadata string (null when unset). */
    public Builder processorClass(String processorClass) {
      this.processorClass = processorClass;
      return this;
    }

    /**
     * Builds the validated configuration, running the shared semantic guards (positivity and
     * finiteness).
     *
     * @throws IllegalArgumentException if a required value is unset or invalid
     */
    public Phi3VProcessorConfig build() {
      if (imageMean == null) {
        throw new IllegalArgumentException("imageMean must be set");
      }
      if (imageStd == null) {
        throw new IllegalArgumentException("imageStd must be set");
      }
      return new Phi3VProcessorConfig(numCrops, null, imageMean, imageStd, processorClass);
    }
  }

  /** The HD scale loop's crop budget (4 for the pinned artifact). */
  public int numCrops() {
    return numCrops;
  }

  /**
   * The legacy {@code num_img_tokens} value when present and non-null in the JSON, or {@code null}
   * when absent — the reference processor never reads it, so the model only cross-checks it against
   * the checkpoint's {@code img_processor.num_img_tokens} when it is present.
   */
  public Integer numImgTokens() {
    return numImgTokens;
  }

  /** A copy of the three per-channel float32 means (the CLIP defaults for the pinned artifact). */
  public float[] imageMean() {
    return imageMean.clone();
  }

  /**
   * A copy of the three per-channel float32 standard deviations (CLIP defaults for the artifact).
   */
  public float[] imageStd() {
    return imageStd.clone();
  }

  /** The {@code processor_class} metadata string, or null if absent. */
  public String processorClass() {
    return processorClass;
  }

  @Override
  public String toString() {
    return "Phi3VProcessorConfig{numCrops=" + numCrops + ", processorClass=" + processorClass + "}";
  }
}
