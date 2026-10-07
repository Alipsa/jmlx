package se.alipsa.jmlx.vision;

import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/**
 * Validated SmolVLM/Idefics3 image processor configuration, parsed from a Hugging Face {@code
 * preprocessor_config.json}.
 *
 * <p>Parsing is strict: the root must be a JSON object, every key must be a recognized
 * SmolVLM/Idefics3 processor option, all operational options must be present and have the expected
 * type, the {@code image_processor_type} must be {@code Idefics3ImageProcessor}, and {@code
 * size}/{@code max_image_size} must be objects with a single positive-integer {@code longest_edge}
 * entry. Unrecognized keys, missing operational options, malformed values and unsupported {@code
 * resample} numbers are rejected with a descriptive {@link IllegalArgumentException} that names the
 * offending key; unsupported resample values are never mapped to a nearby filter.
 *
 * <p>The pinned SmolVLM-256M configuration (the reference for this milestone) parses to: resize
 * longest edge to 2048, LANCZOS, image splitting on, tile size 512, rescale {@code
 * 0.00392156862745098}, normalize with mean/std {@code [0.5, 0.5, 0.5]}, pad on.
 */
public final class SmolVlmProcessorConfig {
  private static final ObjectMapper MAPPER = new ObjectMapper();

  private static final Set<String> KNOWN_KEYS =
      Set.of(
          "do_convert_rgb",
          "do_image_splitting",
          "do_normalize",
          "do_pad",
          "do_rescale",
          "do_resize",
          "image_mean",
          "image_processor_type",
          "image_std",
          "max_image_size",
          "processor_class",
          "resample",
          "rescale_factor",
          "size");

  private final boolean doConvertRgb;
  private final boolean doResize;
  private final int sizeLongestEdge;
  private final Resampling resample;
  private final boolean doImageSplitting;
  private final int maxImageSizeLongestEdge;
  private final boolean doRescale;
  private final double rescaleFactor;
  private final boolean doNormalize;
  private final float[] imageMean;
  private final float[] imageStd;
  private final boolean doPad;
  private final String processorClass;

  private SmolVlmProcessorConfig(
      boolean doConvertRgb,
      boolean doResize,
      int sizeLongestEdge,
      Resampling resample,
      boolean doImageSplitting,
      int maxImageSizeLongestEdge,
      boolean doRescale,
      double rescaleFactor,
      boolean doNormalize,
      float[] imageMean,
      float[] imageStd,
      boolean doPad,
      String processorClass) {
    this.doConvertRgb = doConvertRgb;
    this.doResize = doResize;
    this.sizeLongestEdge = sizeLongestEdge;
    this.resample = resample;
    this.doImageSplitting = doImageSplitting;
    this.maxImageSizeLongestEdge = maxImageSizeLongestEdge;
    this.doRescale = doRescale;
    this.rescaleFactor = rescaleFactor;
    this.doNormalize = doNormalize;
    this.imageMean = imageMean;
    this.imageStd = imageStd;
    this.doPad = doPad;
    this.processorClass = processorClass;
  }

  /**
   * Parses and validates a {@code preprocessor_config.json} document.
   *
   * @param json UTF-8 JSON bytes of the processor configuration
   * @return the validated configuration
   * @throws IllegalArgumentException if the document is not valid JSON, has unknown or missing
   *     keys, malformed values, a wrong {@code image_processor_type}, or an unsupported {@code
   *     resample} value
   */
  public static SmolVlmProcessorConfig fromJson(byte[] json) {
    if (json == null) {
      throw new IllegalArgumentException("json must not be null");
    }
    JsonNode root;
    try {
      root = MAPPER.readTree(json);
    } catch (JacksonException e) {
      throw new IllegalArgumentException(
          "preprocessor config is not valid JSON: " + e.getMessage(), e);
    }
    return fromNode(root);
  }

  private static SmolVlmProcessorConfig fromNode(JsonNode root) {
    if (!root.isObject()) {
      throw new IllegalArgumentException("preprocessor config root must be a JSON object");
    }
    Set<String> keys = new LinkedHashSet<>();
    for (Map.Entry<String, JsonNode> entry : root.properties()) {
      keys.add(entry.getKey());
    }
    for (String key : keys) {
      if (!KNOWN_KEYS.contains(key)) {
        throw new IllegalArgumentException(
            "preprocessor config has unrecognized key '" + key + "'");
      }
    }
    for (String key :
        new String[] {
          "do_convert_rgb",
          "do_image_splitting",
          "do_normalize",
          "do_pad",
          "do_rescale",
          "do_resize",
          "image_mean",
          "image_processor_type",
          "image_std",
          "max_image_size",
          "resample",
          "rescale_factor",
          "size"
        }) {
      if (!keys.contains(key)) {
        throw new IllegalArgumentException(
            "preprocessor config is missing required key '" + key + "'");
      }
    }
    if (!root.path("image_processor_type").isString()
        || !"Idefics3ImageProcessor".equals(root.path("image_processor_type").asString())) {
      throw new IllegalArgumentException(
          "preprocessor config image_processor_type must be 'Idefics3ImageProcessor'");
    }
    JsonNode rescaleNode = root.path("rescale_factor");
    if (!rescaleNode.isNumber()) {
      throw new IllegalArgumentException("rescale_factor must be a JSON number");
    }
    double rescaleFactor = rescaleNode.asDouble();
    if (!Double.isFinite(rescaleFactor) || rescaleFactor <= 0.0) {
      throw new IllegalArgumentException(
          "rescale_factor must be finite and positive: " + rescaleFactor);
    }
    boolean doNormalize = bool(root, "do_normalize");
    float[] imageMean = floatVector(root, "image_mean");
    float[] imageStd = floatVector(root, "image_std");
    for (float s : imageStd) {
      if (s <= 0.0f) {
        throw new IllegalArgumentException("image_std values must be positive: " + s);
      }
    }
    boolean doPad = bool(root, "do_pad");
    String processorClass =
        root.has("processor_class") ? root.path("processor_class").asString() : null;

    return new SmolVlmProcessorConfig(
        bool(root, "do_convert_rgb"),
        bool(root, "do_resize"),
        longestEdge(root, "size"),
        parseResample(root.path("resample")),
        bool(root, "do_image_splitting"),
        longestEdge(root, "max_image_size"),
        bool(root, "do_rescale"),
        rescaleFactor,
        doNormalize,
        imageMean,
        imageStd,
        doPad,
        processorClass);
  }

  private static boolean bool(JsonNode root, String key) {
    JsonNode node = root.path(key);
    if (!node.isBoolean()) {
      throw new IllegalArgumentException(key + " must be a JSON boolean");
    }
    return node.asBoolean();
  }

  /** Accepts exactly {"longest_edge": positiveInt}; rejects height/width variants. */
  private static int longestEdge(JsonNode root, String key) {
    JsonNode node = root.path(key);
    if (!node.isObject()) {
      throw new IllegalArgumentException(key + " must be a JSON object");
    }
    int count = 0;
    for (Map.Entry<String, JsonNode> entry : node.properties()) {
      count++;
      if (!"longest_edge".equals(entry.getKey())) {
        throw new IllegalArgumentException(key + " must have exactly the key 'longest_edge'");
      }
    }
    if (count != 1) {
      throw new IllegalArgumentException(key + " must have exactly the key 'longest_edge'");
    }
    JsonNode edge = node.path("longest_edge");
    if (!edge.isIntegralNumber()) {
      throw new IllegalArgumentException(key + ".longest_edge must be an integer");
    }
    int value = edge.asInt();
    if (value <= 0) {
      throw new IllegalArgumentException(key + ".longest_edge must be positive: " + value);
    }
    return value;
  }

  /** Maps the reference's resample integers (1=LANCZOS, 2=BILINEAR, 3=BICUBIC). */
  private static Resampling parseResample(JsonNode node) {
    if (!node.isIntegralNumber()) {
      throw new IllegalArgumentException("resample must be an integer");
    }
    int value = node.asInt();
    return switch (value) {
      case 1 -> Resampling.LANCZOS;
      case 2 -> Resampling.BILINEAR;
      case 3 -> Resampling.BICUBIC;
      default ->
          throw new IllegalArgumentException(
              "unsupported resample value "
                  + value
                  + " (expected 1=LANCZOS, 2=BILINEAR or 3=BICUBIC)");
    };
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

  /** Whether the reference converts decoded input to RGB (the ported decoder always does). */
  public boolean doConvertRgb() {
    return doConvertRgb;
  }

  /** Whether the first resize stage is applied. */
  public boolean doResize() {
    return doResize;
  }

  /** The first-stage longest edge (2048 for the pinned SmolVLM-256M config). */
  public int sizeLongestEdge() {
    return sizeLongestEdge;
  }

  /** The resampling kernel applied at every resize stage. */
  public Resampling resample() {
    return resample;
  }

  /** Whether image splitting into tiles is applied. */
  public boolean doImageSplitting() {
    return doImageSplitting;
  }

  /** The tile/global size (512 for the pinned SmolVLM-256M config). */
  public int maxImageSizeLongestEdge() {
    return maxImageSizeLongestEdge;
  }

  /** Whether the rescale stage is applied. */
  public boolean doRescale() {
    return doRescale;
  }

  /** The rescale factor as parsed (0.00392156862745098 for the pinned config). */
  public double rescaleFactor() {
    return rescaleFactor;
  }

  /** Whether the normalize stage is applied. */
  public boolean doNormalize() {
    return doNormalize;
  }

  /** A copy of the three per-channel float32 means. */
  public float[] imageMean() {
    return imageMean.clone();
  }

  /** A copy of the three per-channel float32 standard deviations. */
  public float[] imageStd() {
    return imageStd.clone();
  }

  /** Whether tiles are padded to the largest tile size. */
  public boolean doPad() {
    return doPad;
  }

  /** The {@code processor_class} metadata string, or null if absent. */
  public String processorClass() {
    return processorClass;
  }

  @Override
  public String toString() {
    return "SmolVlmProcessorConfig{sizeLongestEdge="
        + sizeLongestEdge
        + ", resample="
        + resample
        + ", maxImageSizeLongestEdge="
        + maxImageSizeLongestEdge
        + ", rescaleFactor="
        + rescaleFactor
        + "}";
  }
}
