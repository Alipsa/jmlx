package se.alipsa.jmlx.vision;

import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/**
 * Validated Qwen3-VL image processor configuration, parsed from a Hugging Face {@code
 * preprocessor_config.json}, for the slow PIL {@code Qwen2VLImageProcessor} preprocessing path that
 * {@link Qwen3VlImageProcessor} ports.
 *
 * <p>The pinned Qwen3-VL artifacts declare {@code image_processor_type: Qwen2VLImageProcessorFast},
 * but the recorded reference decision (findings §4.3, {@code
 * req/plans/phase7-3b-reference-findings.md}) is that the slow PIL processor is the ground-truth
 * port target: the ported geometry and pixel math are the slow processor's, byte-verified against
 * it by the WP1 probe goldens. This config therefore accepts both {@code Qwen2VLImageProcessor} and
 * {@code Qwen2VLImageProcessorFast} as the declared type; accepting the Fast declaration is a
 * documented deviation, not an equivalence claim (findings §7 measures and bounds the fast/slow
 * difference).
 *
 * <p>Parsing is strict: the root must be a JSON object, every key must be a recognized processor
 * option, required options must be present with the expected type, and malformed values are
 * rejected with a descriptive {@link IllegalArgumentException} that names the offending key — the
 * same contract as {@link SmolVlmProcessorConfig}, with the message prefix identifying this config
 * as the Qwen3-VL one.
 *
 * <p><b>min/max pixels and the effective cap.</b> The reference {@code __init__} lets explicit
 * {@code min_pixels}/{@code max_pixels} keys override {@code size.shortest_edge}/{@code
 * size.longest_edge} respectively, and this port keeps that precedence. The pinned 4.57.6 slow
 * processor has an {@code __init__} bug that silently replaces a config-supplied {@code size} dict
 * with the defaults {3136, 1003520} when no explicit min/max keys are present (findings §4.2,
 * §4.3); this port reads the JSON directly and is unaffected, which is exactly why the effective
 * cap below matters: without it, a declared 16,777,216-pixel ceiling would plan images with
 * hundreds of thousands of raw patch rows. In {@link #fromJson(byte[])} the effective {@code
 * maxPixels} is the declared value (the {@code max_pixels} override or {@code size.longest_edge})
 * capped at {@code hardCapPixels() = rawRowBudget * patchSize * patchSize} — the Phase 7.3b
 * decision (b) override of the artifact's 16,777,216 px ceiling (1,048,576 at the default 4,096-row
 * budget and patch size 16), not a rejection: the pre-cap value stays available as {@link
 * #declaredMaxPixels()} for provenance.
 *
 * <p><b>Hard budget.</b> {@code rawRowBudget} (default {@link #DEFAULT_RAW_ROW_BUDGET}, the Phase
 * 7.3b measured budget) bounds the raw patch rows a single image may plan to, in both the resize
 * and no-resize paths (see {@link Qwen3VlImageProcessor#geometry(int, int)}). Raising the hard
 * budget requires independent measured memory evidence and an explicit application override through
 * the builder; it is deliberately not a JSON option, so an untrusted config cannot lift the cap.
 */
public final class Qwen3VlProcessorConfig {
  private static final ObjectMapper MAPPER = new ObjectMapper();

  /**
   * Default hard budget: 4,096 raw patch rows per image (Phase 7.3b decision (c), the measured
   * budget). Only the programmatic builder can change it.
   */
  public static final int DEFAULT_RAW_ROW_BUDGET = 4096;

  private static final Set<String> KNOWN_KEYS =
      Set.of(
          "do_convert_rgb",
          "do_normalize",
          "do_rescale",
          "do_resize",
          "image_mean",
          "image_processor_type",
          "image_std",
          "max_pixels",
          "merge_size",
          "min_pixels",
          "patch_size",
          "processor_class",
          "resample",
          "rescale_factor",
          "size",
          "temporal_patch_size");

  private static final Set<String> SIZE_KEYS = Set.of("shortest_edge", "longest_edge");

  private static final String[] REQUIRED_KEYS = {
    "image_mean",
    "image_processor_type",
    "image_std",
    "merge_size",
    "patch_size",
    "size",
    "temporal_patch_size"
  };

  private final int patchSize;
  private final int temporalPatchSize;
  private final int mergeSize;
  private final int minPixels;
  private final int maxPixels;
  private final int declaredMaxPixels;
  private final int rawRowBudget;
  private final int hardCapPixels;
  private final boolean doConvertRgb;
  private final boolean doResize;
  private final Resampling resample;
  private final double rescaleFactor;
  private final float[] imageMean;
  private final float[] imageStd;
  private final String processorClass;

  private Qwen3VlProcessorConfig(
      int patchSize,
      int temporalPatchSize,
      int mergeSize,
      int minPixels,
      int maxPixels,
      int declaredMaxPixels,
      int rawRowBudget,
      boolean doConvertRgb,
      boolean doResize,
      Resampling resample,
      double rescaleFactor,
      float[] imageMean,
      float[] imageStd,
      String processorClass) {
    // Shared semantic validation for both construction paths: the JSON path has already checked
    // types, the builder has not checked anything, so both reach the same guards here.
    if (resample == null) {
      throw new IllegalArgumentException("resample must not be null");
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
    if (!Double.isFinite(rescaleFactor) || rescaleFactor <= 0.0) {
      throw new IllegalArgumentException(
          "rescaleFactor must be finite and positive: " + rescaleFactor);
    }
    if (patchSize <= 0) {
      throw new IllegalArgumentException("patchSize must be positive: " + patchSize);
    }
    if (temporalPatchSize <= 0) {
      throw new IllegalArgumentException(
          "temporalPatchSize must be positive: " + temporalPatchSize);
    }
    if (mergeSize <= 0) {
      throw new IllegalArgumentException("mergeSize must be positive: " + mergeSize);
    }
    if (rawRowBudget <= 0) {
      throw new IllegalArgumentException("rawRowBudget must be positive: " + rawRowBudget);
    }
    if (minPixels <= 0) {
      throw new IllegalArgumentException("minPixels must be positive: " + minPixels);
    }
    if (maxPixels <= 0) {
      throw new IllegalArgumentException("maxPixels must be positive: " + maxPixels);
    }
    long factor = (long) patchSize * mergeSize;
    if (factor > Integer.MAX_VALUE) {
      throw new IllegalArgumentException(
          "patchSize * mergeSize overflows a signed 32-bit integer: " + factor);
    }
    // The patch square alone fits a long (patchSize <= 2^31); the two products below are checked
    // by division (limit / psSq) instead of multiplying first, so a value that would overflow a
    // long is rejected too, not only one that exceeds the int range after the wrap.
    long psSq = (long) patchSize * (long) patchSize;
    long strideMultiplier = 3L * (long) temporalPatchSize;
    if (strideMultiplier > Integer.MAX_VALUE / psSq) {
      throw new IllegalArgumentException(
          "3 * temporalPatchSize * patchSize * patchSize overflows a signed 32-bit integer: "
              + strideMultiplier
              + " x "
              + psSq);
    }
    if ((long) rawRowBudget > Integer.MAX_VALUE / psSq) {
      throw new IllegalArgumentException(
          "rawRowBudget * patchSize * patchSize overflows a signed 32-bit integer: "
              + rawRowBudget
              + " x "
              + psSq);
    }
    long hardCap = (long) rawRowBudget * psSq;
    if (declaredMaxPixels < maxPixels) {
      throw new IllegalArgumentException(
          "declaredMaxPixels must be at least maxPixels: " + declaredMaxPixels + " < " + maxPixels);
    }
    if (minPixels > maxPixels) {
      if (declaredMaxPixels > maxPixels) {
        // The effective value was lowered by the decision (b) cap; naming it as the declared
        // value would be misleading (declaredMaxPixels > maxPixels implies maxPixels == hardCap
        // on the JSON path, and the builder passes equal values, so the else branch below is
        // the only one that can apply there).
        throw new IllegalArgumentException(
            "min_pixels "
                + minPixels
                + " exceeds the effective max_pixels "
                + maxPixels
                + " (declared "
                + declaredMaxPixels
                + ", capped at the raw-row hard cap "
                + hardCap
                + ")");
      }
      throw new IllegalArgumentException(
          "minPixels must be at most maxPixels: " + minPixels + " > " + maxPixels);
    }
    if (maxPixels > hardCap) {
      throw new IllegalArgumentException(
          "maxPixels "
              + maxPixels
              + " exceeds the hard pixel cap of "
              + hardCap
              + " (rawRowBudget "
              + rawRowBudget
              + " x patchSize^2 "
              + (long) patchSize * patchSize
              + ")");
    }
    this.patchSize = patchSize;
    this.temporalPatchSize = temporalPatchSize;
    this.mergeSize = mergeSize;
    this.minPixels = minPixels;
    this.maxPixels = maxPixels;
    this.declaredMaxPixels = declaredMaxPixels;
    this.rawRowBudget = rawRowBudget;
    this.hardCapPixels = (int) hardCap;
    this.doConvertRgb = doConvertRgb;
    this.doResize = doResize;
    this.resample = resample;
    this.rescaleFactor = rescaleFactor;
    this.imageMean = imageMean.clone();
    this.imageStd = imageStd.clone();
    this.processorClass = processorClass;
  }

  /**
   * Parses and validates a {@code preprocessor_config.json} document. The effective {@code
   * maxPixels} is the declared value capped at {@code hardCapPixels()} (decision (b); see the class
   * Javadoc); the pre-cap value is exposed as {@link #declaredMaxPixels()}.
   *
   * @param json UTF-8 JSON bytes of the processor configuration
   * @return the validated configuration
   * @throws IllegalArgumentException if the document is not valid JSON, has unknown or missing
   *     keys, malformed values, a wrong {@code image_processor_type}, an unsupported {@code
   *     resample} value, or a min/max pixel ordering the reference could not plan
   */
  public static Qwen3VlProcessorConfig fromJson(byte[] json) {
    if (json == null) {
      throw new IllegalArgumentException("json must not be null");
    }
    JsonNode root;
    try {
      root = MAPPER.readTree(json);
    } catch (JacksonException e) {
      throw new IllegalArgumentException(
          "qwen3-vl preprocessor config is not valid JSON: " + e.getMessage(), e);
    }
    return fromNode(root);
  }

  private static Qwen3VlProcessorConfig fromNode(JsonNode root) {
    if (!root.isObject()) {
      throw new IllegalArgumentException("qwen3-vl preprocessor config root must be a JSON object");
    }
    Set<String> keys = new LinkedHashSet<>();
    for (Map.Entry<String, JsonNode> entry : root.properties()) {
      keys.add(entry.getKey());
    }
    for (String key : keys) {
      if (!KNOWN_KEYS.contains(key)) {
        throw new IllegalArgumentException(
            "qwen3-vl preprocessor config has unrecognized key '" + key + "'");
      }
    }
    for (String key : REQUIRED_KEYS) {
      if (!keys.contains(key)) {
        throw new IllegalArgumentException(
            "qwen3-vl preprocessor config is missing required key '" + key + "'");
      }
    }
    if (!root.path("image_processor_type").isString()
        || (!"Qwen2VLImageProcessor".equals(root.path("image_processor_type").asString())
            && !"Qwen2VLImageProcessorFast".equals(root.path("image_processor_type").asString()))) {
      throw new IllegalArgumentException(
          "qwen3-vl preprocessor config image_processor_type must be 'Qwen2VLImageProcessor' or "
              + "'Qwen2VLImageProcessorFast'");
    }
    int[] edges = sizeEdges(root);
    int minPixels = edges[0];
    int declaredMaxPixels = edges[1];
    // Reference __init__ precedence: explicit legacy keys override the size dict entries.
    int minOverride = optionalPositiveInt(root, "min_pixels");
    int maxOverride = optionalPositiveInt(root, "max_pixels");
    if (minOverride != 0) {
      minPixels = minOverride;
    }
    if (maxOverride != 0) {
      declaredMaxPixels = maxOverride;
    }
    int patchSize = positiveInt(root, "patch_size");
    // Final: these values are parsed here (in reference key-validation order) but only consumed
    // by the constructor at the end of this method.
    final int temporalPatchSize = positiveInt(root, "temporal_patch_size");
    final int mergeSize = positiveInt(root, "merge_size");
    // Division-ordered check: patch_size^2 can be near 2^62, so multiplying the budget in first
    // can overflow a long and wrap past the range check (see the constructor's note).
    long psSq = (long) patchSize * (long) patchSize;
    if ((long) DEFAULT_RAW_ROW_BUDGET > Integer.MAX_VALUE / psSq) {
      throw new IllegalArgumentException(
          "the default raw row budget ("
              + DEFAULT_RAW_ROW_BUDGET
              + ") times patch_size^2 ("
              + psSq
              + ") overflows a signed 32-bit integer");
    }
    long hardCap = (long) DEFAULT_RAW_ROW_BUDGET * psSq;
    // Decision (b): the declared ceiling is capped, not rejected; the pre-cap value is kept as
    // declaredMaxPixels for provenance.
    final int maxPixels = (int) Math.min(declaredMaxPixels, hardCap);
    final float[] imageMean = floatVector(root, "image_mean");
    final float[] imageStd = floatVector(root, "image_std");
    String processorClass = null;
    if (root.has("processor_class")) {
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
    boolean doRescale = optionalBool(root, "do_rescale", true);
    if (!doRescale) {
      throw new IllegalArgumentException(
          "do_rescale must be true: the ported output contract is rescaled and normalized "
              + "patches, and a config that skips the rescale cannot be honored");
    }
    boolean doNormalize = optionalBool(root, "do_normalize", true);
    if (!doNormalize) {
      throw new IllegalArgumentException(
          "do_normalize must be true: the ported output contract is rescaled and normalized "
              + "patches, and a config that skips the normalize cannot be honored");
    }
    for (float s : imageStd) {
      if (s <= 0.0f) {
        throw new IllegalArgumentException("image_std values must be positive: " + s);
      }
    }
    return new Qwen3VlProcessorConfig(
        patchSize,
        temporalPatchSize,
        mergeSize,
        minPixels,
        maxPixels,
        declaredMaxPixels,
        DEFAULT_RAW_ROW_BUDGET,
        doConvertRgb,
        optionalBool(root, "do_resize", true),
        parseResample(root),
        parseRescaleFactor(root),
        imageMean,
        imageStd,
        processorClass);
  }

  /** Exactly {shortest_edge, longest_edge}, both positive ints; returns [shortest, longest]. */
  private static int[] sizeEdges(JsonNode root) {
    JsonNode node = root.path("size");
    if (!node.isObject()) {
      throw new IllegalArgumentException("size must be a JSON object");
    }
    Set<String> keys = new LinkedHashSet<>();
    for (Map.Entry<String, JsonNode> entry : node.properties()) {
      keys.add(entry.getKey());
    }
    if (keys.size() != SIZE_KEYS.size() || !keys.containsAll(SIZE_KEYS)) {
      throw new IllegalArgumentException(
          "size must have exactly the keys 'shortest_edge' and 'longest_edge'");
    }
    return new int[] {sizeEdge(node, "shortest_edge"), sizeEdge(node, "longest_edge")};
  }

  private static int sizeEdge(JsonNode node, String key) {
    JsonNode edge = node.path(key);
    if (!edge.isIntegralNumber()) {
      throw new IllegalArgumentException("size." + key + " must be an integer");
    }
    // canConvertToInt() first: asInt() throws a JsonNodeException (not the documented
    // IllegalArgumentException) for values outside the int range.
    if (!edge.canConvertToInt()) {
      throw new IllegalArgumentException(
          "size." + key + " must fit in a signed 32-bit integer: " + edge.asString());
    }
    int value = edge.asInt();
    if (value <= 0) {
      throw new IllegalArgumentException("size." + key + " must be positive: " + value);
    }
    return value;
  }

  /** 0 when the key is absent, else the positive int. */
  private static int optionalPositiveInt(JsonNode root, String key) {
    if (!root.has(key)) {
      return 0;
    }
    return positiveInt(root, key);
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

  /**
   * Maps the reference's resample integers (PIL constants 2=BILINEAR, 3=BICUBIC, 4=LANCZOS);
   * 1=NEAREST and every other value are rejected — the port has no NEAREST kernel.
   */
  private static Resampling parseResample(JsonNode root) {
    if (!root.has("resample")) {
      return Resampling.BICUBIC;
    }
    JsonNode node = root.path("resample");
    if (!node.isIntegralNumber()) {
      throw new IllegalArgumentException("resample must be an integer");
    }
    if (!node.canConvertToInt()) {
      throw new IllegalArgumentException(
          "resample must fit in a signed 32-bit integer: " + node.asString());
    }
    int value = node.asInt();
    return switch (value) {
      case 2 -> Resampling.BILINEAR;
      case 3 -> Resampling.BICUBIC;
      case 4 -> Resampling.LANCZOS;
      default ->
          throw new IllegalArgumentException(
              "unsupported resample value "
                  + value
                  + " (expected 2=BILINEAR, 3=BICUBIC or 4=LANCZOS)");
    };
  }

  private static double parseRescaleFactor(JsonNode root) {
    if (!root.has("rescale_factor")) {
      return 1.0 / 255.0;
    }
    JsonNode node = root.path("rescale_factor");
    if (!node.isNumber()) {
      throw new IllegalArgumentException("rescale_factor must be a JSON number");
    }
    double value = node.asDouble();
    if (!Double.isFinite(value) || value <= 0.0) {
      throw new IllegalArgumentException("rescale_factor must be finite and positive: " + value);
    }
    return value;
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

  /**
   * A programmatic builder for configurations that do not come from a pinned artifact JSON (for
   * example the no-resize test and planning paths). {@code maxPixels} here is already the effective
   * value: it is rejected above {@code hardCapPixels()}, so the builder's hard-budget semantics are
   * enforced at construction, before any image planning.
   */
  public static Builder builder() {
    return new Builder();
  }

  /** Programmatic construction; see {@link #builder()}. */
  public static final class Builder {
    private int patchSize;
    private int temporalPatchSize;
    private int mergeSize;
    private int minPixels;
    private int maxPixels;
    private int rawRowBudget = DEFAULT_RAW_ROW_BUDGET;
    private boolean doResize = true;
    private Resampling resample = Resampling.BICUBIC;
    private double rescaleFactor = 1.0 / 255.0;
    private float[] imageMean;
    private float[] imageStd;
    private String processorClass;

    private Builder() {}

    /** The spatial patch size of the vision encoder (16 for the pinned artifacts). */
    public Builder patchSize(int patchSize) {
      this.patchSize = patchSize;
      return this;
    }

    /** The temporal patch size (2 for the pinned artifacts). */
    public Builder temporalPatchSize(int temporalPatchSize) {
      this.temporalPatchSize = temporalPatchSize;
      return this;
    }

    /** The spatial merge size (2 for the pinned artifacts). */
    public Builder mergeSize(int mergeSize) {
      this.mergeSize = mergeSize;
      return this;
    }

    /** The effective min-pixel bound; must be positive and at most {@code maxPixels}. */
    public Builder minPixels(int minPixels) {
      this.minPixels = minPixels;
      return this;
    }

    /**
     * The effective max-pixel bound: unlike {@code fromJson}, it is not capped again — it is
     * rejected outright when it exceeds the hard pixel cap.
     */
    public Builder maxPixels(int maxPixels) {
      this.maxPixels = maxPixels;
      return this;
    }

    /**
     * Overrides the default hard budget of {@link #DEFAULT_RAW_ROW_BUDGET} raw patch rows; the
     * override requires the application to have measured evidence that the larger budget is
     * affordable (see the class Javadoc).
     */
    public Builder rawRowBudget(int rawRowBudget) {
      this.rawRowBudget = rawRowBudget;
      return this;
    }

    /** Whether the resize stage is applied (default true). */
    public Builder doResize(boolean doResize) {
      this.doResize = doResize;
      return this;
    }

    /** The resampling kernel (default BICUBIC, the reference default). */
    public Builder resample(Resampling resample) {
      this.resample = resample;
      return this;
    }

    /** The finite positive rescale factor (default 1/255, the reference default). */
    public Builder rescaleFactor(double rescaleFactor) {
      this.rescaleFactor = rescaleFactor;
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
     * Builds the validated configuration, running the shared semantic guards (positivity, int
     * overflow of the derived factor/stride/cap, the min/max ordering and the hard pixel cap).
     *
     * @throws IllegalArgumentException if a required value is unset, invalid, or {@code maxPixels}
     *     exceeds the hard pixel cap
     */
    public Qwen3VlProcessorConfig build() {
      if (imageMean == null) {
        throw new IllegalArgumentException("imageMean must be set");
      }
      if (imageStd == null) {
        throw new IllegalArgumentException("imageStd must be set");
      }
      // Builder semantics: maxPixels is the effective value, so declaredMaxPixels equals it, and
      // doConvertRgb is true by construction (the decode path always delivers RGB).
      return new Qwen3VlProcessorConfig(
          patchSize,
          temporalPatchSize,
          mergeSize,
          minPixels,
          maxPixels,
          maxPixels,
          rawRowBudget,
          true,
          doResize,
          resample,
          rescaleFactor,
          imageMean,
          imageStd,
          processorClass);
    }
  }

  /** The spatial patch size of the vision encoder (16 for the pinned artifacts). */
  public int patchSize() {
    return patchSize;
  }

  /** The temporal patch size (2 for the pinned artifacts). */
  public int temporalPatchSize() {
    return temporalPatchSize;
  }

  /** The spatial merge size (2 for the pinned artifacts). */
  public int mergeSize() {
    return mergeSize;
  }

  /** The effective min-pixel bound (65,536 for the pinned artifact). */
  public int minPixels() {
    return minPixels;
  }

  /**
   * The effective max-pixel bound: for a parsed config, the declared value capped at {@link
   * #hardCapPixels()} (1,048,576 for the pinned artifact at the default budget); for a built
   * config, the value the application asked for.
   */
  public int maxPixels() {
    return maxPixels;
  }

  /**
   * The declared max-pixel bound before the decision (b) cap (16,777,216 for the pinned artifact);
   * equals {@link #maxPixels()} for a built config.
   */
  public int declaredMaxPixels() {
    return declaredMaxPixels;
  }

  /** The raw-patch-row budget (default {@link #DEFAULT_RAW_ROW_BUDGET}). */
  public int rawRowBudget() {
    return rawRowBudget;
  }

  /** The hard pixel cap {@code rawRowBudget * patchSize * patchSize} (1,048,576 by default). */
  public int hardCapPixels() {
    return hardCapPixels;
  }

  /** Whether the reference converts decoded input to RGB (always true; false is rejected). */
  public boolean doConvertRgb() {
    return doConvertRgb;
  }

  /** Whether the resize stage is applied. */
  public boolean doResize() {
    return doResize;
  }

  /** The resampling kernel applied by the resize stage (BICUBIC for the pinned artifact). */
  public Resampling resample() {
    return resample;
  }

  /** The rescale factor as parsed (1/255 for the pinned artifact). */
  public double rescaleFactor() {
    return rescaleFactor;
  }

  /** A copy of the three per-channel float32 means. */
  public float[] imageMean() {
    return imageMean.clone();
  }

  /** A copy of the three per-channel float32 standard deviations. */
  public float[] imageStd() {
    return imageStd.clone();
  }

  /** The {@code processor_class} metadata string, or null if absent. */
  public String processorClass() {
    return processorClass;
  }

  /**
   * The {@code smart_resize} factor, {@code patchSize * mergeSize} (32 for the pinned artifacts);
   * both resized dimensions are multiples of it.
   */
  public int factor() {
    return patchSize * mergeSize;
  }

  /**
   * The flattened-patch row width, {@code 3 * temporalPatchSize * patchSize * patchSize} (1,536 for
   * the pinned artifacts).
   */
  public int patchStride() {
    return 3 * temporalPatchSize * patchSize * patchSize;
  }

  @Override
  public String toString() {
    return "Qwen3VlProcessorConfig{patchSize="
        + patchSize
        + ", temporalPatchSize="
        + temporalPatchSize
        + ", mergeSize="
        + mergeSize
        + ", minPixels="
        + minPixels
        + ", maxPixels="
        + maxPixels
        + ", declaredMaxPixels="
        + declaredMaxPixels
        + ", rawRowBudget="
        + rawRowBudget
        + ", resample="
        + resample
        + ", rescaleFactor="
        + rescaleFactor
        + "}";
  }
}
