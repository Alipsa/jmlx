package se.alipsa.jmlx.nn;

import java.util.Objects;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXFast;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/** Immutable rotary-position specification with Hugging Face compatible scaling variants. */
public sealed interface RopeSpec
    permits RopeSpec.Base, RopeSpec.Linear, RopeSpec.DynamicNtk, RopeSpec.Llama3, RopeSpec.Yarn {

  /** Returns the base period used by this variant. */
  float theta();

  /** Returns the period for each pair of rotary coordinates. */
  default double[] frequencies(int rotaryDims, int sequenceLength) {
    validateDims(rotaryDims);
    double[] periods = basePeriods(theta(), rotaryDims);
    if (this instanceof DynamicNtk dynamic && sequenceLength > dynamic.maxPositions()) {
      if (rotaryDims <= 2) {
        throw new IllegalArgumentException("dynamic NTK requires rotaryDims > 2");
      }
      double lengthFactor =
          dynamic.factor() * sequenceLength / dynamic.maxPositions() - (dynamic.factor() - 1);
      double scaledTheta =
          dynamic.theta() * Math.pow(lengthFactor, (double) rotaryDims / (rotaryDims - 2));
      periods = basePeriods(scaledTheta, rotaryDims);
    } else if (this instanceof Llama3 llama3) {
      double lowWavelength = llama3.originalMaxPositions() / llama3.lowFreqFactor();
      double highWavelength = llama3.originalMaxPositions() / llama3.highFreqFactor();
      for (int i = 0; i < periods.length; i++) {
        double wavelength = 2 * Math.PI * periods[i];
        if (wavelength > lowWavelength) {
          periods[i] *= llama3.factor();
        } else if (wavelength >= highWavelength) {
          double smooth =
              (llama3.originalMaxPositions() / wavelength - llama3.lowFreqFactor())
                  / (llama3.highFreqFactor() - llama3.lowFreqFactor());
          periods[i] /= (1 - smooth) / llama3.factor() + smooth;
        }
      }
    } else if (this instanceof Yarn yarn) {
      double low = correctionDimension(yarn.betaFast(), rotaryDims, yarn);
      double high = correctionDimension(yarn.betaSlow(), rotaryDims, yarn);
      if (yarn.truncate()) {
        low = Math.floor(low);
        high = Math.ceil(high);
      }
      low = Math.max(low, 0);
      high = Math.min(high, rotaryDims - 1);
      if (low == high) {
        high += 0.001;
      }
      for (int i = 0; i < periods.length; i++) {
        double ramp = Math.max(0, Math.min(1, (i - low) / (high - low)));
        double extrapolation = 1 - ramp;
        double invFrequency =
            (1 - extrapolation) / (yarn.factor() * periods[i]) + extrapolation / periods[i];
        periods[i] = 1 / invFrequency;
      }
    }
    return periods;
  }

  /** Whether frequencies can be built once and shared by all layers. */
  default boolean isStatic() {
    return !(this instanceof DynamicNtk);
  }

  /** Position scale supplied to the MLX rotary kernel. */
  default float scale() {
    return this instanceof Linear linear ? 1 / linear.factor() : 1;
  }

  /** Multiplicative factor on rotated coordinates, used by YaRN. */
  default float attentionScaling() {
    if (!(this instanceof Yarn yarn)) {
      return 1;
    }
    if (yarn.attentionFactor() != null) {
      return yarn.attentionFactor();
    }
    if (yarn.mscale() != 0 && yarn.mscaleAllDim() != 0) {
      return (float)
          (mscale(yarn.factor(), yarn.mscale()) / mscale(yarn.factor(), yarn.mscaleAllDim()));
    }
    return (float) mscale(yarn.factor(), 1);
  }

  /**
   * Builds the frequency array for a step ending at {@code sequenceLength}, or returns {@code null}
   * when this variant is static and callers should share one precomputed array instead.
   */
  default MLXArray stepFrequencies(MLXScope scope, int rotaryDims, int sequenceLength) {
    return isStatic() ? null : frequencyArray(scope, rotaryDims, sequenceLength);
  }

  /**
   * Builds the one frequency array every layer and step can share, or returns {@code null} when
   * this variant needs none ({@link Base}, {@link Linear}) or varies with length ({@link
   * DynamicNtk}).
   */
  default MLXArray staticFrequencies(MLXScope scope, int rotaryDims) {
    return usesFrequencyArray() && isStatic() ? frequencyArray(scope, rotaryDims, 1) : null;
  }

  private boolean usesFrequencyArray() {
    return !(this instanceof Base) && !(this instanceof Linear);
  }

  private MLXArray frequencyArray(MLXScope scope, int rotaryDims, int sequenceLength) {
    double[] periods = frequencies(rotaryDims, sequenceLength);
    float[] data = new float[periods.length];
    for (int i = 0; i < data.length; i++) {
      data[i] = (float) periods[i];
    }
    return MLX.array(scope, data, new int[] {data.length});
  }

  /** Applies RoPE to the rotary prefix and preserves any remaining head coordinates. */
  default MLXArray apply(MLXArray x, int rotaryDims, int offset, MLXArray staticFreqs) {
    Objects.requireNonNull(x, "x");
    validateDims(rotaryDims);
    int[] shape = x.shape();
    int last = shape.length - 1;
    if (last < 1 || rotaryDims > shape[last]) {
      throw new IllegalArgumentException("rotaryDims exceeds head dimension");
    }
    MLXArray prefix = x;
    if (rotaryDims < shape[last]) {
      int[] start = new int[shape.length];
      int[] stop = shape.clone();
      stop[last] = rotaryDims;
      prefix = MLXShape.slice(x, start, stop);
    }
    MLXArray freqs = staticFreqs;
    if (freqs == null && usesFrequencyArray()) {
      freqs = frequencyArray(x.scope(), rotaryDims, offset + shape[last - 1]);
    }
    MLXArray rotated =
        MLXFast.rope(
            prefix, rotaryDims, false, freqs == null ? theta() : null, scale(), offset, freqs);
    float attentionScaling = attentionScaling();
    if (attentionScaling != 1) {
      rotated =
          MLXOps.multiply(
              rotated, MLX.full(x.scope(), new int[] {1}, attentionScaling, rotated.dtype()));
    }
    if (rotaryDims == shape[last]) {
      return rotated;
    }
    int[] start = new int[shape.length];
    int[] stop = shape.clone();
    start[last] = rotaryDims;
    return MLXShape.concatenate(new MLXArray[] {rotated, MLXShape.slice(x, start, stop)}, last);
  }

  private static double[] basePeriods(double theta, int dims) {
    double[] periods = new double[dims / 2];
    for (int i = 0; i < periods.length; i++) {
      periods[i] = Math.pow(theta, 2.0 * i / dims);
    }
    return periods;
  }

  private static double correctionDimension(float rotations, int dims, Yarn yarn) {
    return dims
        * Math.log(yarn.originalMaxPositions() / (rotations * 2 * Math.PI))
        / (2 * Math.log(yarn.theta()));
  }

  private static double mscale(float factor, float coefficient) {
    return factor <= 1 ? 1 : 0.1 * coefficient * Math.log(factor) + 1;
  }

  private static void validateDims(int dims) {
    if (dims <= 0 || (dims & 1) != 0) {
      throw new IllegalArgumentException("rotaryDims must be positive and even");
    }
  }

  private static void positive(float value, String key) {
    if (!Float.isFinite(value) || value <= 0) {
      throw new IllegalArgumentException(key + " must be positive and finite");
    }
  }

  /** Standard RoPE with a fixed frequency base. */
  record Base(float theta) implements RopeSpec {
    /** Validates the base. */
    public Base {
      positive(theta, "rope_theta");
    }
  }

  /** RoPE with linearly interpolated positions. */
  record Linear(float theta, float factor) implements RopeSpec {
    /** Validates the base and scaling factor. */
    public Linear {
      positive(theta, "rope_theta");
      positive(factor, "factor");
    }
  }

  /** Dynamic NTK scaling, recomputed when the sequence passes its original context length. */
  record DynamicNtk(float theta, float factor, int maxPositions) implements RopeSpec {
    /** Validates the base, factor, and original context length. */
    public DynamicNtk {
      positive(theta, "rope_theta");
      positive(factor, "factor");
      if (maxPositions <= 0) {
        throw new IllegalArgumentException("max_position_embeddings must be positive");
      }
    }
  }

  /** Llama 3's smooth interpolation between unscaled and divided frequencies. */
  record Llama3(
      float theta,
      float factor,
      float lowFreqFactor,
      float highFreqFactor,
      int originalMaxPositions)
      implements RopeSpec {
    /** Validates the scaling bands. */
    public Llama3 {
      positive(theta, "rope_theta");
      positive(factor, "factor");
      positive(lowFreqFactor, "low_freq_factor");
      positive(highFreqFactor, "high_freq_factor");
      if (highFreqFactor <= lowFreqFactor || originalMaxPositions <= 0) {
        throw new IllegalArgumentException("invalid llama3 RoPE frequency range");
      }
    }
  }

  /** YaRN frequency interpolation and attention scaling. */
  record Yarn(
      float theta,
      float factor,
      int originalMaxPositions,
      float betaFast,
      float betaSlow,
      float mscale,
      float mscaleAllDim,
      Float attentionFactor,
      boolean truncate)
      implements RopeSpec {
    /** Validates YaRN parameters. */
    public Yarn {
      positive(theta, "rope_theta");
      positive(factor, "factor");
      positive(betaFast, "beta_fast");
      positive(betaSlow, "beta_slow");
      if (originalMaxPositions <= 0) {
        throw new IllegalArgumentException("original_max_position_embeddings must be positive");
      }
      if (attentionFactor != null) {
        positive(attentionFactor, "attention_factor");
      }
    }
  }
}
