package se.alipsa.jmlx.models;

/**
 * One image's contiguous feature destination range in expanded-ID space: the run of image tokens
 * that the vision tower's merged rows replace, in prompt order. Every Qwen3-VL image expands to one
 * contiguous run (unlike SmolVLM's per-tile destinations), so a placement is a start position and a
 * count. Equality is by value.
 *
 * @param featureStart the first expanded-ID position holding this image's tokens
 * @param featureCount the image's merged feature row count (the run's length)
 */
record Qwen3VlImagePlacement(int featureStart, int featureCount) {

  public Qwen3VlImagePlacement {
    if (featureStart < 0) {
      throw new IllegalArgumentException("featureStart must be non-negative: " + featureStart);
    }
    if (featureCount <= 0) {
      throw new IllegalArgumentException("featureCount must be positive: " + featureCount);
    }
  }

  /** The run's exclusive end in expanded-ID space, in checked arithmetic. */
  int featureEnd() {
    return Math.addExact(featureStart, featureCount);
  }
}
