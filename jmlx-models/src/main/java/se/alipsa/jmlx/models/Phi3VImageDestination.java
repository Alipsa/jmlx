package se.alipsa.jmlx.models;

/**
 * One image's contiguous sentinel run in expanded-ID space: the {@code numImgTokens} rows of {@code
 * -imageIndex} that the vision tower's projected rows replace by row position, in prompt order.
 * Every Phi-3.5-Vision image expands to one contiguous run (the marker's sentinel block), so a
 * destination is a start position and a count. Equality is by value.
 *
 * @param sentinelStart the first expanded-ID position holding this image's sentinel rows
 * @param sentinelCount the image's image-token count (the run's length)
 */
record Phi3VImageDestination(int sentinelStart, int sentinelCount) {

  public Phi3VImageDestination {
    if (sentinelStart < 0) {
      throw new IllegalArgumentException("sentinelStart must be non-negative: " + sentinelStart);
    }
    if (sentinelCount <= 0) {
      throw new IllegalArgumentException("sentinelCount must be positive: " + sentinelCount);
    }
  }

  /** The run's exclusive end in expanded-ID space, in checked arithmetic. */
  int sentinelEnd() {
    return Math.addExact(sentinelStart, sentinelCount);
  }
}
