package se.alipsa.jmlx.models;

import java.util.Objects;

/**
 * One image's ordered tile feature destinations in expanded-ID space: each entry is the position
 * where that tile's image-token run begins, in the processor's frame order (crops row-major, global
 * thumbnail last). Every tile run is exactly as long as the plan's tokens-per-tile, so the tile's
 * feature rows occupy {@code [tileStart, tileStart + tokensPerTile)}.
 *
 * @param tileStarts one start position per tile, strictly increasing
 */
record SmolVlmImagePlacement(int[] tileStarts) {

  public SmolVlmImagePlacement {
    tileStarts = Objects.requireNonNull(tileStarts, "tileStarts").clone();
    if (tileStarts.length == 0) {
      throw new IllegalArgumentException("at least one tile start is required");
    }
    for (int i = 1; i < tileStarts.length; i++) {
      if (tileStarts[i] <= tileStarts[i - 1]) {
        throw new IllegalArgumentException(
            "tile starts must be strictly increasing: "
                + tileStarts[i]
                + " at index "
                + i
                + " after "
                + tileStarts[i - 1]);
      }
    }
  }

  /** One start position per tile, strictly increasing; defensive copy. */
  public int[] tileStarts() {
    return tileStarts.clone();
  }

  /** Number of tiles this image expands to. */
  public int tileCount() {
    return tileStarts.length;
  }
}
