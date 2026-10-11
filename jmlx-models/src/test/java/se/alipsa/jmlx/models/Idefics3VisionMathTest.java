package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Pure math for the Idefics3 vision tower and connector: the variable-resolution position ids, the
 * pixel-to-patch mask unfold, the zero-tile predicate, the inputs-merger gather indices and the
 * pixel-shuffle permutation. No native code is involved. The non-trivial vectors are cross-checked
 * against the pinned transformers 4.57.6 / torch 2.6.0 float32 reference (the {@code
 * tools/hf-reference} venv), and the 4x4 position ids match every tile of the committed {@code
 * tools/hf-reference/goldens/idefics3.json} fixture (P = 4, all-ones masks).
 */
class Idefics3VisionMathTest {

  private static int[] mask(int width, int... flat) {
    if (flat.length % width != 0) {
      throw new IllegalArgumentException("mask values must fill whole rows");
    }
    return flat;
  }

  // --- positionIds -------------------------------------------------------------

  @Test
  void positionIdsFull4x4MatchEveryGoldenTile() {
    // Every tile of tools/hf-reference/goldens/idefics3.json carries this exact 16-id vector
    // (all-ones 4x4 patch mask, P = 4): HF's h / nb * (1 - 1e-6) bucketize with the 0.25/0.5/0.75
    // boundaries and the square stride.
    int[] full = mask(4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
    assertArrayEquals(
        new int[] {0, 0, 1, 2, 0, 0, 1, 2, 4, 4, 5, 6, 8, 8, 9, 10},
        Idefics3VisionTower.positionIds(full, 4, 4));
  }

  @Test
  void positionIdsMaskedColumnsAndRows() {
    // venv-verified (HF replica and the Java formula agree): a valid patch at (r, c) buckets its
    // own row and column fractions, so the dropped columns shift the surviving ids.
    int[] maskedColumns = mask(4, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0);
    assertArrayEquals(
        new int[] {0, 1, 0, 0, 0, 1, 0, 0, 4, 5, 0, 0, 8, 9, 0, 0},
        Idefics3VisionTower.positionIds(maskedColumns, 4, 4));

    int[] maskedRows = mask(4, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
    assertArrayEquals(
        new int[] {0, 0, 1, 2, 4, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0},
        Idefics3VisionTower.positionIds(maskedRows, 4, 4));
  }

  @Test
  void positionIdsDegenerateMasksYieldZeros() {
    // An all-zero tile and a tile whose column 0 (or row 0) is entirely padded would crash HF's
    // position_ids[mask] = pos_ids assignment; the Java helper degenerates to the 0 fill instead.
    int[] zero = mask(4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    int[] zeroIds = Idefics3VisionTower.positionIds(zero, 4, 4);
    for (int id : zeroIds) {
      assertEquals(0, id);
    }

    int[] columnZeroGap = mask(2, 0, 1, 0, 1);
    assertArrayEquals(new int[] {0, 0, 0, 0}, Idefics3VisionTower.positionIds(columnZeroGap, 2, 2));
  }

  @Test
  void positionIdsGeneralizeRectangularGrids() {
    // Non-square grid (ph = 2, pw = 4) with a non-rectangular validity mask: the per-(r, c)
    // formula with the per-axis boundaries and the pw stride, venv-verified.
    int[] rect = mask(4, 1, 1, 0, 0, 0, 1, 0, 0);
    assertArrayEquals(
        new int[] {0, 1, 0, 0, 0, 5, 0, 0}, Idefics3VisionTower.positionIds(rect, 2, 4));
  }

  @Test
  void positionIdsRejectInvalidArguments() {
    int[] full = mask(3, 1, 1, 1, 1, 1, 1, 1, 1, 1);
    assertThrows(IllegalArgumentException.class, () -> Idefics3VisionTower.positionIds(full, 2, 3));
    assertThrows(IllegalArgumentException.class, () -> Idefics3VisionTower.positionIds(full, 4, 3));
    assertThrows(IllegalArgumentException.class, () -> Idefics3VisionTower.positionIds(full, 0, 3));
    int[] notBinary = mask(3, 1, 2, 1, 1, 1, 1, 1, 1, 1);
    assertThrows(
        IllegalArgumentException.class, () -> Idefics3VisionTower.positionIds(notBinary, 3, 3));
    assertThrows(NullPointerException.class, () -> Idefics3VisionTower.positionIds(null, 3, 3));
  }

  // --- patchMaskFromPixelMask --------------------------------------------------

  @Test
  void patchMaskFromPixelMaskCountsAnyValidPixel() {
    // HF's unfold-then-sum(-1, -2) > 0: a patch survives when ANY of its 8x8 pixels is set,
    // not when all of them are.
    int[] allOnes = new int[32 * 32];
    java.util.Arrays.fill(allOnes, 1);
    int[] allValid = Idefics3VisionTower.patchMaskFromPixelMask(allOnes, 32, 32, 8);
    for (int value : allValid) {
      assertEquals(1, value);
    }

    int[] oneZeroPatch = allOnes.clone();
    for (int dr = 8; dr < 16; dr++) {
      for (int dc = 8; dc < 16; dc++) {
        oneZeroPatch[dr * 32 + dc] = 0;
      }
    }
    int[] oneInvalid = Idefics3VisionTower.patchMaskFromPixelMask(oneZeroPatch, 32, 32, 8);
    for (int i = 0; i < 16; i++) {
      assertEquals(i == 5 ? 0 : 1, oneInvalid[i], "patch index " + i);
    }

    int[] allZeros = new int[32 * 32];
    int[] noneValid = Idefics3VisionTower.patchMaskFromPixelMask(allZeros, 32, 32, 8);
    for (int value : noneValid) {
      assertEquals(0, value);
    }

    // A single non-zero pixel in an otherwise empty tile keeps exactly its own patch.
    int[] singlePixel = new int[32 * 32];
    singlePixel[20 * 32 + 3] = 1;
    int[] singleValid = Idefics3VisionTower.patchMaskFromPixelMask(singlePixel, 32, 32, 8);
    for (int i = 0; i < 16; i++) {
      assertEquals(i == 2 * 4 + 0 ? 1 : 0, singleValid[i], "patch index " + i);
    }
  }

  @Test
  void patchMaskFromPixelMaskRejectsInvalidArguments() {
    int[] any = new int[16];
    assertThrows(
        IllegalArgumentException.class,
        () -> Idefics3VisionTower.patchMaskFromPixelMask(any, 10, 16, 8));
    assertThrows(
        IllegalArgumentException.class,
        () -> Idefics3VisionTower.patchMaskFromPixelMask(any, 16, 10, 8));
    assertThrows(
        IllegalArgumentException.class,
        () -> Idefics3VisionTower.patchMaskFromPixelMask(any, 16, 16, 8));
    assertThrows(
        IllegalArgumentException.class,
        () -> Idefics3VisionTower.patchMaskFromPixelMask(any, 16, 16, 0));
    int[] notBinary = new int[16];
    notBinary[0] = 2;
    assertThrows(
        IllegalArgumentException.class,
        () -> Idefics3VisionTower.patchMaskFromPixelMask(notBinary, 4, 4, 2));
    assertThrows(
        NullPointerException.class,
        () -> Idefics3VisionTower.patchMaskFromPixelMask(null, 4, 4, 2));
  }

  // --- isZeroTile --------------------------------------------------------------

  @Test
  void isZeroTileDetectsExactlyAllZeroTiles() {
    assertTrue(Idefics3VisionTower.isZeroTile(new float[] {}));
    assertTrue(Idefics3VisionTower.isZeroTile(new float[] {0.0f, -0.0f, 0.0f}));
    assertFalse(Idefics3VisionTower.isZeroTile(new float[] {0.0f, 1e-30f, 0.0f}));
    assertFalse(Idefics3VisionTower.isZeroTile(new float[] {0.25f, 0.5f}));
    assertFalse(Idefics3VisionTower.isZeroTile(new float[] {Float.NaN}));
  }

  // --- mergeGatherIndices ------------------------------------------------------

  @Test
  void mergeGatherIndicesScatterFeatureRowsIntoImageTokens() {
    // masked_scatter semantics: the f-th feature row overwrites the f-th destination position,
    // every other position keeps its own embedding (no shifting of later positions).
    assertArrayEquals(
        new int[] {0, 6, 2, 3, 7, 5}, Idefics3VisionTower.mergeGatherIndices(6, List.of(1, 4)));
  }

  @Test
  void mergeGatherIndicesMatchTheGoldenExpansion() {
    // Case "one_image" of tools/hf-reference/goldens/idefics3.json: 46 expanded tokens with the
    // 20 image tokens at the recorded feature_destinations.
    int[] destinations =
        new int[] {2, 3, 4, 5, 8, 9, 10, 11, 15, 16, 17, 18, 21, 22, 23, 24, 29, 30, 31, 32};
    List<Integer> boxed = new ArrayList<>();
    for (int destination : destinations) {
      boxed.add(destination);
    }
    assertArrayEquals(
        new int[] {
          0, 1, 46, 47, 48, 49, 6, 7, 50, 51, 52, 53, 12, 13, 14, 54, 55, 56, 57, 19, 20, 58, 59,
          60, 61, 25, 26, 27, 28, 62, 63, 64, 65, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45
        },
        Idefics3VisionTower.mergeGatherIndices(46, boxed));
  }

  @Test
  void mergeGatherIndicesKeepNonImagePositionsAndCoverEachFeatureOnce() {
    int length = 9;
    List<Integer> destinations = List.of(0, 4, 8);
    int[] indices = Idefics3VisionTower.mergeGatherIndices(length, destinations);
    assertEquals(length, indices.length);
    for (int i = 0; i < length; i++) {
      boolean isDestination = destinations.contains(i);
      int expected = i;
      int feature = destinations.indexOf(i);
      if (isDestination) {
        expected = length + feature;
      }
      assertEquals(expected, indices[i], "position " + i);
    }
    // Every feature row is gathered exactly once and every index stays in range.
    for (int f = 0; f < destinations.size(); f++) {
      int count = 0;
      for (int index : indices) {
        if (index == length + f) {
          count++;
        }
        assertTrue(index >= 0 && index < length + destinations.size());
      }
      assertEquals(1, count, "feature row " + f);
    }
  }

  @Test
  void mergeGatherIndicesEmptyDestinationsAreTheIdentity() {
    assertArrayEquals(new int[] {0, 1, 2, 3}, Idefics3VisionTower.mergeGatherIndices(4, List.of()));
    assertArrayEquals(new int[] {}, Idefics3VisionTower.mergeGatherIndices(0, List.of()));
  }

  @Test
  void mergeGatherIndicesRejectInvalidDestinations() {
    assertThrows(
        IllegalArgumentException.class,
        () -> Idefics3VisionTower.mergeGatherIndices(-1, List.of()));
    assertThrows(
        IllegalArgumentException.class,
        () -> Idefics3VisionTower.mergeGatherIndices(4, List.of(-1)));
    assertThrows(
        IllegalArgumentException.class,
        () -> Idefics3VisionTower.mergeGatherIndices(4, List.of(4)));
    assertThrows(
        IllegalArgumentException.class,
        () -> Idefics3VisionTower.mergeGatherIndices(4, List.of(1, 1)));
    assertThrows(
        IllegalArgumentException.class,
        () -> Idefics3VisionTower.mergeGatherIndices(4, List.of(2, 1)));
    assertThrows(NullPointerException.class, () -> Idefics3VisionTower.mergeGatherIndices(4, null));
  }

  // --- pixelShuffleIndices -----------------------------------------------------

  @Test
  void pixelShuffleIndicesMatchTheReferenceReduction() {
    // (4, 4, 2, 64): the pinned fixture geometry. The venv-verified permutation of 0..1023
    // reproduces HF's flat output entry for entry; the recorded slices and spot entries.
    int[] perm = Idefics3Connector.pixelShuffleIndices(4, 4, 2, 64);
    assertEquals(1024, perm.length);
    for (int i = 0; i < 12; i++) {
      assertEquals(i, perm[i]);
      assertEquals(64 + i, perm[64 + i]);
      assertEquals(256 + i, perm[128 + i]);
      assertEquals(128 + i, perm[256 + i]);
    }
    assertEquals(1020, perm[1020]);
    assertEquals(1021, perm[1021]);
    assertEquals(1022, perm[1022]);
    assertEquals(1023, perm[1023]);
    assertEquals(63, perm[63]);
    assertEquals(256, perm[128]);
    assertEquals(383, perm[255]);
    assertEquals(128, perm[256]);
    assertEquals(1023, perm[1023]);
    // It is a true permutation: every source element appears exactly once.
    boolean[] seen = new boolean[1024];
    for (int source : perm) {
      assertTrue(source >= 0 && source < 1024, "source " + source);
      assertFalse(seen[source], "source " + source + " appears twice");
      seen[source] = true;
    }
  }

  @Test
  void pixelShuffleIndicesGeneralizeRectangularGrids() {
    // (4, 6, 2, 8): the venv-verified generalized view/permute chain (HF itself only reaches the
    // square case through its int(seq ** 0.5) assumption).
    int[] perm = Idefics3Connector.pixelShuffleIndices(4, 6, 2, 8);
    assertEquals(192, perm.length);
    boolean[] seen = new boolean[192];
    for (int source : perm) {
      assertTrue(source >= 0 && source < 192, "source " + source);
      assertFalse(seen[source], "source " + source + " appears twice");
      seen[source] = true;
    }
  }

  @Test
  void pixelShuffleIndicesRejectInvalidDimensions() {
    assertThrows(
        IllegalArgumentException.class, () -> Idefics3Connector.pixelShuffleIndices(4, 4, 3, 64));
    assertThrows(
        IllegalArgumentException.class, () -> Idefics3Connector.pixelShuffleIndices(3, 4, 2, 8));
    assertThrows(
        IllegalArgumentException.class, () -> Idefics3Connector.pixelShuffleIndices(4, 6, 4, 8));
    assertThrows(
        IllegalArgumentException.class, () -> Idefics3Connector.pixelShuffleIndices(4, 4, 2, 10));
    assertThrows(
        IllegalArgumentException.class, () -> Idefics3Connector.pixelShuffleIndices(0, 4, 2, 8));
    assertThrows(
        IllegalArgumentException.class, () -> Idefics3Connector.pixelShuffleIndices(4, 0, 2, 8));
    assertThrows(
        IllegalArgumentException.class, () -> Idefics3Connector.pixelShuffleIndices(4, 4, 0, 8));
    assertThrows(
        IllegalArgumentException.class, () -> Idefics3Connector.pixelShuffleIndices(4, 4, 2, 0));
  }
}
