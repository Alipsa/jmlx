package se.alipsa.jmlx.vision;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class ImageTensorTest {

  @Test
  void constructorRejectsNullArguments() {
    assertThrows(
        IllegalArgumentException.class,
        () -> new ImageTensor(null, new int[] {1, 2, 3, 3}, Layout.NHWC));
    assertThrows(
        IllegalArgumentException.class, () -> new ImageTensor(new float[18], null, Layout.NHWC));
    assertThrows(
        IllegalArgumentException.class,
        () -> new ImageTensor(new float[18], new int[] {1, 2, 3, 3}, null));
  }

  @Test
  void constructorEnforcesLayoutShapeContracts() {
    // HWC must be exactly [height, width, 3].
    assertThrows(
        IllegalArgumentException.class,
        () -> new ImageTensor(new float[6], new int[] {2, 3}, Layout.HWC));
    assertThrows(
        IllegalArgumentException.class,
        () -> new ImageTensor(new float[12], new int[] {2, 2, 4}, Layout.HWC));
    assertThrows(
        IllegalArgumentException.class,
        () -> new ImageTensor(new float[12], new int[] {2, 2, 3, 1}, Layout.HWC));
    // NHWC must be exactly [1, height, width, 3].
    assertThrows(
        IllegalArgumentException.class,
        () -> new ImageTensor(new float[18], new int[] {1, 2, 3}, Layout.NHWC));
    assertThrows(
        IllegalArgumentException.class,
        () -> new ImageTensor(new float[18], new int[] {2, 2, 3, 3}, Layout.NHWC));
    assertThrows(
        IllegalArgumentException.class,
        () -> new ImageTensor(new float[18], new int[] {1, 2, 2, 4}, Layout.NHWC));
    // Positive dimensions and an exact element count.
    assertThrows(
        IllegalArgumentException.class,
        () -> new ImageTensor(new float[0], new int[] {0, 2, 3}, Layout.HWC));
    assertThrows(
        IllegalArgumentException.class,
        () -> new ImageTensor(new float[11], new int[] {2, 2, 3}, Layout.HWC));
    assertThrows(
        IllegalArgumentException.class,
        () -> new ImageTensor(new float[19], new int[] {1, 2, 3, 3}, Layout.NHWC));
  }

  @Test
  void validHwcAndNhwcConstructions() {
    ImageTensor hwc = new ImageTensor(new float[6], new int[] {2, 1, 3}, Layout.HWC);
    assertArrayEquals(new int[] {2, 1, 3}, hwc.shape());
    assertEquals(Layout.HWC, hwc.layout());
    ImageTensor nhwc = new ImageTensor(new float[18], new int[] {1, 2, 3, 3}, Layout.NHWC);
    assertArrayEquals(new int[] {1, 2, 3, 3}, nhwc.shape());
    assertEquals(Layout.NHWC, nhwc.layout());
  }

  @Test
  void accessorsReturnDefensiveCopies() {
    float[] values = new float[] {1, 2, 3};
    int[] shape = new int[] {1, 1, 1, 3};
    ImageTensor tensor = new ImageTensor(values, shape, Layout.NHWC);
    values[0] = 99;
    shape[0] = 7;
    assertEquals(1f, tensor.values()[0]);
    assertEquals(1, tensor.shape()[0]);
    float[] view = tensor.values();
    view[0] = 77;
    assertEquals(1f, tensor.values()[0]);
    int[] shapeView = tensor.shape();
    shapeView[1] = 9;
    assertEquals(1, tensor.shape()[1]);
  }

  @Test
  void copyValuesToBulkCopies() {
    ImageTensor tensor =
        new ImageTensor(new float[] {1, 2, 3, 4, 5, 6}, new int[] {1, 1, 2, 3}, Layout.NHWC);
    float[] destination = new float[10];
    tensor.copyValuesTo(destination, 3);
    assertArrayEquals(new float[] {0, 0, 0, 1, 2, 3, 4, 5, 6, 0}, destination);
    assertThrows(IllegalArgumentException.class, () -> tensor.copyValuesTo(null, 0));
    assertThrows(IndexOutOfBoundsException.class, () -> tensor.copyValuesTo(destination, -1));
    assertThrows(IndexOutOfBoundsException.class, () -> tensor.copyValuesTo(destination, 7));
    assertThrows(IndexOutOfBoundsException.class, () -> tensor.copyValuesTo(new float[5], 0));
  }

  @Test
  void equalityComparesLayoutShapeAndValues() {
    ImageTensor a = new ImageTensor(new float[] {1, 2, 3}, new int[] {1, 1, 1, 3}, Layout.NHWC);
    ImageTensor b = new ImageTensor(new float[] {1, 2, 3}, new int[] {1, 1, 1, 3}, Layout.NHWC);
    assertEquals(a, b);
    assertEquals(a.hashCode(), b.hashCode());
    assertNotEquals(a, new ImageTensor(new float[] {1, 2, 3}, new int[] {1, 1, 3}, Layout.HWC));
    assertNotEquals(a, new ImageTensor(new float[] {1, 2, 4}, new int[] {1, 1, 1, 3}, Layout.NHWC));
    assertNotEquals(
        a, new ImageTensor(new float[] {1, 2, 3, 4, 5, 6}, new int[] {1, 1, 2, 3}, Layout.NHWC));
    assertNotEquals(a, "tensor");
    assertFalse(a.equals(null));
  }

  @Test
  void toStringPrintsBoundedSummaryNeverValues() {
    ImageTensor tensor =
        new ImageTensor(
            new float[] {0.1234567f, 0.1234567f, 0.1234567f}, new int[] {1, 1, 1, 3}, Layout.NHWC);
    String text = tensor.toString();
    assertEquals("ImageTensor{layout=NHWC, shape=[1, 1, 1, 3], elements=3}", text);
    assertFalse(text.contains("0.1234567"));
  }
}
