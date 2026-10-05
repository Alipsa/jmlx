package se.alipsa.jmlx.nn;

import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.memory.MLXScope;

/** Fixed ordered children, registered with decimal parameter-path names. */
public final class Sequential extends UnaryLayer implements Iterable<UnaryLayer> {
  private final List<UnaryLayer> layers;

  /** Creates fixed membership; null children are rejected. */
  public Sequential(MLXScope scope, UnaryLayer... layers) {
    super(scope);
    this.layers = List.of(layers.clone());
    for (int i = 0; i < this.layers.size(); i++) {
      child(Integer.toString(i), Objects.requireNonNull(this.layers.get(i)));
    }
  }

  /** Number of children. */
  public int size() {
    return layers.size();
  }

  /** Child at the supplied index. */
  public UnaryLayer get(int index) {
    return layers.get(index);
  }

  @Override
  public Iterator<UnaryLayer> iterator() {
    return layers.iterator();
  }

  @Override
  public MLXArray forward(MLXArray x) {
    for (UnaryLayer layer : layers) {
      x = layer.forward(x);
    }
    return x;
  }
}
