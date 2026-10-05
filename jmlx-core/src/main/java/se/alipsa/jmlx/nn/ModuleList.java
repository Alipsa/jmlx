package se.alipsa.jmlx.nn;

import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import se.alipsa.jmlx.memory.MLXScope;

/** Fixed ordered children, registered with decimal parameter-path names. */
public final class ModuleList extends Module implements Iterable<Module> {
  private final List<Module> layers;

  /** Creates fixed membership; null children are rejected. */
  public ModuleList(MLXScope scope, Module... layers) {
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
  public Module get(int index) {
    return layers.get(index);
  }

  @Override
  public Iterator<Module> iterator() {
    return layers.iterator();
  }
}
