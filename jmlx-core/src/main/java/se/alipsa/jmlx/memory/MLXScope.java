package se.alipsa.jmlx.memory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.lang.ref.Cleaner;
import java.util.LinkedHashSet;
import se.alipsa.jmlx.ffi.NativeLoader;
import se.alipsa.jmlx.ffi.mlx_h;

/**
 * Owns the {@code mlx_array} native handles allocated on it and frees them in reverse insertion order on
 * {@link #close()}. See req/initial-plan.md §6 and req/phase4-plan.md §2.
 *
 * <p>
 * Thread-safety contract: an {@code MLXScope} and the {@code MLXArray}s it owns are confined to the thread that created
 * the scope. The sole exception is the JVM's {@link Cleaner} thread invoking the backstop below, which is why the
 * backstop touches only {@link Holder} and never the scope itself.
 *
 * <p>
 * req/phase4-plan.md §2: a scope may have a {@link #parent()}, formed via {@link #newChild()}. {@link #newChild()}
 * calls {@link #checkAccess()} on the parent first, so a child always has the same owner thread as its parent, by
 * construction -- ancestor-related scopes are therefore same-thread automatically, which is why the ancestor rule used
 * by op helpers (see {@code NativeOps.scopeOf} in {@code se.alipsa.jmlx.core}) is an ownership-comprehensibility
 * invariant, not a confinement guard.
 *
 * <p>
 * Implements {@link SegmentAllocator} so it can be passed directly to mlx-c's struct-returning constructors (e.g.
 * {@code mlx_array_new}, {@code mlx_array_new_data}), which allocate their result through whatever allocator they are
 * given. {@link #allocate} is reserved for that use: it assumes every segment handed out represents an
 * {@code mlx_array} struct and will later be passed to {@code mlx_array_free}. Passing a scope as an allocator for
 * anything else corrupts the handle list.
 */
public final class MLXScope implements AutoCloseable, SegmentAllocator {

  // Same reasoning as MLX's static initializer: jextract binds each
  // downcall's method handle lazily, the first time that function is
  // actually called, which fails unless the dylib is already loaded by
  // then. A scope can be constructed and used without ever touching the
  // MLX facade class, so it needs this guard independently.
  static {
    NativeLoader.ensureLoaded();
  }

  private static final Cleaner CLEANER = Cleaner.create();

  /**
   * Capture rule: the failure mode of this whole pattern is the cleanup action holding a reference path back to the
   * object it is meant to clean up after, which keeps that object permanently reachable and the action never runs. This
   * holder -- not {@code MLXScope} -- is what gets registered with the {@link Cleaner}, so the action below can only
   * ever reach {@code this} holder, never the owning scope.
   */
  private static final class Holder {
    private final Arena arena = Arena.ofShared();
    // LinkedHashSet, not a List: freeOne()'s removal must be O(1), not an
    // O(n) scan -- an explicit close of n arrays would otherwise be O(n^2).
    // Insertion order is preserved for closeAll()'s reverse-order free.
    private final LinkedHashSet<MemorySegment> handles = new LinkedHashSet<>();
    // req/phase4-plan.md §2: a child scope's Holder is added here by the
    // child's constructor (capturing only the child's Holder, never its
    // MLXScope -- same capture rule as `cleanable` below). closeAll()
    // cascades these in reverse insertion order before freeing this
    // Holder's own handles, so a Cleaner-driven cascade of an unreachable
    // parent still frees still-reachable children rather than leaking them.
    private final LinkedHashSet<Holder> children = new LinkedHashSet<>();
    // volatile, not a plain field guarded solely by synchronized methods:
    // MLXScope.ensureOpen() reads isClosed() on every MLXArray access (shape(),
    // dtype(), handle(), ...), several times per op, so a synchronized read
    // here would put a monitor acquisition on the hottest path in the facade.
    // An unsynchronized volatile read is the one option that is both correct
    // (visible across the Cleaner thread, which writes it under this
    // Holder's monitor) and free. volatile only strengthens the guarantees
    // the existing synchronized methods below already provide.
    private volatile boolean closed = false;

    synchronized MemorySegment allocate(long byteSize, long byteAlignment) {
      // No "if (!closed)" guard here: closeAll() closes this shared arena
      // before releasing the monitor, so once closed is true, the call
      // above already throws IllegalStateException -- this line is
      // unreachable with closed still true.
      MemorySegment seg = arena.allocate(byteSize, byteAlignment);
      handles.add(seg);
      return seg;
    }

    synchronized void addChild(Holder child) {
      children.add(child);
    }

    // Called only from MLXScope.close(), after THIS Holder's own closeAll()
    // has returned and released its monitor -- never from inside a Holder
    // method. The cascade below takes the parent's monitor and then each
    // child's (parent-then-child); calling this from inside closeAll()
    // would take child-then-parent on the owner thread while the Cleaner
    // thread can independently cascade a parent (unreachable MLXScope) and
    // take parent-then-child -- a classic ABBA deadlock. A cascading parent
    // never calls this either: it is discarding its whole child set anyway.
    synchronized void removeChild(Holder child) {
      children.remove(child);
    }

    synchronized void closeAll() {
      if (closed) {
        return;
      }
      closed = true;
      // Cascade children BEFORE freeing this Holder's own handles: an
      // ArrayDesc owns its graph inputs by value (std::vector<array>), and
      // each array is a shared_ptr<ArrayDesc>, so freeing a child's handles
      // only decrements a refcount -- it cannot touch a graph a hoisted
      // output still references (req/phase4-plan.md §2, Research findings).
      // Order relative to this Holder's own handles is otherwise immaterial
      // for that reason, but children-first is what makes a Cleaner-driven
      // cascade of an unreachable parent free every still-reachable child's
      // native memory rather than leaking it.
      Holder[] toClose = children.toArray(new Holder[0]);
      for (int i = toClose.length - 1; i >= 0; i--) {
        toClose[i].closeAll();
      }
      children.clear();
      // Open question (req/initial-plan.md, Open questions): whether
      // mlx_array_free is safe to call from the Cleaner thread rather
      // than the scope's owning thread. Not resolved by the §2 probe;
      // implemented as specified pending that determination -- see the
      // plan for the fallback (enqueue onto the owning thread instead)
      // if this turns out to be unsafe in practice.
      MemorySegment[] toFree = handles.toArray(new MemorySegment[0]);
      for (int i = toFree.length - 1; i >= 0; i--) {
        mlx_h.mlx_array_free(toFree[i]);
      }
      handles.clear();
      arena.close();
    }

    synchronized void freeOne(MemorySegment handle) {
      if (closed) {
        return;
      }
      if (handles.remove(handle)) {
        mlx_h.mlx_array_free(handle);
      }
    }

    // Unsynchronized on purpose -- see the field javadoc above.
    boolean isClosed() {
      return closed;
    }
  }

  private final Holder holder = new Holder();
  private final Cleaner.Cleanable cleanable = CLEANER.register(this, holder::closeAll);
  private final Thread owner = Thread.currentThread();
  private final MLXScope parent;
  private volatile boolean closed = false;

  /** Creates a root scope: {@link #parent()} is {@code null}. */
  public MLXScope() {
    this(null);
  }

  private MLXScope(MLXScope parent) {
    this.parent = parent;
    if (parent != null) {
      parent.holder.addChild(holder);
    }
  }

  /**
   * Creates a child of this scope. Calls {@link #checkAccess()} on this scope first, so the child is guaranteed to
   * share this scope's owner thread -- ancestor-related scopes are therefore same-thread automatically. Closing this
   * scope (or a further ancestor of it) closes the child too, via the cascade described on {@link Holder}.
   */
  public MLXScope newChild() {
    checkAccess();
    return new MLXScope(this);
  }

  /** This scope's parent, or {@code null} if it is a root. */
  public MLXScope parent() {
    return parent;
  }

  /** Distance from the nearest root: {@code 0} for a root, {@code parent().depth() + 1} otherwise. */
  public int depth() {
    return parent == null ? 0 : parent.depth() + 1;
  }

  /**
   * Whether {@code other} is this scope or a descendant of it. Reflexive: {@code x.isAncestorOf(x)} is {@code true} --
   * see {@link se.alipsa.jmlx.core.MLX#hoist} for why that matters.
   */
  public boolean isAncestorOf(MLXScope other) {
    for (MLXScope s = other; s != null; s = s.parent) {
      if (s == this) {
        return true;
      }
    }
    return false;
  }

  /**
   * The innermost (deepest) of {@code x} and {@code y}, which must be the same scope or one an ancestor of the other --
   * order-independent, unlike picking "the first operand's scope". {@link #depth()} makes the pick O(1); the ancestor
   * check below is the O(depth) part, and depth is at most about three in every realistic use (root -> model -> step),
   * so it is not worth caching.
   *
   * @throws IllegalArgumentException if neither is an ancestor of the other (siblings, or two independent roots)
   */
  public static MLXScope innermost(MLXScope x, MLXScope y) {
    if (x == y) {
      return x;
    }
    MLXScope deeper = x.depth() >= y.depth() ? x : y;
    MLXScope shallower = deeper == x ? y : x;
    if (!shallower.isAncestorOf(deeper)) {
      throw new IllegalArgumentException("MLXScope: unrelated scopes (neither is an ancestor of the other)");
    }
    return deeper;
  }

  @Override
  public MemorySegment allocate(long byteSize, long byteAlignment) {
    checkThread();
    ensureOpen();
    return holder.allocate(byteSize, byteAlignment);
  }

  /**
   * Frees one handle ahead of scope close, so an owner can release its native resource early without waiting for the
   * whole scope to close. Not for general use: {@code handle} must be a segment this scope itself handed out via
   * {@link #allocate}, or this is a no-op.
   */
  public void free(MemorySegment handle) {
    checkThread();
    holder.freeOne(handle);
  }

  /**
   * Internal cross-package hook for {@code MLXArray.ensureOpen()}: asserts that the calling thread is this scope's
   * owner and that the scope has not been closed. Not for general use -- {@code owner} and {@code closed} are private
   * to this class, so an {@link se.alipsa.jmlx.core.MLXArray} confined to this scope has no other way to see either.
   */
  public void checkAccess() {
    checkThread();
    ensureOpen();
  }

  // Consults holder.isClosed() too, not just this scope's own `closed` flag:
  // a parent cascade (Holder.closeAll(), triggered by an explicit close() on
  // an ancestor, or by the Cleaner backstop on an unreachable one) frees this
  // scope's handles without ever touching this MLXScope object -- flipping
  // `closed` here would be the capture the Holder/Cleaner split exists to
  // avoid. Without this, a cascaded-but-still-reachable child would read
  // closed == false and this method would let a use-after-free through
  // instead of throwing.
  private void ensureOpen() {
    if (closed || holder.isClosed()) {
      throw new IllegalStateException("MLXScope is closed");
    }
  }

  private void checkThread() {
    Thread current = Thread.currentThread();
    if (current != owner) {
      throw new IllegalStateException("MLXScope is confined to " + owner + " but was accessed from " + current);
    }
  }

  @Override
  public void close() {
    checkThread();
    closed = true;
    holder.closeAll();
    // After holder.closeAll() has returned and released its monitor, never
    // from inside a Holder method -- see removeChild's javadoc for the ABBA
    // deadlock that ordering avoids. A no-op if this is a root (parent ==
    // null) or if an ancestor's cascade already cleared this scope out of
    // its parent's children set.
    if (parent != null) {
      parent.holder.removeChild(holder);
    }
    cleanable.clean();
  }
}
