package se.alipsa.jmlx.memory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.lang.ref.Cleaner;
import java.util.ArrayList;
import java.util.List;
import se.alipsa.jmlx.ffi.NativeLoader;
import se.alipsa.jmlx.ffi.mlx_h;

/**
 * Owns the {@code mlx_array} native handles allocated on it and frees them in
 * reverse insertion order on {@link #close()}. See req/initial-plan.md §6.
 *
 * <p>Thread-safety contract: an {@code MLXScope} and the {@code MLXArray}s it
 * owns are confined to the thread that created the scope. The sole exception
 * is the JVM's {@link Cleaner} thread invoking the backstop below, which is
 * why the backstop touches only {@link Holder} and never the scope itself.
 *
 * <p>Implements {@link SegmentAllocator} so it can be passed directly to
 * mlx-c's struct-returning constructors (e.g. {@code mlx_array_new},
 * {@code mlx_array_new_data}), which allocate their result through whatever
 * allocator they are given. {@link #allocate} is reserved for that use: it
 * assumes every segment handed out represents an {@code mlx_array} struct
 * and will later be passed to {@code mlx_array_free}. Passing a scope as an
 * allocator for anything else corrupts the handle list.
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
     * Capture rule: the failure mode of this whole pattern is the cleanup
     * action holding a reference path back to the object it is meant to
     * clean up after, which keeps that object permanently reachable and the
     * action never runs. This holder -- not {@code MLXScope} -- is what gets
     * registered with the {@link Cleaner}, so the action below can only ever
     * reach {@code this} holder, never the owning scope.
     */
    private static final class Holder {
        private final Arena arena = Arena.ofShared();
        private final List<MemorySegment> handles = new ArrayList<>();
        private boolean closed = false;

        synchronized MemorySegment allocate(long byteSize, long byteAlignment) {
            MemorySegment seg = arena.allocate(byteSize, byteAlignment);
            if (!closed) {
                handles.add(seg);
            }
            return seg;
        }

        synchronized void closeAll() {
            if (closed) {
                return;
            }
            closed = true;
            // Open question (req/initial-plan.md, Open questions): whether
            // mlx_array_free is safe to call from the Cleaner thread rather
            // than the scope's owning thread. Not resolved by the §2 probe;
            // implemented as specified pending that determination -- see the
            // plan for the fallback (enqueue onto the owning thread instead)
            // if this turns out to be unsafe in practice.
            for (int i = handles.size() - 1; i >= 0; i--) {
                mlx_h.mlx_array_free(handles.get(i));
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
    }

    private final Holder holder = new Holder();
    private final Cleaner.Cleanable cleanable = CLEANER.register(this, holder::closeAll);
    private final Thread owner = Thread.currentThread();
    private volatile boolean closed = false;

    @Override
    public MemorySegment allocate(long byteSize, long byteAlignment) {
        checkThread();
        ensureOpen();
        return holder.allocate(byteSize, byteAlignment);
    }

    /**
     * Frees one handle ahead of scope close, so an owner can release its
     * native resource early without waiting for the whole scope to close.
     * Not for general use: {@code handle} must be a segment this scope
     * itself handed out via {@link #allocate}, or this is a no-op.
     */
    public void free(MemorySegment handle) {
        checkThread();
        holder.freeOne(handle);
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("MLXScope is closed");
        }
    }

    private void checkThread() {
        Thread current = Thread.currentThread();
        if (current != owner) {
            throw new IllegalStateException(
                "MLXScope is confined to " + owner + " but was accessed from " + current);
        }
    }

    @Override
    public void close() {
        checkThread();
        closed = true;
        holder.closeAll();
        cleanable.clean();
    }
}
