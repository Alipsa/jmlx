/**
 * Tensor operations over Apple's MLX: {@link se.alipsa.jmlx.core.MLX} and its op facades, {@link
 * se.alipsa.jmlx.core.MLXArray}, autograd and checkpoint I/O.
 *
 * <h2>Numerical precision</h2>
 *
 * <p>MLX may use reduced-precision matrix kernels for {@code FLOAT32} operations on supported
 * hardware, while keeping input and output arrays in {@code FLOAT32}. For comparisons with full
 * float32 references, set the native environment variable {@code MLX_ENABLE_TF32=0} when launching
 * the Java or Groovy process. The pinned runtime defaults to {@code 1} and caches the setting on
 * first use; a Java system property does not select native precision. Choose the mode at process
 * startup and benchmark its performance on the application's actual model and shapes. Full float32
 * does not guarantee identical results across hardware or batch shapes.
 *
 * <h2>Threading rule</h2>
 *
 * <p>Use MLX from <b>at most one thread at a time per process</b>. MLX's native state -- the
 * default stream and the Metal device -- is process-wide, and its streams are bound to the thread
 * that created them, so concurrent MLX use from several threads is unsupported, not merely
 * unsynchronized. An {@link se.alipsa.jmlx.memory.MLXScope} and every {@link
 * se.alipsa.jmlx.core.MLXArray} allocated from it are confined to the thread that created the
 * scope; each scope carries that thread's own stream, so a different thread can be the single MLX
 * thread later, but only with scopes it created itself. To serve many callers from one process,
 * give them a single MLX-owning worker (the {@code BatchGenerationScheduler} in {@code jmlx-models}
 * is one) rather than sharing scopes.
 *
 * <p>The rule is per process. Library code can only enforce it within one jmlx classloader (the
 * scheduler rejects a second running scheduler), and the direct {@code generate} API cannot be
 * blocked, so mixing the direct path with a running scheduler is a documentation-only rule.
 *
 * <p><b>Per-thread cost.</b> Every thread that uses MLX directly carries its own scheduler stream:
 * the first root {@code MLXScope} a thread builds creates it (about 60 KB measured per stream), and
 * it is never freed -- mlx-c has no call that removes a stream from MLX's scheduler, and freeing
 * the handle would reclaim nothing but a few bytes. A process that churns MLX-using threads
 * therefore accumulates one stream per thread for its whole life: a thread-per-request server that
 * calls the direct {@code generate} API one request at a time keeps growing one stream per request
 * thread with no upper bound, and a scheduler that is closed and restarted adds one per worker
 * thread. Prefer one long-lived MLX thread (the scheduler's worker is the built-in shape) over many
 * short-lived ones.
 *
 * <p><b>Virtual threads.</b> Virtual threads cannot be MLX threads. The stream binding is to the OS
 * thread -- in the pinned mlx-metal 0.31.2, each stream's Metal command state is a {@code static
 * thread_local} in {@code mlx/backend/metal/device.cpp} -- while a Java virtual thread migrates
 * between carrier OS threads between calls. The Java-side confinement checks compare thread objects
 * and would keep passing after a migration, so the failure would surface, if at all, at a random
 * native call; creating a root {@code MLXScope} on a virtual thread is therefore rejected with
 * {@link IllegalStateException} up front.
 *
 * <p><b>Known exception.</b> The {@link java.lang.ref.Cleaner} backstops free native handles from
 * the JVM Cleaner thread when a scope or function was never closed: {@code MLXScope} calls {@code
 * mlx_array_free}, and {@code MLXGrad.Fn} calls {@code mlx_closure_free} and {@code
 * mlx_closure_value_and_grad_free}. Whether freeing from another thread is safe is the unresolved
 * "Is {@code mlx_array_free} safe to call from the Cleaner thread?" question in {@code
 * req/initial-plan.md} (Open questions); the closure frees are a related, separate case that
 * question does not cover. Do not rely on either backstop: always close scopes and functions.
 */
package se.alipsa.jmlx.core;
