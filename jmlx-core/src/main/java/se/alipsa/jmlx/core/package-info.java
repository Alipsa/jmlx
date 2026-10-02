/**
 * Tensor operations over Apple's MLX: {@link se.alipsa.jmlx.core.MLX} and its op facades, {@link
 * se.alipsa.jmlx.core.MLXArray}, autograd and checkpoint I/O.
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
 * <p><b>Known exception.</b> The {@link java.lang.ref.Cleaner} backstops free native handles from
 * the JVM Cleaner thread when a scope or function was never closed: {@code MLXScope} calls {@code
 * mlx_array_free}, and {@code MLXGrad.Fn} calls {@code mlx_closure_free} and {@code
 * mlx_closure_value_and_grad_free}. Whether freeing from another thread is safe is the unresolved
 * "Is {@code mlx_array_free} safe to call from the Cleaner thread?" question in {@code
 * req/initial-plan.md} (Open questions); the closure frees are a related, separate case that
 * question does not cover. Do not rely on either backstop: always close scopes and functions.
 */
package se.alipsa.jmlx.core;
