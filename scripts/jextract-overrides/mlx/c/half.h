/* jextract-time override of mlx-c's mlx/c/half.h.
 *
 * v0.1 supports only float32/int32 (see req/initial-plan.md, "Out of scope for
 * v0.1"). The real half.h's HAS_BFLOAT16 block guards `typedef __bf16
 * bfloat16_t;` with `defined(__ARM_FEATURE_BF16) || defined(__aarch64__)` --
 * which is *always* true on this target -- and jextract's Clang frontend
 * hard-errors on `__bf16` ("not supported on this target"), producing **zero**
 * output for the entire umbrella-header run, not a per-symbol skip.
 *
 * Dropping the HAS_BFLOAT16 block here means every mlx-c declaration gated on
 * `#ifdef HAS_BFLOAT16` (mlx_array_item_bfloat16, mlx_array_data_bfloat16)
 * cleanly disappears instead of erroring -- exactly the outcome we want for a
 * dtype v0.1 doesn't support anyway.
 *
 * HAS_FLOAT16 / __fp16 is kept: it parses fine under jextract and is merely
 * skipped from Java wrapping (unsupported for direct mapping), which is a
 * per-symbol skip, not a fatal error.
 *
 * regen-bindings.sh copies this file over the installed
 * native/install/include/mlx/c/half.h before invoking jextract, then restores
 * the original afterward so the installed header tree stays byte-for-byte
 * what bootstrap-native.sh actually staged.
 */
#ifndef MLX_HALF_H
#define MLX_HALF_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || defined(__aarch64__)
#define HAS_FLOAT16
#include <arm_fp16.h>
typedef __fp16 float16_t;
#endif

#ifdef __cplusplus
}
#endif

#endif
