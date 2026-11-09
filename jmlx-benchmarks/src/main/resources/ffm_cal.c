/*
 * Phase 8.2 WP2: FFM downcall-shape calibration library (plan section 5.1).
 *
 * Every function below has the same parameter/return register shape as a
 * representative class of the generated mlx-c bindings (the shape keys are
 * listed in FfmCalWorker), but the body does no work: it folds the argument
 * values into a volatile sink (an observable side effect, so -O2 cannot
 * eliminate the call) and never dereferences an input pointer. A measurement
 * of one of these calls therefore captures the FFM transition itself
 * (method-handle dispatch, argument marshalling, return handling) plus a few
 * ALU operations, and it does NOT include the real mlx-c's C-side work
 * (Metal dispatch, kernel launches, synchronization, allocator work).
 *
 * Carrier fidelity: in the pinned mlx-c header every opaque struct
 * (mlx_array, mlx_stream, mlx_device, ...) is passed by VALUE, and jextract
 * maps such a parameter to the struct's 8-byte layout, not to a pointer. On
 * AArch64 an 8-byte struct parameter is carried in one 64-bit register,
 * exactly like a void*, so the C ABI is identical; the Java side is what this
 * by-value form mirrors (the 8-byte struct read from the segment, matching
 * the generated wrappers' marshalling). The struct-return shapes use the same
 * register convention (an 8-byte struct return by value).
 *
 * Naming: cal_<param-keys>_<return-key>, where M = 8-byte struct carrier
 * (by value), A = the struct-return allocator slot (Java side only; no C
 * parameter), I = int32, J = 64-bit integer, Z = _Bool, V = void.
 */
#include <stdbool.h>
#include <stdint.h>

/* The 8-byte struct mirror of every opaque mlx-c struct (by value). */
struct cal_eight {
    void *ctx;
};

static volatile uint64_t cal_sink;

static uint64_t cal_fold(uint64_t value) {
    cal_sink ^= value + 0x9e3779b97f4a7c15ULL;
    return cal_sink;
}

/* Single-carrier shapes. */
int cal_M_I(struct cal_eight a) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a.ctx) & 0xFFFFu);
}

long cal_M_J(struct cal_eight a) {
    return (long)cal_fold((uint64_t)(uintptr_t)a.ctx);
}

bool cal_M_Z(struct cal_eight a) {
    return (cal_fold((uint64_t)(uintptr_t)a.ctx) & 1u) != 0;
}

void cal_M_V(struct cal_eight a) {
    cal_fold((uint64_t)(uintptr_t)a.ctx);
}

void *cal_M_M(struct cal_eight a) {
    return (void *)(uintptr_t)(cal_fold((uint64_t)(uintptr_t)a.ctx) | 8u);
}

struct cal_eight cal_AM_M(struct cal_eight a) {
    struct cal_eight result;
    result.ctx = (void *)(uintptr_t)(cal_fold((uint64_t)(uintptr_t)a.ctx) | 8u);
    return result;
}

/* Multi-carrier shapes. */
int cal_MM_I(struct cal_eight a, struct cal_eight b) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a.ctx ^ (uint64_t)(uintptr_t)b.ctx) & 0xFFFFu);
}

int cal_MMM_I(struct cal_eight a, struct cal_eight b, struct cal_eight c) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a.ctx ^
                          (uint64_t)(uintptr_t)b.ctx ^
                          (uint64_t)(uintptr_t)c.ctx) &
                 0xFFFFu);
}

int cal_MMMM_I(struct cal_eight a, struct cal_eight b,
               struct cal_eight c, struct cal_eight d) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a.ctx ^
                          (uint64_t)(uintptr_t)b.ctx ^
                          (uint64_t)(uintptr_t)c.ctx ^
                          (uint64_t)(uintptr_t)d.ctx) &
                 0xFFFFu);
}

int cal_MMMMM_I(struct cal_eight a, struct cal_eight b, struct cal_eight c,
                struct cal_eight d, struct cal_eight e) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a.ctx ^
                          (uint64_t)(uintptr_t)b.ctx ^
                          (uint64_t)(uintptr_t)c.ctx ^
                          (uint64_t)(uintptr_t)d.ctx ^
                          (uint64_t)(uintptr_t)e.ctx) &
                 0xFFFFu);
}

int cal_MI_I(struct cal_eight a, int i) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a.ctx ^ (uint32_t)i) & 0xFFFFu);
}

int cal_MMIM_I(struct cal_eight a, struct cal_eight b, int i, struct cal_eight c) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a.ctx ^
                          (uint64_t)(uintptr_t)b.ctx ^
                          (uint32_t)i ^
                          (uint64_t)(uintptr_t)c.ctx) &
                 0xFFFFu);
}

int cal_MMIIM_I(struct cal_eight a, struct cal_eight b, int i, int j,
                struct cal_eight c) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a.ctx ^
                          (uint64_t)(uintptr_t)b.ctx ^
                          (uint32_t)i ^
                          (uint32_t)j ^
                          (uint64_t)(uintptr_t)c.ctx) &
                 0xFFFFu);
}

int cal_MZM_I(struct cal_eight a, bool z, struct cal_eight b) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a.ctx ^
                          (uint8_t)(z ? 1u : 0u) ^
                          (uint64_t)(uintptr_t)b.ctx) &
                 0xFFFFu);
}

int cal_MMMJM_I(struct cal_eight a, struct cal_eight b, struct cal_eight c,
                long j, struct cal_eight d) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a.ctx ^
                          (uint64_t)(uintptr_t)b.ctx ^
                          (uint64_t)(uintptr_t)c.ctx ^
                          (uint64_t)(uintptr_t)j ^
                          (uint64_t)(uintptr_t)d.ctx) &
                 0xFFFFu);
}

int cal_MMIMM_I(struct cal_eight a, struct cal_eight b, int i,
                struct cal_eight c, struct cal_eight d) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a.ctx ^
                          (uint64_t)(uintptr_t)b.ctx ^
                          (uint32_t)i ^
                          (uint64_t)(uintptr_t)c.ctx ^
                          (uint64_t)(uintptr_t)d.ctx) &
                 0xFFFFu);
}
