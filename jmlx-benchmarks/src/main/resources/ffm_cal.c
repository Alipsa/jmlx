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
 * Carrier fidelity: every opaque mlx-c struct (mlx_array, mlx_stream,
 * mlx_device, ...) is a single 8-byte pointer, and on AArch64 an 8-byte
 * struct is passed in a 64-bit register exactly like a void*; an 8-byte
 * struct return by value uses the same register convention. The no-op
 * shapes therefore measure the same carrier classes as the real bindings.
 *
 * Naming: cal_<param-keys>_<return-key>, where M = 8-byte carrier (a struct
 * value or a pointer), A = the struct-return allocator slot (Java side only;
 * no C parameter), I = int32, J = 64-bit integer, Z = _Bool, V = void.
 */
#include <stdbool.h>
#include <stdint.h>

/* The 8-byte struct mirror of every opaque mlx-c struct. */
struct cal_eight {
    void *ctx;
};

static volatile uint64_t cal_sink;

static uint64_t cal_fold(uint64_t value) {
    cal_sink ^= value + 0x9e3779b97f4a7c15ULL;
    return cal_sink;
}

/* Single-carrier shapes. */
int cal_M_I(void *a) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a) & 0xFFFFu);
}

long cal_M_J(void *a) {
    return (long)cal_fold((uint64_t)(uintptr_t)a);
}

bool cal_M_Z(void *a) {
    return (cal_fold((uint64_t)(uintptr_t)a) & 1u) != 0;
}

void cal_M_V(void *a) {
    cal_fold((uint64_t)(uintptr_t)a);
}

void *cal_M_M(void *a) {
    return (void *)(uintptr_t)(cal_fold((uint64_t)(uintptr_t)a) | 8u);
}

struct cal_eight cal_AM_M(void *a) {
    struct cal_eight result;
    result.ctx = (void *)(uintptr_t)(cal_fold((uint64_t)(uintptr_t)a) | 8u);
    return result;
}

/* Multi-carrier shapes. */
int cal_MM_I(void *a, void *b) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a ^ (uint64_t)(uintptr_t)b) & 0xFFFFu);
}

int cal_MMM_I(void *a, void *b, void *c) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a ^
                          (uint64_t)(uintptr_t)b ^
                          (uint64_t)(uintptr_t)c) &
                 0xFFFFu);
}

int cal_MMMM_I(void *a, void *b, void *c, void *d) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a ^
                          (uint64_t)(uintptr_t)b ^
                          (uint64_t)(uintptr_t)c ^
                          (uint64_t)(uintptr_t)d) &
                 0xFFFFu);
}

int cal_MMMMM_I(void *a, void *b, void *c, void *d, void *e) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a ^
                          (uint64_t)(uintptr_t)b ^
                          (uint64_t)(uintptr_t)c ^
                          (uint64_t)(uintptr_t)d ^
                          (uint64_t)(uintptr_t)e) &
                 0xFFFFu);
}

int cal_MI_I(void *a, int i) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a ^ (uint32_t)i) & 0xFFFFu);
}

int cal_MMIM_I(void *a, void *b, int i, void *c) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a ^
                          (uint64_t)(uintptr_t)b ^
                          (uint32_t)i ^
                          (uint64_t)(uintptr_t)c) &
                 0xFFFFu);
}

int cal_MMIIM_I(void *a, void *b, int i, int j, void *c) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a ^
                          (uint64_t)(uintptr_t)b ^
                          (uint32_t)i ^
                          (uint32_t)j ^
                          (uint64_t)(uintptr_t)c) &
                 0xFFFFu);
}

int cal_MZM_I(void *a, bool z, void *b) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a ^ (uint8_t)(z ? 1u : 0u) ^
                          (uint64_t)(uintptr_t)b) &
                 0xFFFFu);
}

int cal_MMMJM_I(void *a, void *b, void *c, long j, void *d) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a ^
                          (uint64_t)(uintptr_t)b ^
                          (uint64_t)(uintptr_t)c ^
                          (uint64_t)(uintptr_t)j ^
                          (uint64_t)(uintptr_t)d) &
                 0xFFFFu);
}

int cal_MMIMM_I(void *a, void *b, int i, void *c, void *d) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a ^
                          (uint64_t)(uintptr_t)b ^
                          (uint32_t)i ^
                          (uint64_t)(uintptr_t)c ^
                          (uint64_t)(uintptr_t)d) &
                 0xFFFFu);
}
