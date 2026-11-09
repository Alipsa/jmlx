/*
 * Phase 8.2 WP2: FFM downcall-shape calibration library (plan section 5.1).
 *
 * Every function below has the same parameter/return shape as a class of the
 * generated mlx-c bindings (the shape keys are listed in FfmCalWorker), but the
 * body does no work: it folds the argument values into a volatile sink (an
 * observable side effect, so -O2 cannot eliminate the call) and never
 * dereferences an input pointer. A measurement of one of these calls therefore
 * captures the FFM transition itself (method-handle dispatch, argument
 * marshalling, return handling) plus a few ALU operations, and it does NOT
 * include the real mlx-c's C-side work (Metal dispatch, kernel launches,
 * synchronization, allocator work).
 *
 * Key set (review round 2, A): the 27 keys are the union of the native
 * descriptor classes of the bindings actually called by the controlled
 * binding-count workloads (TinyMistral-6x248M, Qwen3-0.6B, Qwen3-8B-4bit; the
 * derivation is recorded in findings section 15.4), replacing the round-1
 * representative-class set with the classes the estimate actually consumes.
 *
 * Carrier fidelity: in the pinned mlx-c header (a26f2d5, v0.8.0) every
 * by-value struct is 8 bytes and passed by value: the opaque handles
 * (mlx_array, mlx_stream, mlx_vector_*) are a single 8-byte pointer member,
 * and the mlx_optional_* scalar wrappers are a value/has-value pair padded
 * to 8 bytes. jextract maps such a parameter to the struct's 8-byte layout,
 * not to a pointer. On AArch64 an 8-byte struct parameter is carried in one
 * 64-bit register, exactly like a void*, so the C ABI of the M slots
 * (declared struct cal_eight below) is identical to the real one; the Java
 * side is what the by-value form mirrors (the 8-byte struct read from the
 * segment, matching the generated wrappers' marshalling). The struct-return
 * shapes use the same register convention (an 8-byte struct return by value).
 *
 * Naming: cal_<param-keys>_<return-key>, where P = void* parameter (the C
 * form of the J's a8 pointer slots), M = 8-byte struct carrier (by value),
 * I = int32, J = 64-bit integer, F = float, D = double, Z = _Bool, V = void.
 * The one no-parameter shape is cal__M (empty param list).
 */
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

/* The 8-byte struct mirror of every opaque mlx-c struct (by value). */
struct cal_eight {
    void *ctx;
};

static volatile uint64_t cal_sink;

/* Pointer returns fold into the sink and then hand back a static dummy
 * (non-null, constant): the real pointer-carrying shapes return a live
 * handle, and the Java side only needs the value to be a plausible pointer. */
static char cal_dummy;

static uint64_t cal_fold(uint64_t value) {
    cal_sink ^= value + 0x9e3779b97f4a7c15ULL;
    return cal_sink;
}

static uint64_t cal_foldf(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof bits);
    return cal_fold(bits);
}

static uint64_t cal_foldd(double d) {
    uint64_t bits;
    memcpy(&bits, &d, sizeof bits);
    return cal_fold(bits);
}

/* (M)I — e.g. mlx_array_free, mlx_array_eval, mlx_eval. */
int cal_M_I(struct cal_eight a) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a.ctx) & 0xFFFFu);
}

/* (M)J — e.g. mlx_array_ndim, mlx_array_size. */
long cal_M_J(struct cal_eight a) {
    return (long)cal_fold((uint64_t)(uintptr_t)a.ctx);
}

/* (M)P — e.g. mlx_array_shape, mlx_array_data_int32. */
void *cal_M_P(struct cal_eight a) {
    cal_fold((uint64_t)(uintptr_t)a.ctx);
    return &cal_dummy;
}

/* (PM)I — e.g. mlx_array_set. */
int cal_P_M_I(void *a, struct cal_eight b) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a ^ (uint64_t)(uintptr_t)b.ctx) &
                  0xFFFFu);
}

/* (PMM)I — e.g. mlx_transpose, mlx_sigmoid, mlx_stop_gradient. */
int cal_P_M_M_I(void *a, struct cal_eight b, struct cal_eight c) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a ^
                           (uint64_t)(uintptr_t)b.ctx ^
                           (uint64_t)(uintptr_t)c.ctx) &
                  0xFFFFu);
}

/* (PMMM)I — e.g. mlx_add, mlx_divide, mlx_equal, mlx_matmul. */
int cal_P_M_M_M_I(void *a, struct cal_eight b, struct cal_eight c,
                  struct cal_eight d) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a ^
                           (uint64_t)(uintptr_t)b.ctx ^
                           (uint64_t)(uintptr_t)c.ctx ^
                           (uint64_t)(uintptr_t)d.ctx) &
                  0xFFFFu);
}

/* (PMMMM)I — e.g. mlx_where. */
int cal_P_M_M_M_M_I(void *a, struct cal_eight b, struct cal_eight c,
                    struct cal_eight d, struct cal_eight e) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a ^
                           (uint64_t)(uintptr_t)b.ctx ^
                           (uint64_t)(uintptr_t)c.ctx ^
                           (uint64_t)(uintptr_t)d.ctx ^
                           (uint64_t)(uintptr_t)e.ctx) &
                  0xFFFFu);
}

/* (PMMMMZM)I — e.g. mlx_gather_mm. */
int cal_P_M_M_M_M_Z_M_I(void *a, struct cal_eight b, struct cal_eight c,
                        struct cal_eight d, struct cal_eight e, bool z,
                        struct cal_eight f) {
    uint64_t x = (uint64_t)(uintptr_t)a;
    x ^= (uint64_t)(uintptr_t)b.ctx;
    x ^= (uint64_t)(uintptr_t)c.ctx;
    x ^= (uint64_t)(uintptr_t)d.ctx;
    x ^= (uint64_t)(uintptr_t)e.ctx;
    x ^= (uint64_t)(z ? 1u : 0u);
    x ^= (uint64_t)(uintptr_t)f.ctx;
    return (int)(cal_fold(x) & 0xFFFFu);
}

/* (PMMMMZMMPM)I — e.g. mlx_quantized_matmul. */
int cal_P_M_M_M_M_Z_M_M_P_M_I(
    void *a, struct cal_eight b, struct cal_eight c, struct cal_eight d,
    struct cal_eight e, bool z, struct cal_eight f, struct cal_eight g,
    void *h, struct cal_eight i) {
    uint64_t x = (uint64_t)(uintptr_t)a;
    x ^= (uint64_t)(uintptr_t)b.ctx;
    x ^= (uint64_t)(uintptr_t)c.ctx;
    x ^= (uint64_t)(uintptr_t)d.ctx;
    x ^= (uint64_t)(uintptr_t)e.ctx;
    x ^= (uint64_t)(z ? 1u : 0u);
    x ^= (uint64_t)(uintptr_t)f.ctx;
    x ^= (uint64_t)(uintptr_t)g.ctx;
    x ^= (uint64_t)(uintptr_t)h;
    x ^= (uint64_t)(uintptr_t)i.ctx;
    return (int)(cal_fold(x) & 0xFFFFu);
}

/* (PMMMMMPMMM)I — e.g. mlx_dequantize. */
int cal_P_M_M_M_M_M_P_M_M_M_I(
    void *a, struct cal_eight b, struct cal_eight c, struct cal_eight d,
    struct cal_eight e, struct cal_eight f, void *g, struct cal_eight h,
    struct cal_eight i, struct cal_eight j) {
    uint64_t x = (uint64_t)(uintptr_t)a;
    x ^= (uint64_t)(uintptr_t)b.ctx;
    x ^= (uint64_t)(uintptr_t)c.ctx;
    x ^= (uint64_t)(uintptr_t)d.ctx;
    x ^= (uint64_t)(uintptr_t)e.ctx;
    x ^= (uint64_t)(uintptr_t)f.ctx;
    x ^= (uint64_t)(uintptr_t)g;
    x ^= (uint64_t)(uintptr_t)h.ctx;
    x ^= (uint64_t)(uintptr_t)i.ctx;
    x ^= (uint64_t)(uintptr_t)j.ctx;
    return (int)(cal_fold(x) & 0xFFFFu);
}

/* (PMZM)I — e.g. mlx_contiguous. */
int cal_P_M_Z_M_I(void *a, struct cal_eight b, bool z, struct cal_eight c) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a ^
                           (uint64_t)(uintptr_t)b.ctx ^
                           (uint64_t)(z ? 1u : 0u) ^
                           (uint64_t)(uintptr_t)c.ctx) &
                  0xFFFFu);
}

/* (PMIZM)I — e.g. mlx_argmax_axis, mlx_softmax_axis. */
int cal_P_M_I_Z_M_I(void *a, struct cal_eight b, int i, bool z,
                    struct cal_eight c) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a ^
                           (uint64_t)(uintptr_t)b.ctx ^
                           (uint64_t)(uint32_t)i ^
                           (uint64_t)(z ? 1u : 0u) ^
                           (uint64_t)(uintptr_t)c.ctx) &
                  0xFFFFu);
}

/* (PMIZMFIMM)I — e.g. mlx_fast_rope. */
int cal_P_M_I_Z_M_F_I_M_M_I(
    void *a, struct cal_eight b, int i, bool z, struct cal_eight c, float f,
    int j, struct cal_eight d, struct cal_eight e) {
    uint64_t x = (uint64_t)(uintptr_t)a;
    x ^= (uint64_t)(uintptr_t)b.ctx;
    x ^= (uint64_t)(uint32_t)i;
    x ^= (uint64_t)(z ? 1u : 0u);
    x ^= (uint64_t)(uintptr_t)c.ctx;
    {
        uint32_t fb;
        memcpy(&fb, &f, sizeof fb);
        x ^= (uint64_t)fb;
    }
    x ^= (uint64_t)(uint32_t)j;
    x ^= (uint64_t)(uintptr_t)d.ctx;
    x ^= (uint64_t)(uintptr_t)e.ctx;
    return (int)(cal_fold(x) & 0xFFFFu);
}

/* (PMIIM)I — e.g. mlx_flatten, mlx_swapaxes. */
int cal_P_M_I_I_M_I(void *a, struct cal_eight b, int i, int j,
                    struct cal_eight c) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a ^
                           (uint64_t)(uintptr_t)b.ctx ^
                           (uint64_t)(uint32_t)i ^
                           (uint64_t)(uint32_t)j ^
                           (uint64_t)(uintptr_t)c.ctx) &
                  0xFFFFu);
}

/* (PMIM)I — e.g. mlx_argsort_axis, mlx_astype, mlx_concatenate_axis. */
int cal_P_M_I_M_I(void *a, struct cal_eight b, int i, struct cal_eight c) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a ^
                           (uint64_t)(uintptr_t)b.ctx ^
                           (uint64_t)(uint32_t)i ^
                           (uint64_t)(uintptr_t)c.ctx) &
                  0xFFFFu);
}

/* (PMMIM)I — e.g. mlx_take_axis, mlx_take_along_axis. */
int cal_P_M_M_I_M_I(void *a, struct cal_eight b, struct cal_eight c, int i,
                    struct cal_eight d) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a ^
                           (uint64_t)(uintptr_t)b.ctx ^
                           (uint64_t)(uintptr_t)c.ctx ^
                           (uint64_t)(uint32_t)i ^
                           (uint64_t)(uintptr_t)d.ctx) &
                  0xFFFFu);
}

/* (PMMFM)I — e.g. mlx_fast_rms_norm. */
int cal_P_M_M_F_M_I(void *a, struct cal_eight b, struct cal_eight c, float f,
                    struct cal_eight d) {
    uint64_t x = (uint64_t)(uintptr_t)a;
    x ^= (uint64_t)(uintptr_t)b.ctx;
    x ^= (uint64_t)(uintptr_t)c.ctx;
    {
        uint32_t fb;
        memcpy(&fb, &f, sizeof fb);
        x ^= (uint64_t)fb;
    }
    x ^= (uint64_t)(uintptr_t)d.ctx;
    return (int)(cal_fold(x) & 0xFFFFu);
}

/* (PMMMFPMMZM)I — e.g. mlx_fast_scaled_dot_product_attention. */
int cal_P_M_M_M_F_P_M_M_Z_M_I(
    void *a, struct cal_eight b, struct cal_eight c, struct cal_eight d,
    float f, void *e, struct cal_eight g, struct cal_eight h, bool z,
    struct cal_eight i) {
    uint64_t x = (uint64_t)(uintptr_t)a;
    x ^= (uint64_t)(uintptr_t)b.ctx;
    x ^= (uint64_t)(uintptr_t)c.ctx;
    x ^= (uint64_t)(uintptr_t)d.ctx;
    {
        uint32_t fb;
        memcpy(&fb, &f, sizeof fb);
        x ^= (uint64_t)fb;
    }
    x ^= (uint64_t)(uintptr_t)e;
    x ^= (uint64_t)(uintptr_t)g.ctx;
    x ^= (uint64_t)(uintptr_t)h.ctx;
    x ^= (uint64_t)(z ? 1u : 0u);
    x ^= (uint64_t)(uintptr_t)i.ctx;
    return (int)(cal_fold(x) & 0xFFFFu);
}

/* (PMPJM)I — e.g. mlx_reshape, mlx_broadcast_to, mlx_transpose_axes. */
int cal_P_M_P_J_M_I(void *a, struct cal_eight b, void *c, long j,
                    struct cal_eight d) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a ^
                           (uint64_t)(uintptr_t)b.ctx ^
                           (uint64_t)(uintptr_t)c ^
                           (uint64_t)(uintptr_t)j ^
                           (uint64_t)(uintptr_t)d.ctx) &
                  0xFFFFu);
}

/* (PMPJZM)I — e.g. mlx_sum_axes. */
int cal_P_M_P_J_Z_M_I(void *a, struct cal_eight b, void *c, long j, bool z,
                      struct cal_eight d) {
    uint64_t x = (uint64_t)(uintptr_t)a;
    x ^= (uint64_t)(uintptr_t)b.ctx;
    x ^= (uint64_t)(uintptr_t)c;
    x ^= (uint64_t)(uintptr_t)j;
    x ^= (uint64_t)(z ? 1u : 0u);
    x ^= (uint64_t)(uintptr_t)d.ctx;
    return (int)(cal_fold(x) & 0xFFFFu);
}

/* (PMPJPJPJM)I — e.g. mlx_slice. */
int cal_P_M_P_J_P_J_P_J_M_I(
    void *a, struct cal_eight b, void *c, long j, void *d, long k, void *e,
    long l, struct cal_eight f) {
    uint64_t x = (uint64_t)(uintptr_t)a;
    x ^= (uint64_t)(uintptr_t)b.ctx;
    x ^= (uint64_t)(uintptr_t)c;
    x ^= (uint64_t)(uintptr_t)j;
    x ^= (uint64_t)(uintptr_t)d;
    x ^= (uint64_t)(uintptr_t)k;
    x ^= (uint64_t)(uintptr_t)e;
    x ^= (uint64_t)(uintptr_t)l;
    x ^= (uint64_t)(uintptr_t)f.ctx;
    return (int)(cal_fold(x) & 0xFFFFu);
}

/* (PDDDIM)I — e.g. mlx_arange. */
int cal_P_D_D_D_I_M_I(void *a, double d1, double d2, double d3, int i,
                      struct cal_eight b) {
    uint64_t x = (uint64_t)(uintptr_t)a;
    {
        uint64_t bits;
        memcpy(&bits, &d1, sizeof bits);
        x ^= bits;
    }
    {
        uint64_t bits;
        memcpy(&bits, &d2, sizeof bits);
        x ^= bits;
    }
    {
        uint64_t bits;
        memcpy(&bits, &d3, sizeof bits);
        x ^= bits;
    }
    x ^= (uint64_t)(uint32_t)i;
    x ^= (uint64_t)(uintptr_t)b.ctx;
    return (int)(cal_fold(x) & 0xFFFFu);
}

/* (PPJMIM)I — e.g. mlx_full. */
int cal_P_P_J_M_I_M_I(void *a, void *b, long j, struct cal_eight c, int i,
                      struct cal_eight d) {
    return (int)(cal_fold((uint64_t)(uintptr_t)a ^
                           (uint64_t)(uintptr_t)b ^
                           (uint64_t)(uintptr_t)j ^
                           (uint64_t)(uintptr_t)c.ctx ^
                           (uint64_t)(uint32_t)i ^
                           (uint64_t)(uintptr_t)d.ctx) &
                  0xFFFFu);
}

/* (F)M — e.g. mlx_array_new_float32. */
struct cal_eight cal_F_M(float f) {
    struct cal_eight result;
    result.ctx = (void *)(uintptr_t)(cal_foldf(f) | 8u);
    return result;
}

/* ()M — e.g. mlx_array_new. */
struct cal_eight cal__M(void) {
    struct cal_eight result;
    result.ctx = (void *)(uintptr_t)(cal_fold(0x9e3779b97f4a7c15ULL) | 8u);
    return result;
}

/* (PJ)M — e.g. mlx_vector_array_new_data. */
struct cal_eight cal_P_J_M(void *a, long j) {
    struct cal_eight result;
    result.ctx =
        (void *)(uintptr_t)(cal_fold((uint64_t)(uintptr_t)a ^
                                     (uint64_t)(uintptr_t)j) |
                            8u);
    return result;
}

/* (PPII)M — e.g. mlx_array_new_data. */
struct cal_eight cal_P_P_I_I_M(void *a, void *b, int i, int j) {
    struct cal_eight result;
    result.ctx = (void *)(uintptr_t)(cal_fold((uint64_t)(uintptr_t)a ^
                                              (uint64_t)(uintptr_t)b ^
                                              (uint64_t)(uint32_t)i ^
                                              (uint64_t)(uint32_t)j) |
                                     8u);
    return result;
}
