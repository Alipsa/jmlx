"""Re-runnable evidence for req/plans/phase6-3-performance.md "Verified facts".

Run with the hash-locked oracle environment (same mlx/mlx-metal wheels as the native pin):

    ./tools/mlx-oracle/install.sh
    ./tools/mlx-oracle/.venv/bin/python tools/mlx-oracle/probes/moe_gather_probe.py

Each check prints PASS/FAIL (or MEASURED for timing and memory) with its fact number. The script
exits non-zero if any correctness check fails. Facts 3 and 5 cover runtime behaviour that no Java
test can observe: re-run this script whenever the native pin changes.
"""

import gc
import math
import os
import sys
import tempfile
import time

import mlx.core as mx

FAILURES = []


def check(fact, name, ok, detail=""):
    print(f"[fact {fact}] {'PASS' if ok else 'FAIL'} {name} {detail}".rstrip())
    if not ok:
        FAILURES.append(f"fact {fact}: {name}")


def measured(fact, name, detail):
    print(f"[fact {fact}] MEASURED {name} {detail}")


def switch(x, w, idx, sort):
    """mlx-lm SwitchLinear-style gathered projection; returns [B, T, K, F]."""
    b, t, h = x.shape
    k = idx.shape[-1]
    if not sort:
        y = mx.gather_mm(mx.expand_dims(x, (-2, -3)), w.swapaxes(-1, -2), rhs_indices=idx)
        return y.squeeze(-2)
    n = b * t
    flat = idx.flatten()
    order = mx.argsort(flat)
    inv = mx.argsort(order)
    tok = mx.broadcast_to(mx.arange(n, dtype=mx.int32)[:, None], (n, k)).flatten()
    xs = mx.take(x.reshape(n, 1, h), mx.take(tok, order), axis=0)
    y = mx.gather_mm(xs, w.swapaxes(-1, -2), rhs_indices=mx.take(flat, order), sorted_indices=True)
    return mx.take(y, inv, axis=0).reshape(b, t, k, -1)


def random_indices(b, t, k, e):
    rows = [mx.random.permutation(e)[:k] for _ in range(b * t)]
    return mx.stack(rows).reshape(b, t, k).astype(mx.int32)


def fact1():
    b, t, h, f, e, k = 1, 3, 4, 5, 4, 2
    x = mx.random.normal((b, t, h))
    w = mx.random.normal((e, f, h))
    idx = mx.array([[[0, 2], [3, 0], [1, 1]]], dtype=mx.int32)
    y = mx.gather_mm(mx.expand_dims(x, (-2, -3)), w.swapaxes(-1, -2), rhs_indices=idx)
    check(1, "output shape [B,T,K,1,F]", y.shape == (b, t, k, 1, f), str(y.shape))
    ref = mx.stack(
        [mx.stack([x[0, i] @ w[int(idx[0, i, j])].T for j in range(k)]) for i in range(t)]
    )
    check(1, "equals per-token loop", mx.abs(y[0, :, :, 0, :] - ref).max().item() == 0.0)
    try:
        mx.gather_mm(x, w.swapaxes(-1, -2), rhs_indices=idx.astype(mx.float32))
        check(1, "float indices rejected", False)
    except ValueError:
        check(1, "float indices rejected", True)


def fact2():
    b, t, h, f, e = 1, 40, 8, 6, 4
    x = mx.random.normal((b, t, h))
    w = mx.random.normal((e, f, h))
    for k in (2, e):
        idx = random_indices(b, t, k, e)
        fwd = mx.abs(switch(x, w, idx, False) - switch(x, w, idx, True)).max().item()
        gw_u = mx.grad(lambda w_: switch(x, w_, idx, False).sum())(w)
        gw_s = mx.grad(lambda w_: switch(x, w_, idx, True).sum())(w)
        gx_u = mx.grad(lambda x_: switch(x_, w, idx, False).sum())(x)
        gx_s = mx.grad(lambda x_: switch(x_, w, idx, True).sum())(x)
        bwd = max(mx.abs(gw_u - gw_s).max().item(), mx.abs(gx_u - gx_s).max().item())
        check(2, f"sorted == unsorted fwd+bwd (K={k})", fwd < 1e-5 and bwd < 1e-4,
              f"fwd {fwd:.2g} bwd {bwd:.2g}")


def fact3():
    x = mx.random.normal((1, 3, 1, 1, 4))
    w = mx.random.normal((4, 4, 5))
    idx = mx.array([[[3, 0], [2, 1], [1, 3]]], dtype=mx.int32)  # deliberately unsorted
    good = mx.gather_mm(x, w, rhs_indices=idx)
    flagged = mx.gather_mm(x, w, rhs_indices=idx, sorted_indices=True)
    same = mx.abs(good - flagged).max().item() == 0.0
    # Recorded, not asserted: "sorted_indices is only a hint" is what makes a wrong flag
    # untestable. If this ever flips to DIFFERENT, a Java test for the flag becomes possible.
    measured(3, "sorted_indices=True on unsorted indices", "SAME result" if same else "DIFFERENT")


def fact4():
    e, h, f = 3, 2, 4
    x = mx.array([[[1.0, 2.0], [0.5, -1.0]]])

    def moe(params, gather, stop):
        wr, br, wg, wu, wd = params
        p = mx.softmax((x @ wr.T + br).astype(mx.float32), axis=-1, precise=True)
        idx = mx.argmax(p, axis=-1, keepdims=True).astype(mx.int32)
        if stop:
            idx = mx.stop_gradient(idx)
        sel = mx.take_along_axis(p, idx, axis=-1)
        wgt = sel / sel
        if gather:
            xe = mx.expand_dims(x, (-2, -3))
            g = mx.gather_mm(xe, wg.swapaxes(-1, -2), rhs_indices=idx)
            u = mx.gather_mm(xe, wu.swapaxes(-1, -2), rhs_indices=idx)
            y = mx.gather_mm(g * mx.sigmoid(g) * u, wd.swapaxes(-1, -2), rhs_indices=idx)
            return (y.squeeze(-2) * mx.expand_dims(wgt, -1)).sum(axis=2).sum()
        out = 0
        for j in range(e):
            g = x @ wg[j].T
            u = x @ wu[j].T
            out = out + mx.where(idx == j, (g * mx.sigmoid(g) * u) @ wd[j].T, 0) * wgt
        return out.sum()

    params = [mx.zeros((e, h)), mx.array([1.0, 0.0, -5.0]), mx.random.normal((e, f, h)),
              mx.random.normal((e, f, h)), mx.random.normal((e, h, f))]
    try:
        mx.grad(lambda p: moe(p, True, False))(params)
        check(4, "gather_mm VJP without stop_gradient raises", False)
    except ValueError as err:
        check(4, "gather_mm VJP without stop_gradient raises", "VJP" in str(err), str(err))
    dense = mx.grad(lambda p: moe(p, False, False))(params)
    gathered = mx.grad(lambda p: moe(p, True, True))(params)
    diff = max(mx.abs(a - b).max().item() for a, b in zip(dense, gathered))
    check(4, "stop_gradient grads == dense grads", diff == 0.0, f"max diff {diff:.2g}")


def fact5():
    x = mx.random.normal((1, 3, 1, 1, 4))
    w = mx.random.normal((2, 4, 5))
    idx = mx.array([[[0, 9], [0, 0], [1, 0]]], dtype=mx.int32)  # 9 is out of range
    try:
        mx.eval(mx.gather_mm(x, w, rhs_indices=idx))
        measured(5, "out-of-range index", "NO error (not bounds-checked)")
    except Exception as err:  # noqa: BLE001 - recording behaviour, not handling it
        measured(5, "out-of-range index", f"raised {type(err).__name__}")


def fact6():
    layers, e, f, h = 4, 8, 256, 1024
    total = layers * e * 2 * f * h * 4
    path = os.path.join(tempfile.mkdtemp(), "moe_layers.safetensors")
    weights = {f"l{l}.e{j}.{p}": mx.random.normal((f, h))
               for l in range(layers) for j in range(e) for p in ("w1", "w3")}
    mx.eval(weights)
    mx.save_safetensors(path, weights)
    del weights
    gc.collect()

    def run(close_sources, eval_per_layer):
        mx.clear_cache()
        mx.reset_peak_memory()
        base = mx.get_active_memory()
        t = mx.load(path)
        kept = []
        for l in range(layers):
            for p in ("w1", "w3"):
                keys = [f"l{l}.e{j}.{p}" for j in range(e)]
                st = mx.stack([t[key] for key in keys])
                if close_sources:
                    for key in keys:
                        del t[key]
                if eval_per_layer:
                    mx.eval(st)
                kept.append(st)
        mx.eval(kept)
        steady = (mx.get_active_memory() - base) / total
        peak = (mx.get_peak_memory() - base) / total
        del t, kept
        gc.collect()
        return f"steady {steady:.3f}x peak {peak:.3f}x"

    measured(6, "close sources + eval per layer", run(True, True))
    measured(6, "keep sources", run(False, True))
    measured(6, "close sources, single eval", run(True, False))


def fact7():
    h, f, e, k = 1024, 3584, 8, 2
    wg, wu = (mx.random.normal((e, f, h)).astype(mx.bfloat16) for _ in range(2))
    wd = mx.random.normal((e, h, f)).astype(mx.bfloat16)
    mx.eval(wg, wu, wd)

    def silu(a):
        return a * mx.sigmoid(a)

    def dense(x, idx):
        out = mx.zeros(x.shape, x.dtype)
        for j in range(e):
            y = (silu(x @ wg[j].T) * (x @ wu[j].T)) @ wd[j].T
            for s in range(k):
                out = out + mx.where(idx[..., s:s + 1] == j, y, 0)
        return out

    def gathered(x, idx, sort):
        hid = silu(switch(x, wg, idx, sort)) * switch(x, wu, idx, sort)
        b, t, _ = x.shape
        y = mx.gather_mm(hid.reshape(b, t, k, 1, f), wd.swapaxes(-1, -2), rhs_indices=idx)
        return y.squeeze(-2).sum(-2)

    def bench(fn, *args, n=20):
        for _ in range(3):
            mx.eval(fn(*args))
        start = time.perf_counter()
        for _ in range(n):
            mx.eval(fn(*args))
        return (time.perf_counter() - start) / n * 1e3

    for t in (1, 128):
        x = mx.random.normal((1, t, h)).astype(mx.bfloat16)
        idx = random_indices(1, t, k, e)
        mx.eval(x, idx)
        line = f"T={t} dense {bench(dense, x, idx):.2f} ms, gathered {bench(gathered, x, idx, False):.2f} ms"
        if t > 1:
            line += f", gathered-sorted {bench(gathered, x, idx, True):.2f} ms"
        measured(7, "one bf16 MoE MLP (H=1024 F=3584 E=8 K=2)", line)


def fact8():
    """bf16/f16 values vs an exact closed form; gathered vs dense bit-identity."""
    e, h = 3, 4

    def pattern(n, salt):
        return [2.0 * math.sin(0.37 * i + 1.3 * salt) for i in range(n)]

    def closed(xv, t, k):
        out = []
        for tok in range(t):
            row = xv[tok * h:(tok + 1) * h]
            logits = row[:e]
            m = max(logits)
            p = [math.exp(v - m) for v in logits]
            z = sum(p)
            p = [v / z for v in p]
            rem = p[:]
            sel = []
            for _ in range(k):
                j = max(range(e), key=lambda c: (rem[c], -c))
                sel.append(j)
                rem[j] = -1
            s = sum(p[j] for j in sel)
            scale = sum((j + 1) * p[j] / s for j in sel)
            out += [scale * v * v / (1 + math.exp(-v)) for v in row]
        return out

    def route(x, k, dt):
        p = mx.softmax((x @ mx.eye(h)[:e].astype(dt).T).astype(mx.float32), axis=-1,
                       precise=True)
        rem, idx, sp = p, [], []
        for _ in range(k):
            i = mx.argmax(rem, axis=-1, keepdims=True)
            idx.append(i)
            sp.append(mx.take_along_axis(p, i, axis=-1))
            rem = mx.where(mx.arange(e) == i, -mx.inf, rem)
        total = sp[0]
        for v in sp[1:]:
            total = total + v
        return idx, [(v / total).astype(dt) for v in sp]

    for dt, bound in ((mx.bfloat16, 0.02), (mx.float16, 0.005)):
        for t, k in ((2, 2), (40, 2), (40, 3)):
            xq = mx.array(pattern(t * h, 3)).reshape(1, t, h).astype(dt)
            ref = closed(xq.astype(mx.float32).flatten().tolist(), t, k)
            idx, w = route(xq, k, dt)
            eye = mx.eye(h).astype(dt)
            gate = mx.stack([eye] * e)
            down = mx.stack([eye * (j + 1) for j in range(e)])
            sel = mx.concatenate(idx, -1).astype(mx.int32)
            g = mx.gather_mm(xq.reshape(1, t, 1, 1, h), gate.swapaxes(-1, -2), rhs_indices=sel)
            y = mx.gather_mm(g * mx.sigmoid(g) * g, down.swapaxes(-1, -2), rhs_indices=sel)
            gathered = (y.reshape(1, t, k, h) * mx.concatenate(w, -1)[..., None]).sum(axis=2)
            dense = mx.zeros(xq.shape, dt)
            for j in range(e):
                a = xq @ eye.T
                yj = (a * mx.sigmoid(a) * a) @ (eye * (j + 1)).T
                for s in range(k):
                    dense = dense + mx.where(idx[s] == j, yj, mx.zeros((1,), dt)) * w[s]
            got = gathered.astype(mx.float32).flatten().tolist()
            worst = max(abs(a - r) / (abs(r) + 1) for a, r in zip(got, ref))
            ident = mx.abs(gathered.astype(mx.float32) - dense.astype(mx.float32)).max().item()
            check(8, f"{dt} T={t} K={k} |err|/(|ref|+1) <= {bound}", worst <= bound,
                  f"worst {worst:.4f}; gathered-vs-dense {ident:.2g}")


def main():
    print(f"mlx {mx.__version__}")
    mx.random.seed(0)
    for fn in (fact1, fact2, fact3, fact4, fact5, fact8):
        fn()
    if "--no-slow" not in sys.argv:
        fact6()
        fact7()
    if FAILURES:
        print("FAILED: " + "; ".join(FAILURES))
        sys.exit(1)
    print("all correctness checks passed")


if __name__ == "__main__":
    main()
