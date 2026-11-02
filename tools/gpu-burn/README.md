# gpu-burn

A deliberately induced GPU load generator for the timing methodology that the
Phase 8.2 `mlx_lm` baseline must follow: wall-time measurements only mean
something with the host state recorded, and a *surprising* number must be
re-run before being attributed to the runtime. This tool reproduces the other
half of that methodology — a controlled, reproducible degraded host state —
so that a suspected measurement-window artifact can be distinguished from a
genuine runtime effect (both the original issue #73 window and its resolution
rest on exactly this distinction).

## Usage

```sh
xcrun swiftc -O burn.swift -o burn -framework Metal   # once
./burn                                                # runs until killed
```

Environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `BURN_ON_MS` | `100` | ms of sustained compute dispatch per cycle |
| `BURN_OFF_MS` | `25` | ms of idle per cycle (default duty ≈ 80 %) |
| `BURN_SECONDS` | (unset) | optional; exit after this many seconds |

The kernel is a constant-compute stream (FMA loops over a 256 K-thread
dispatch, 64 in-flight command buffers). It is **not** a calibrated
bandwidth or power probe: do not infer GPU utilization from it. For the
measurement itself, record host state with the numbers (load, `pmset -g therm`
warning level, free memory, resident GPU-using processes) and alternate the
comparands back to back — the issue #73 session (`req/benchmarks/phase7-5-wp6-
tierb-degraded-*.xml`) is the worked example.

## Reference effect (M5 Max, macOS 27.2, default duty)

Measured in the issue #73 session against the two Tier-B workloads, clean vs.
under the burner:

| Workload | Clean | Under burner | Factor |
| --- | ---: | ---: | ---: |
| Qwen3-VL 2B Tier-B generation (light) | 243–245 ms | 2,011–2,145 ms | 8.2–8.8× |
| Phi-3.5-Vision 4B Tier-B generation (heavy) | 3,538–3,658 ms | 13,526–14,555 ms | 3.8–4.0× |

Both MLX pins degraded by the same factor (new/old ratios 0.90–1.07, no
thermal or performance warning level recorded in any run). A real degraded
window with a different contention mix (e.g. memory-bandwidth-heavy serving
rather than a constant compute stream) need not show the same per-model
factors.
