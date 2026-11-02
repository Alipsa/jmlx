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
| `BURN_ON_MS` | `100` | ms of compute dispatch per cycle (CPU submission pattern) |
| `BURN_OFF_MS` | `25` | ms of CPU sleep per cycle |
| `BURN_SECONDS` | (unset) | optional; exit after this many seconds |

**The GPU stays saturated (~100 % duty) for the default settings.** Each
command buffer is a 262,144-thread dispatch of 200,000 dependent FMAs, and the
loop holds 64 buffers in flight: when each `BURN_OFF_MS` sleep starts, ~63
buffers are still queued, and draining them inside 25 ms would need over
1e14 FMA/s — beyond any M-series GPU. `BURN_ON_MS`/`BURN_OFF_MS` therefore
shape the CPU submission pattern, not the GPU utilization; treat this as a
saturating contention source rather than a tunable duty cycle.

The kernel is a constant-compute stream — not a calibrated bandwidth or power
probe. For the measurement itself, record host state with the numbers (load,
`pmset -g therm` warning level, free memory, resident GPU-using processes) —
`hostrec.sh` next to this file is the per-run recorder committed from the
issue #73 session, with its field semantics documented — and alternate the
comparands back to back; the issue #73 session
(`req/benchmarks/phase7-5-wp6-tierb-degraded-*.xml` plus the
`phase7-5-wp6-issue73-*` host-state and forensics records) is the worked
example.

## Reference effect (M5 Max, macOS 27.2, default settings)

Measured in the issue #73 session against the two Tier-B workloads, clean vs.
under the burner (new 0.32.4 / old 0.31.2):

| Workload | Clean (new / old) | Under burner (new / old) | Factor | New/old under load |
| --- | ---: | --- | ---: | ---: |
| Qwen3-VL 2B Tier-B generation (light) | 243 / 245 ms | 2,145/2,057 / 2,064/2,011 ms | 8.2–8.8× | 1.00–1.07 |
| Phi-3.5-Vision 4B Tier-B generation (heavy) | 3,538 / 3,658 ms | 13,526/13,587 / 14,555/14,273 ms | 3.8–4.0× | 0.93–0.95 |

No thermal or performance warning level was recorded in any run. A real
degraded window with a different contention mix (e.g. memory-bandwidth-heavy
serving rather than a constant compute stream) need not show the same
per-model factors — the issue #73 window inverted them (Qwen3-VL 2.3–3.9× vs.
Phi-3.5-Vision 7.1–14.9×).
