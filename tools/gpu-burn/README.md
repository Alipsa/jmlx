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
BURN_SECONDS=1 ./burn                                # exits with timing summary
```

Environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `BURN_ON_MS` | `100` | ms of compute submission per cycle, before draining |
| `BURN_OFF_MS` | `25` | ms of sleep after draining this queue |
| `BURN_DRAIN_BEFORE_SLEEP` | `1` | accepts only `0` or `1`; `0` preserves the issue #73 loop |
| `BURN_SECONDS` | (unset) | optional; exit after this many seconds |

Defaults apply only when a variable is unset. Malformed numeric settings
(including empty strings) print an error and exit with status 1. `BURN_ON_MS`
must be a positive integer, `BURN_OFF_MS` a nonnegative integer, and
`BURN_SECONDS` a finite positive number.

The default allows only one command buffer in flight and waits for it before
submitting another. This limits drain overshoot to one dispatch rather than
a backlog of 63 buffers. It waits for all submitted work before starting the
sleep, guaranteeing that this burner's queue is idle during that interval.
The active interval includes drain time, so 100/25 ms does **not** promise an
80 % duty cycle. With `BURN_SECONDS`, the tool prints measured host active
seconds, `queue_idle_s`, and their active fraction over completed cycles. These are host
intervals, not a hardware GPU utilization measurement; other processes may
still use the GPU during the burner's idle interval. The duration limit is
checked between cycles and may overrun by the current cycle, including a
dispatch that can take longer under contention. Endless mode (`./burn`)
prints no timing summary: SIGINT/SIGTERM terminate it immediately. Use
`BURN_SECONDS` when collecting timing output. The duration and summary begin
after device/pipeline setup; first-use Metal compilation can add startup time
to the process wall time.

The issue #73 measurements used the original loop, now selected with
`BURN_DRAIN_BEFORE_SLEEP=0`. Up to 63 command buffers could remain queued
during each CPU sleep, so no GPU idle interval was guaranteed. The load is
treated as sustained compute contention; the previously stated ~100 % GPU
utilization was inferred, not measured. The recorded timings below apply
to that historical mode, not the current default.
Its summary prints only `cpu_submit_wait_s` and `cpu_sleep_s`, with no active
fraction or GPU idle claim. These CPU intervals exclude the final GPU drain.

The kernel is a constant-compute stream — not a calibrated bandwidth or power
probe. For the measurement itself, record host state with the numbers (load,
`pmset -g therm` warning level, free memory, resident GPU-using processes) —
`hostrec.sh` next to this file is the per-run recorder committed from the
issue #73 session, with its field semantics documented — and alternate the
comparands back to back; the issue #73 session
(`req/benchmarks/phase7-5-wp6-tierb-degraded-*.xml` plus the
`phase7-5-wp6-issue73-*` host-state and forensics records) is the worked
example, with a gap: the four clean runs lack individual host samples because
the recorder initially failed. The session-start sample provides context only.

## Reference effect (M5 Max, macOS 27.2, historical mode)

Measured in the issue #73 session against the two Tier-B workloads, clean vs.
under the burner (new 0.32.4 / old 0.31.2):

| Workload | Clean (new / old) | Under burner (new / old) | Factor | New/old under load |
| --- | ---: | --- | ---: | ---: |
| Qwen3-VL 2B Tier-B generation (light) | 243 / 245 ms | 2,145/2,057 / 2,064/2,011 ms | 8.2–8.8× | 1.00–1.07 |
| Phi-3.5-Vision 4B Tier-B generation (heavy) | 3,538 / 3,658 ms | 13,526/13,587 / 14,555/14,273 ms | 3.8–4.0× | 0.93–0.95 |

The available pre-run samples show no thermal or performance warning; no
continuous monitoring was recorded. A real
degraded window with a different contention mix (e.g. memory-bandwidth-heavy
serving rather than a constant compute stream) need not show the same
per-model factors — the issue #73 window inverted them (Qwen3-VL 2.3–3.9× vs.
Phi-3.5-Vision 7.1–14.9×).
