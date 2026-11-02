import Metal
import Foundation

// GPU load generator for controlled-degradation timing methodology (Phase 8.2).
// Sustains a stream of compute work on the default GPU so that concurrently
// running MLX workloads contend for GPU time, reproducing a degraded host
// state on demand. First used to investigate issue #73: both MLX pins
// degraded similarly under this burner; the original host state is unknown.
//
// The default drains this queue before sleeping. The active interval includes
// drain time, so ON/(ON+OFF) is not the effective duty cycle. Printed timings
// measure host active/idle intervals, not hardware GPU utilization.
// BURN_DRAIN_BEFORE_SLEEP=0 preserves the original issue #73 submission loop:
// queued work can execute throughout the CPU sleep; no GPU idle is guaranteed.
//
// Build:  xcrun swiftc -O burn.swift -o burn -framework Metal
// Run:    ./burn            (forever; stop with SIGTERM/SIGINT)
//         BURN_SECONDS=1 ./burn  (bounded run with timing summary)
// Signals terminate immediately without a summary.
//
// Env: BURN_ON_MS (default 100), BURN_OFF_MS (default 25),
//      BURN_DRAIN_BEFORE_SLEEP (default 1; 0 for historical behavior),
//      BURN_SECONDS (optional; run for this many seconds and exit).

// Defaults apply only to absent settings. A typo in a requested time limit
// must never turn a bounded run into an endless one.
func readSetting<T>(_ name: String, parse: (String) -> T?) -> T? {
    guard let raw = ProcessInfo.processInfo.environment[name] else { return nil }
    guard let value = parse(raw) else {
        FileHandle.standardError.write("invalid numeric value for \(name)\n".data(using: .utf8)!)
        exit(1)
    }
    return value
}

let onMs = readSetting("BURN_ON_MS", parse: { Int($0) }) ?? 100
let offMs = readSetting("BURN_OFF_MS", parse: { Int($0) }) ?? 25
let maxSeconds = readSetting("BURN_SECONDS", parse: { Double($0) })
let drainSetting = ProcessInfo.processInfo.environment["BURN_DRAIN_BEFORE_SLEEP"] ?? "1"
guard drainSetting == "0" || drainSetting == "1" else {
    FileHandle.standardError.write("BURN_DRAIN_BEFORE_SLEEP must be 0 or 1\n".data(using: .utf8)!)
    exit(1)
}
let drainBeforeSleep = drainSetting == "1"
// A single buffer bounds drain overshoot to one dispatch in the default mode.
// Keep the original queue depth only when reproducing the historical load.
let maxInFlight = drainBeforeSleep ? 1 : 64

guard onMs > 0, offMs >= 0, maxSeconds.map({ $0.isFinite && $0 > 0 }) ?? true else {
    FileHandle.standardError.write("ON must be positive, OFF nonnegative, SECONDS finite and positive\n".data(using: .utf8)!)
    exit(1)
}

guard let device = MTLCreateSystemDefaultDevice() else {
    FileHandle.standardError.write("no default GPU device\n".data(using: .utf8)!)
    exit(1)
}
let queue = device.makeCommandQueue()!

let src = """
#include <metal_stdlib>
using namespace metal;
kernel void burn(device atomic<float>* sink [[buffer(0)]],
                 constant uint& n [[buffer(1)]]) {
    float v = 1.0000001f;
    for (uint i = 0; i < n; i++) { v = fma(v, 1.0000002f, 0.5f); }
    atomic_fetch_add_explicit(sink, v, memory_order_relaxed);
}
"""

guard let lib = try? device.makeLibrary(source: src, options: nil),
      let fn = lib.makeFunction(name: "burn"),
      let pipe = try? device.makeComputePipelineState(function: fn) else {
    FileHandle.standardError.write("pipeline creation failed\n".data(using: .utf8)!)
    exit(1)
}

let sinkBuf = device.makeBuffer(length: MemoryLayout<Float>.size, options: [])!
memset(sinkBuf.contents(), 0, MemoryLayout<Float>.size)
var n: UInt32 = 200_000
let nBuf = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: [])!
memcpy(nBuf.contents(), &n, MemoryLayout<UInt32>.size)

let tg = MTLSize(width: 1024, height: 1, depth: 1)
let grid = MTLSize(width: 1024 * 256, height: 1, depth: 1)

let started = ProcessInfo.processInfo.systemUptime
var activeSeconds = 0.0
var idleSeconds = 0.0
var cycles = 0
var outstanding: [MTLCommandBuffer] = []
while true {
    if let maxSeconds, ProcessInfo.processInfo.systemUptime - started > maxSeconds {
        break
    }
    let phaseStart = ProcessInfo.processInfo.systemUptime
    while ProcessInfo.processInfo.systemUptime - phaseStart < Double(onMs) / 1000.0 {
        guard let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else { break }
        enc.setComputePipelineState(pipe)
        enc.setBuffer(sinkBuf, offset: 0, index: 0)
        enc.setBuffer(nBuf, offset: 0, index: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cmd.commit()
        outstanding.append(cmd)
        if outstanding.count >= maxInFlight {
            outstanding.removeFirst().waitUntilCompleted()
        }
    }
    if drainBeforeSleep {
        for cb in outstanding { cb.waitUntilCompleted() }
        outstanding.removeAll(keepingCapacity: true)
    }
    let idleStart = ProcessInfo.processInfo.systemUptime
    activeSeconds += idleStart - phaseStart
    Thread.sleep(forTimeInterval: Double(offMs) / 1000.0)
    idleSeconds += ProcessInfo.processInfo.systemUptime - idleStart
    cycles += 1
}
for cb in outstanding { cb.waitUntilCompleted() }
if drainBeforeSleep {
    let cycleSeconds = activeSeconds + idleSeconds
    let activeFraction = cycleSeconds > 0 ? activeSeconds / cycleSeconds : 0
    print("drain_before_sleep=true cycles=\(cycles) active_s=\(activeSeconds) queue_idle_s=\(idleSeconds) host_active_fraction=\(activeFraction)")
} else {
    // CPU sleep overlaps queued GPU execution. Do not present a duty fraction
    // or call the sleep time idle; the final GPU drain is outside these totals.
    print("drain_before_sleep=false cycles=\(cycles) cpu_submit_wait_s=\(activeSeconds) cpu_sleep_s=\(idleSeconds)")
}
print("burn done")
