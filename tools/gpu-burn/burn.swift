import Metal
import Foundation

// GPU load generator for controlled-degradation timing methodology (Phase 8.2).
// Sustains a duty-cycled stream of compute work on the default GPU so that
// concurrently running MLX workloads contend for GPU time, reproducing a
// degraded host state on demand. First used to settle issue #73 (the 10:53-11:29
// UTC 2026-11-02 wall-time window was a host artifact: both MLX pins degraded
// by the same factor under this burner).
//
// Build:  xcrun swiftc -O burn.swift -o burn -framework Metal
// Run:    ./burn            (forever; stop with SIGTERM/SIGINT)
//
// Env: BURN_ON_MS (default 100), BURN_OFF_MS (default 25, i.e. ~80% duty),
//      BURN_SECONDS (optional; run for this many seconds and exit).

let onMs = ProcessInfo.processInfo.environment["BURN_ON_MS"].flatMap { Int($0) } ?? 100
let offMs = ProcessInfo.processInfo.environment["BURN_OFF_MS"].flatMap { Int($0) } ?? 25
let maxSeconds = ProcessInfo.processInfo.environment["BURN_SECONDS"].flatMap { Double($0) }

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

let started = Date()
var outstanding: [MTLCommandBuffer] = []
while true {
    if let maxSeconds, Date().timeIntervalSince(started) > maxSeconds {
        break
    }
    let phaseStart = Date()
    while Date().timeIntervalSince(phaseStart) < Double(onMs) / 1000.0 {
        guard let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else { break }
        enc.setComputePipelineState(pipe)
        enc.setBuffer(sinkBuf, offset: 0, index: 0)
        enc.setBuffer(nBuf, offset: 0, index: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cmd.commit()
        outstanding.append(cmd)
        if outstanding.count >= 64 {
            outstanding.removeFirst().waitUntilCompleted()
        }
    }
    Thread.sleep(forTimeInterval: Double(offMs) / 1000.0)
}
for cb in outstanding { cb.waitUntilCompleted() }
print("burn done")
