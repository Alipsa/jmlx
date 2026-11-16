# Phase 8.4 release notes — draft

Not published. The release candidate SHA, candidate/module tag mapping, deployment ID, exact
bundle digest and verified receipt will be filled from the approved release record.

The proposed tuple is native/models/adapters 0.1.0, ffi/core 0.5.0, jinja 0.6.2,
tokenizer/vision 0.1.2. See the [readiness checklist](phase8-4-release-checklist.md) and
[manifest](../tools/release-smoke/released-8.4.json). The
[post-plugin artifact audit](phase8-4-pure-java-audit.md) supports new pure-Java versions;
tuple review and final candidate verification remain open. The original 8.1 coordinates
remain available unchanged.

Native inference requires Java 25 and Apple Silicon macOS 26.2 or later. This minimum reflects
bundled dylib metadata; physical minimum-host execution must pass before publication. No claim
of all Apple GPU families is made. Record tested chip(s), OS versions and physical host identity
here when that evidence exists. Pure-Java modules retain their Java 21 headless contract.

Native pins: mlx-metal 0.32.4 (wheel SHA-256
514df92d41a286895d2564b4b18f4da45096bd7a4a0c47e1042f345ad5a7f936), mlx-c
`a26f2d584a14ec619ae8db9a09be03becd47598a`. The locally compiled wrapper now targets 26.2 and
maps build-machine paths to relative paths. Wheel binaries are unchanged. Full license texts
and source attribution travel inside the native artifact and its documentation companions.

Spring AI 2.0.2 and LangChain4j 1.23.0 adapters provide local text chat, incremental streaming
and supported BERT sentence embeddings through a shared scheduler owner. Tool calls, media,
response formats and stop strings are rejected. Spring Boot auto-configuration and Quarkus
extension wiring are not claimed. See the [8.3 report](phase8-3-implementation-report.md) for
streaming, cancellation, queue-capacity and callback-cooperation limits.

The [compatibility matrix](phase6-compatibility.md) distinguishes synthetic fixtures from real
artifact evidence. The [8.2 benchmark handoff](phase8-2-benchmark.md#8-handoff-to-84--what-the-first-native-release-may-state)
bounds performance statements to its measured host/settings and retains unstable, IO-affected,
unmatched-generation and cross-pin limitations. Packaging changes do not establish new
performance measurements. No golden, binding or upstream native pin changes are part of 8.4.
