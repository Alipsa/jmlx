# Phase 8.4 release notes — draft

Not published. RC3 technical verification and final-receipt revalidation passed. The owner
approved the signed candidate, delivery terms, jinja review and both license-finding
dispositions on 2026-11-23. Portal capacity and final publication authorization remain
pending. The
[RC3 review](phase8-4-rc3-final-review.md) records the exact tag, SHA, signed ZIP and receipt.
Module release tags and deployment identity remain for the authorized publication record.

The proposed tuple is native/models/adapters 0.1.0, ffi/core 0.5.0, jinja 0.6.2,
tokenizer/vision 0.1.2. See the [readiness checklist](phase8-4-release-checklist.md) and
[manifest](../tools/release-smoke/released-8.4.json). The
[post-plugin artifact audit](phase8-4-pure-java-audit.md) supports new pure-Java versions;
the approved tuple's RC3 technical verification is complete. The original 8.1 coordinates
remain available unchanged.

Native inference requires Java 25 and is supported on Apple Silicon macOS 27.2 or later
(all 27.2.x patch releases). The supported minimum was raised by owner decision on 2026-11-16.
Physical RC3 execution passed on Apple M5 Max, macOS 27.2.1 (26B131), using the dedicated
non-admin runner account in
[run 45206346138](https://github.com/Alipsa/jmlx/actions/runs/45206346138).
This is the same physical Mac as the local macOS runs, under a separate account.
No claim of all Apple GPU families is made. Pure-Java modules retain their Java 21
headless contract.

Native pins: mlx-metal 0.32.4 (wheel SHA-256
514df92d41a286895d2564b4b18f4da45096bd7a4a0c47e1042f345ad5a7f936), mlx-c
`a26f2d584a14ec619ae8db9a09be03becd47598a`. The binary deployment target remains 26.2 for
hosted CI coverage. Binaries may load below 27.2, but those OS versions are unsupported.
The wrapper maps build-machine paths to relative paths. Wheel binaries are unchanged. Full
license texts and source attribution travel inside the native artifact and its documentation
companions.

Spring AI 2.0.2 and LangChain4j 1.23.0 adapters provide local text chat, incremental streaming
and supported BERT sentence embeddings through a shared scheduler owner. Tool calls, media,
response formats and stop strings are rejected. Spring Boot auto-configuration and Quarkus
extension wiring are not claimed. See the [8.3 report](phase8-3-implementation-report.md) for
streaming, cancellation, queue-capacity and callback-cooperation limits.

The [compatibility matrix](phase6-compatibility.md) distinguishes synthetic fixtures from real
artifact evidence. The
[8.2 benchmark handoff](phase8-2-benchmark.md#8-handoff-to-84--what-the-first-native-release-may-state)
bounds performance statements to its measured host/settings and retains unstable, IO-affected,
unmatched-generation and cross-pin limitations. Packaging changes do not establish new
performance measurements. No golden, binding or upstream native pin changes are part of 8.4.

## Native license scope

The native distribution retains MLX's MIT grant for Apple/MLX contributions and separately identifies Norbert Juffa's covered expression in the reviewed Metal erf and Metal/CPU SIMD erfinv material under CC BY-SA 4.0. META-INF/THIRD-PARTY-NOTICES records the reviewed scope, revisions, modifications and redistribution basis; META-INF/LICENSE-CC-BY-SA-4.0 contains the complete CC license. No CC adapter's license is identified. Other bundled material retains its applicable licenses and notices. The owner-approved interpretation and its recorded counterargument are documented in the repository's owner license decision record.

See the [owner decision](phase8-4-owner-license-decision.md) and
[approved notice](https://github.com/Alipsa/jmlx/blob/551eebeeb66a80ef3af467a6c1b4995c2a7f7034/jmlx-native-macos-arm64/THIRD-PARTY-NOTICES).
Final local metadata was accepted in PR #115. All five signed RC3 native publications
match those approved byte pins, including Javadoc and module metadata. The owner's
2026-11-23 approval accepts the signed candidate and delivery terms and resolves both
findings for this exact scope under the previously recorded interpretations. The
[RC3 review](phase8-4-rc3-final-review.md) retains that decision and its evidence limits.
Current Portal account capacity and final publication authorization remain pending.
