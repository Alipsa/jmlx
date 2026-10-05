"""Host policy shared by environment verification and direct fixture execution."""
import platform


def verify_host(family, profile):
    actual = int(platform.mac_ver()[0].split(".")[0])
    recorded = int(profile["macOSMajor"])
    policy = profile["macOSMajorPolicy"]
    if policy not in {"exact", "minimum"}:
        raise SystemExit(f"unsupported macOS policy for {family}: {policy}")
    if profile["device"] == "gpu" and policy != "exact":
        raise SystemExit(f"GPU profile {family} must use an exact macOS policy")
    if (policy == "exact" and actual != recorded) or (policy == "minimum" and actual < recorded):
        raise SystemExit(
            f"{family} oracle requires macOS major {recorded} ({policy}); found {actual}. "
            "Phase 6 GPU fixtures must be generated and verified on macOS 26; "
            "use -PmlxOracleFamily=phase7-1 for CPU fixtures on newer macOS."
        )
