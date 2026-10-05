"""Host policy shared by environment verification and direct fixture execution."""
import platform


class IncompatibleHost(SystemExit):
    """A valid profile cannot reproduce fixtures on this host."""


def verify_host(family, profile):
    actual = int(platform.mac_ver()[0].split(".")[0])
    recorded = int(profile["macOSMajor"])
    policy = profile["macOSMajorPolicy"]
    if policy not in {"exact", "minimum"}:
        raise SystemExit(f"unsupported macOS policy for {family}: {policy}")
    if profile["device"] == "gpu" and policy != "exact":
        raise SystemExit(f"GPU profile {family} must use an exact macOS policy")
    if (policy == "exact" and actual != recorded) or (policy == "minimum" and actual < recorded):
        raise IncompatibleHost(
            f"{family} oracle requires macOS major {recorded} ({policy}); found {actual}. "
            "Phase 6 GPU fixtures must be generated and verified on macOS 26; "
            "omit -PmlxOracleFamily to verify only the compatible profiles."
        )


def select_profiles(profiles, requested=None, require_all=False):
    if require_all:
        if requested:
            raise SystemExit("require-all-profiles cannot be combined with a family selection")
        for family, profile in profiles.items():
            verify_host(family, profile)
    if requested:
        if requested not in profiles:
            raise SystemExit(f"missing {requested} provenance profile")
        verify_host(requested, profiles[requested])
        return [requested]
    selected = []
    for family, profile in profiles.items():
        try:
            verify_host(family, profile)
        except IncompatibleHost as error:
            print(f"MLX oracle skipped {family}: {error}")
        else:
            selected.append(family)
    if not selected:
        raise SystemExit("no oracle profiles are compatible with this host")
    print(f"MLX oracle selected profiles: {', '.join(selected)}")
    return selected
