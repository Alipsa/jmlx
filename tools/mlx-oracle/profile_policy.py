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
            f"{family} oracle requires macOS major {recorded} ({policy}); found {actual}."
        )


def select_profiles(profiles, requested=None, require_all=False):
    if require_all:
        if requested:
            raise SystemExit("require-all-profiles cannot be combined with a family selection")
        for family, profile in profiles.items():
            try:
                verify_host(family, profile)
            except IncompatibleHost as error:
                raise IncompatibleHost(
                    f"{error} All profiles are required; the runner must satisfy every profile's "
                    f"OS requirement (use macOS {profile['macOSMajor']} for {family})."
                ) from error
    if requested:
        if requested not in profiles:
            raise SystemExit(f"missing {requested} provenance profile")
        try:
            verify_host(requested, profiles[requested])
        except IncompatibleHost as error:
            raise IncompatibleHost(
                f"{error} Omit -PmlxOracleFamily (or --family) to verify only compatible profiles."
            ) from error
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
