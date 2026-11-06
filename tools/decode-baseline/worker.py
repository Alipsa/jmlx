#!/usr/bin/env python3
"""Phase 8.2 decode-baseline Python worker (controlled and stock adapters).

Consumes a launcher-written job file (schema jmlx-decode-baseline-job-1) and
writes a per-worker report (schema jmlx-decode-baseline-1) plus a worker-ready
marker into the job's output directory. Frozen contracts:
req/plans/phase8-2-reference-findings.md sections 6 (adapters), 7.3 (priming)
and 10 (schema), plan sections 3.2/3.3/4.

Rules enforced here:

- the frozen workload manifest is re-hashed and every consumed file digest is
  re-verified before any model work; the venv pins (versions and the installed
  mlx_lm generate.py digest) are re-checked the same way;
- MLX_ENABLE_TF32 is set from the job before any mlx import, so a
  shell-inherited precision mode can never select the run;
- one sequential full read of every checkpoint safetensors file precedes each
  request (cache_regime: primed-per-request); prime_s is recorded and excluded
  from every measured interval;
- stock runs execute the installed stream_generate unmodified; the resolved
  wrapper stop set is completed to the manifest's frozen set with the shipped
  add_eos_token API and asserted equal; the full-generation timer ends with an
  explicit mx.synchronize() after the iterator finishes;
- a worker wall-clock timeout produces a failed report and a nonzero exit; a
  worker never presents a partial result as success.
"""

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path

# -I (isolated mode) keeps this file's directory off sys.path; the controlled
# loop module is a sibling and must be importable either way.
sys.path.insert(0, str(Path(__file__).resolve().parent))

JOB_SCHEMA = "jmlx-decode-baseline-job-1"
REPORT_SCHEMA = "jmlx-decode-baseline-1"
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_TIMEOUT = 3

# Runtime pins must match tools/decode-baseline/requirements.lock; the
# generate.py digest comes from installed-source-digests.json at run time.
EXPECTED_VERSIONS = {"mlx": "0.32.4", "mlx-metal": "0.32.4", "mlx-lm": "0.32.2"}
PRIME_CHUNK = 1 << 20
WATCHDOG_GRACE_S = 30.0

state = {
    "utc_start": None,
    "utc_end": None,
    "status": "failed",
    "failure_reason": None,
    "adapter": None,
    "case": None,
    "prompt_length": None,
    "mode": None,
    "identity": {},
    "stop_token_ids": [],
    "load_s": None,
    "load_state": None,
    "requests": [],
    "notes": [],
    "report_written": False,
    "out_dir": None,
}
PRIMING_ENABLED = True
deadline = {"t": None, "fired": threading.Event()}


class WorkerError(Exception):
    pass


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fid:
        for chunk in iter(lambda: fid.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(cond, reason):
    if not cond:
        raise WorkerError(reason)
    return cond


def write_report(out_dir):
    if state["report_written"]:
        return
    state["report_written"] = True
    state["utc_end"] = utcnow()
    report = {
        "schema": REPORT_SCHEMA,
        "worker": {
            "runtime": "python",
            "adapter": state["adapter"],
            "case": state["case"],
            "prompt_length": state["prompt_length"],
            "mode": state["mode"],
            "status": state["status"],
            "failure_reason": state["failure_reason"],
            "utc_start": state["utc_start"],
            "utc_end": state["utc_end"],
            "identity": state["identity"],
            "stop_token_ids": state["stop_token_ids"],
            "load_s": state["load_s"],
            "load_state": state["load_state"],
            "requests": state["requests"],
            "notes": state["notes"],
        },
    }
    tmp = out_dir / "report.json.tmp"
    with open(tmp, "w") as fid:
        json.dump(report, fid, indent=1)
        fid.write("\n")
    os.replace(tmp, out_dir / "report.json")


def watchdog():
    """Hard backstop: if the main loop has not exited shortly after the
    deadline, write the timeout report and kill the process."""
    while not deadline["fired"].is_set():
        time.sleep(1.0)
        if time.monotonic() < deadline["t"]:
            continue
        deadline["fired"].set()
        time.sleep(WATCHDOG_GRACE_S)
        state["status"] = "timeout"
        state["failure_reason"] = "worker wall-clock timeout exceeded"
        write_report(Path(state["out_dir"]))
        os._exit(EXIT_TIMEOUT)


def validate(job):
    require(job.get("schema") == JOB_SCHEMA, f"job schema must be {JOB_SCHEMA}")
    adapter = job.get("adapter")
    require(adapter in ("controlled", "stock"), "adapter must be controlled or stock")
    mode = job.get("mode")
    require(
        mode in ("headline", "fixed-continuation", "reference-pair"),
        "mode must be headline, fixed-continuation or reference-pair",
    )
    case_name = job.get("case")
    require(isinstance(case_name, str) and case_name, "case is required")
    prompt_length = job.get("prompt_length")
    require(isinstance(prompt_length, int) and prompt_length > 0, "prompt_length is required")
    require(job.get("precision_env") in ("1", "0"), "precision_env must be '1' or '0'")
    timeout_s = job.get("timeout_s")
    require(isinstance(timeout_s, int) and timeout_s > 0, "timeout_s must be a positive integer")
    for key in ("artifact_dir", "manifest_path", "manifest_sha256", "repo_dir", "out_dir"):
        require(isinstance(job.get(key), str) and job[key], f"{key} is required")

    manifest_path = Path(job["manifest_path"])
    require(manifest_path.is_file(), "manifest path does not exist")
    manifest_sha = sha256_file(manifest_path)
    require(manifest_sha == job["manifest_sha256"], "manifest SHA-256 mismatch (frozen contract)")
    man = json.load(open(manifest_path))

    cases = {c["name"]: c for c in man.get("text_cases", [])}
    vlm_names = {v["name"] for v in man.get("vlm_cases", [])}
    require(case_name in cases, f"case {case_name!r} is not a text case (VLM cases are WP4)")
    require(case_name not in vlm_names, "VLM cases are out of scope for WP1 text workers")
    case = cases[case_name]
    require(
        prompt_length in man["controlled_generation"]["prompt_lengths"],
        f"prompt_length {prompt_length} is not a frozen manifest length",
    )

    policy = man.get("run_policy")
    require(isinstance(policy, dict), "manifest run_policy block is missing")
    requests = job.get("requests")
    require(isinstance(requests, dict), "requests block is required")
    for key in ("first", "warmups", "measured"):
        require(
            isinstance(requests.get(key), int) and requests[key] >= 0,
            f"requests.{key} must be a nonnegative integer",
        )
    if mode == "reference-pair":
        # Observed-cache reference pair (findings section 7.3): no priming,
        # one measured request, the separate shorter timeout.
        require(
            requests == {"first": 0, "warmups": 0, "measured": 1},
            "reference-pair must be exactly one measured request with no priming",
        )
    else:
        require(requests["warmups"] == policy.get("warmups"), "requests.warmups differs from run_policy")
        require(
            requests["measured"] == policy.get("measured_requests"),
            "requests.measured differs from run_policy",
        )
    emitted = policy.get("emitted_tokens")
    require(
        emitted == man["controlled_generation"]["max_new_tokens"],
        "run_policy.emitted_tokens differs from controlled_generation.max_new_tokens",
    )
    expected_timeout = (
        policy.get("reference_pair_worker_timeout_s") if mode == "reference-pair"
        else policy.get("worker_timeout_s")
    )
    require(
        timeout_s == expected_timeout and expected_timeout is not None and expected_timeout > 0,
        "timeout_s differs from the manifest run_policy value",
    )
    io_affected = f"{case_name}/{prompt_length}" in policy.get("io_affected_cells", [])

    artifact_dir = Path(job["artifact_dir"])
    require(artifact_dir.is_dir(), "artifact_dir does not exist")
    for name, pin in case["files"].items():
        path = artifact_dir / name
        require(path.is_file(), f"checkpoint file missing: {name}")
        actual = sha256_file(path)
        require(actual == pin["sha256"], f"checkpoint digest mismatch for {name}")
    safetensors = sorted(artifact_dir.glob("*.safetensors"))
    require(safetensors, "no safetensors files in artifact_dir (priming cannot run)")

    # Venv pins: versions from the lock, installed source digest from the freeze.
    tool_dir = manifest_path.parent
    freeze = json.load(open(tool_dir / "installed-source-digests.json"))
    generate_digest = freeze["files"]["generate.py"]
    import importlib.util

    spec = importlib.util.find_spec("mlx_lm")
    generate_path = Path(spec.origin).parent / "generate.py"
    require(sha256_file(generate_path) == generate_digest, "installed mlx_lm generate.py digest drifted")
    for dist, expected in EXPECTED_VERSIONS.items():
        require(pkg_version(dist) == expected, f"installed {dist} != {expected} (requirements.lock)")

    repo_dir = Path(job["repo_dir"])
    out_dir = Path(job["out_dir"])
    require(out_dir.is_dir(), "out_dir must be created by the launcher")
    require(not (out_dir / "report.json").exists(), "out_dir already contains a report (overwrite rejected)")

    prompt_entry = case["prompts"]["per_length"][str(prompt_length)]
    require(isinstance(prompt_entry, dict) and isinstance(prompt_entry.get("ids"), list),
            "frozen prompt entry must carry ids and sha256")
    prompt = prompt_entry["ids"]
    require(len(prompt) == prompt_length, "frozen prompt length mismatch")
    canonical = hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode("utf-8")).hexdigest()
    require(canonical == prompt_entry["sha256"], "frozen prompt canonical SHA-256 mismatch")
    fed_ids = None
    if mode == "fixed-continuation":
        cont = job.get("fixed_continuation")
        require(isinstance(cont, dict), "fixed_continuation spec is required in this mode")
        base = case["prompts"]["per_length"]["512"]["ids"]
        offset = cont.get("offset")
        count = cont.get("count")
        require(
            offset == 32 and count == 32 and len(base) >= offset + count,
            "fixed_continuation must be prefix512[32:64] (manifest prompt_generator rule)",
        )
        fed_ids = base[offset : offset + count]

    return {
        "case": case,
        "prompt": prompt,
        "fed_ids": fed_ids,
        "io_affected": io_affected,
        "requests": requests,
        "adapter": adapter,
        "mode": mode,
        "prompt_length": prompt_length,
        "artifact_dir": artifact_dir,
        "manifest_path": manifest_path,
        "manifest_sha": manifest_sha,
        "repo_dir": repo_dir,
        "out_dir": out_dir,
        "timeout_s": timeout_s,
        "policy": policy,
        "wired_limit": (
            man["controlled_generation"]["wired_limit"]
            if adapter == "controlled"
            else man["stock_generation"]["wired_limit"]
        ),
        "device_name": None,
    }


def repo_identity(repo_dir):
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo_dir), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    dirty = bool(
        subprocess.run(
            ["git", "-C", str(repo_dir), "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip()
    )
    return {"sha": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "dirty": dirty}


def prime(artifact_dir):
    """One sequential full read of every checkpoint safetensors file.

    Skipped (0.0) for reference-pair workers, which document the observed
    cache regime without priming (findings section 7.3)."""
    if not PRIMING_ENABLED:
        return 0.0
    t0 = time.monotonic()
    for path in sorted(artifact_dir.glob("*.safetensors")):
        with open(path, "rb") as fid:
            while fid.read(PRIME_CHUNK):
                pass
    return time.monotonic() - t0


def controlled_request(model, prompt, fed_ids, emitted, io_affected, index, kind, artifact_dir):
    import controlled

    prime_s = prime(artifact_dir)
    t_entry = time.monotonic()
    if fed_ids is not None:
        prefill_s, steps = controlled.run_fixed_continuation(model, prompt, fed_ids)
        record = {
            "index": index,
            "kind": kind,
            "prime_s": prime_s,
            "prime_excluded": True,
            "ttft_s": None,
            "prefill_s": prefill_s,
            "steps": steps,
            "event_s": [],
            "tokens": [],
            "streaming_decode_tps": None,
            "full_generation_s": prefill_s + sum(s["s"] for s in steps),
            "emitted": 0,
            "finish_reason": "fixed-continuation",
            "unused_lookahead_forwards": 0,
            "memory": None,
            "io_affected": io_affected,
        }
    else:
        tokens, prefill_s, steps, rel = controlled.run(model, prompt, emitted)
        n = len(rel)
        streaming = (n - 1) / (rel[-1] - rel[0]) if n >= 2 else None
        record = {
            "index": index,
            "kind": kind,
            "prime_s": prime_s,
            "prime_excluded": True,
            "ttft_s": rel[0],
            "prefill_s": prefill_s,
            "steps": steps,
            "event_s": rel,
            "tokens": tokens,
            "streaming_decode_tps": streaming,
            "full_generation_s": rel[-1],
            "emitted": n,
            "finish_reason": controlled.CONTROLLED_FINISH_REASON,
            "unused_lookahead_forwards": 0,
            "memory": None,
            "io_affected": io_affected,
        }
    return record


def stock_request(mx, stream_generate, model, tokenizer, prompt, max_tokens, io_affected, index, kind, artifact_dir):
    prime_s = prime(artifact_dir)
    gen = stream_generate(model, tokenizer, prompt=list(prompt), max_tokens=max_tokens)
    t_entry = time.monotonic()
    events = []
    tokens = []
    finish_reason = None
    for resp in gen:
        events.append(time.monotonic() - t_entry)
        tokens.append(int(resp.token))
        if resp.finish_reason is not None:
            finish_reason = resp.finish_reason
    # The shipped loop leaves its last lookahead forward queued; drain it and
    # count it (plan section 3.2: the extra forward is reported, not hidden).
    mx.synchronize()
    full_s = time.monotonic() - t_entry
    n = len(tokens)
    streaming = (n - 1) / (events[-1] - events[0]) if n >= 2 else None
    return {
        "index": index,
        "kind": kind,
        "prime_s": prime_s,
        "prime_excluded": True,
        "ttft_s": events[0] if events else None,
        "prefill_s": None,
        "steps": None,
        "event_s": events,
        "tokens": tokens,
        "streaming_decode_tps": streaming,
        "full_generation_s": full_s,
        "emitted": n,
        "finish_reason": finish_reason,
        "unused_lookahead_forwards": 1 if n > 0 else 0,
        "memory": None,
        "io_affected": io_affected,
    }


def main():
    global PRIMING_ENABLED
    require(len(sys.argv) == 2, "usage: worker.py <job.json>")
    job = json.load(open(sys.argv[1]))
    state["utc_start"] = utcnow()
    # Captured before validate so a validation failure can still leave a failed
    # report in the launcher-created out dir (identity stays empty).
    raw_out_dir = job.get("out_dir")
    if isinstance(raw_out_dir, str) and raw_out_dir:
        state["out_dir"] = raw_out_dir
    v = validate(job)
    state.update(
        adapter=v["adapter"], case=v["case"]["name"], prompt_length=v["prompt_length"], mode=v["mode"]
    )
    state["out_dir"] = str(v["out_dir"])
    PRIMING_ENABLED = v["mode"] != "reference-pair"

    # Set before any mlx import (plan section 3.2).
    os.environ["MLX_ENABLE_TF32"] = job["precision_env"]
    import mlx.core as mx

    device = mx.default_device()
    require(device == mx.gpu, f"default device is {device}, expected gpu")
    info = mx.device_info()
    device_info = {
        "device_name": str(info.get("device_name", "")),
        "memory_size_bytes": int(info.get("memory_size", 0)),
        "max_recommended_working_set_bytes": int(info.get("max_recommended_working_set_size", 0)),
    }
    state["stop_token_ids"] = sorted(v["case"]["stop_token_ids"])
    state["identity"] = {
        "code": repo_identity(v["repo_dir"]),
        "manifest_sha256": v["manifest_sha"],
        "run_policy": {
            "worker_timeout_s": v["policy"].get("worker_timeout_s"),
            "reference_pair_worker_timeout_s": v["policy"].get("reference_pair_worker_timeout_s"),
            "cell_time_budget_s": v["policy"].get("cell_time_budget_s"),
            "session_time_budget_s": v["policy"].get("session_time_budget_s"),
            "warmups": v["policy"].get("warmups"),
            "measured_requests": v["policy"].get("measured_requests"),
            "emitted_tokens": v["policy"].get("emitted_tokens"),
            "cache_regime": v["policy"].get("cache_regime"),
        },
        "precision_env": f"MLX_ENABLE_TF32={job['precision_env']}",
        "python": {
            "version": ".".join(map(str, sys.version_info[:3])),
            "venv": sys.prefix,
            "mlx": pkg_version("mlx"),
            "mlx_metal": pkg_version("mlx-metal"),
            "mlx_lm": pkg_version("mlx-lm"),
        },
        "device": device_info,
        "wired_limit_policy": v["wired_limit"],
        "cache_regime": (
            "observed"
            if v["mode"] == "reference-pair"
            else v["policy"].get("cache_regime", "primed-per-request")
        ),
        "io_affected": v["io_affected"],
        "instrumented": False,
    }

    ready = v["out_dir"] / "worker-ready"
    with open(ready, "w") as fid:
        json.dump({"utc": utcnow(), "pid": os.getpid(), "adapter": v["adapter"], "case": v["case"]["name"]}, fid)
        fid.write("\n")

    deadline["t"] = time.monotonic() + v["timeout_s"]
    threading.Thread(target=watchdog, daemon=True).start()

    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.models.cache import make_prompt_cache

    t_load = time.monotonic()
    model, tokenizer = load(str(v["artifact_dir"]))
    state["load_s"] = time.monotonic() - t_load
    state["load_state"] = "materialized"

    if v["adapter"] == "stock":
        missing = [t for t in v["case"]["stop_token_ids"] if t not in tokenizer.eos_token_ids]
        for t in missing:
            tokenizer.add_eos_token(t)
        require(
            sorted(tokenizer.eos_token_ids) == state["stop_token_ids"],
            "resolved stop set differs from the manifest frozen set",
        )
        if missing:
            state["notes"].append(
                f"stock stop set completed to the manifest set via add_eos_token for {missing} "
                "(shipped TokenizerWrapper API, same mechanism as the --extra-eos-token CLI "
                "flag); the final resolved set is asserted equal to the manifest set"
            )
        else:
            state["notes"].append(
                "stock stop set resolved by mlx_lm.load already equals the manifest set "
                "(config.json and generation_config.json eos_token_id are merged into the "
                "tokenizer wrapper by the pinned mlx_lm load path)"
            )
        state["notes"].append(
            "stock retains shipped detokenization and finalization inside the stream; "
            "event times include it. unused_lookahead_forwards is derived from the pinned "
            "loop (one queued forward after the last yielded token) and drained by the "
            "explicit mx.synchronize() at full completion"
        )
    if v["mode"] == "fixed-continuation":
        state["notes"].append(
            "fixed-continuation steps the manifest prefix512[32:64] ids through the "
            "loaded model (stock: shipped model call with last-position slice; "
            "controlled: own-head projection); no sampling, no lookahead pipeline; "
            "qualification workload per plan section 3.3"
        )

    kinds = ["first"] * v["requests"]["first"] + ["warmup"] * v["requests"]["warmups"] + [
        "measured"
    ] * v["requests"]["measured"]
    emitted = v["policy"]["emitted_tokens"]
    for index, kind in enumerate(kinds):
        if deadline["fired"].is_set():
            state["status"] = "timeout"
            state["failure_reason"] = "worker wall-clock timeout exceeded"
            break
        if v["adapter"] == "controlled":
            record = controlled_request(
                model, v["prompt"], v["fed_ids"], emitted, v["io_affected"], index, kind, v["artifact_dir"]
            )
        elif v["mode"] == "fixed-continuation":
            # Stock fixed-continuation: the shipped model call, one id per step,
            # last-position slice (the stock loop's own projection path), no
            # lookahead or cache-clear pipeline.
            from controlled import cache_states

            prime_s = prime(v["artifact_dir"])
            t_entry = time.monotonic()
            from controlled import cache_states

            cache = make_prompt_cache(model)
            logits = model(mx.array(v["prompt"], dtype=mx.int32)[None], cache=cache)
            mx.eval(logits, *cache_states(cache))
            prefill_s = time.monotonic() - t_entry
            steps = []
            for k, fed in enumerate(v["fed_ids"]):
                step_start = time.monotonic()
                logits = model(mx.array([int(fed)], dtype=mx.int32)[None], cache=cache)
                logits = logits[:, -1:, :]
                mx.eval(logits, *cache_states(cache))
                top = mx.sort(mx.reshape(logits[0], (-1,))).tolist()
                steps.append(
                    {
                        "position": len(v["prompt"]) + k + 1,
                        "s": time.monotonic() - step_start,
                        "argmax_id": int(mx.argmax(logits[0]).item()),
                        "top2_margin": float(top[-1] - top[-2]),
                    }
                )
            record = {
                "index": index,
                "kind": kind,
                "prime_s": prime_s,
                "prime_excluded": True,
                "ttft_s": None,
                "prefill_s": prefill_s,
                "steps": steps,
                "event_s": [],
                "tokens": [],
                "streaming_decode_tps": None,
                "full_generation_s": time.monotonic() - t_entry,
                "emitted": 0,
                "finish_reason": "fixed-continuation",
                "unused_lookahead_forwards": 0,
                "memory": None,
                "io_affected": v["io_affected"],
            }
        else:
            record = stock_request(
                mx, stream_generate, model, tokenizer, v["prompt"], emitted, v["io_affected"], index, kind, v["artifact_dir"]
            )
        state["requests"].append(record)

    if state["status"] == "failed" and state["failure_reason"] is None:
        state["status"] = "ok"
    write_report(v["out_dir"])
    if state["status"] == "ok":
        sys.exit(EXIT_OK)
    sys.exit(EXIT_TIMEOUT if state["status"] == "timeout" else EXIT_FAILED)


if __name__ == "__main__":
    try:
        main()
    except WorkerError as exc:
        if state["utc_start"] is None:
            state["utc_start"] = utcnow()
        state["status"] = "failed"
        state["failure_reason"] = str(exc)
        out_dir = state.get("out_dir")
        if out_dir and Path(out_dir).is_dir():
            write_report(Path(out_dir))
        else:
            print(f"worker validation failed: {exc}", file=sys.stderr)
        sys.exit(EXIT_FAILED)
    except Exception as exc:  # unexpected: still leave a failed report when possible
        state["status"] = "failed"
        state["failure_reason"] = f"unexpected: {type(exc).__name__}: {exc}"
        if state.get("out_dir"):
            try:
                write_report(Path(state["out_dir"]))
            except Exception:
                pass
        print(f"worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(EXIT_FAILED)
