#!/usr/bin/env python3
"""Adapter separation harness: the final proof artifact for this project.

What this measures. For every arm (base, meridian, vantage) and every goal in a
held-out goal set, it sends one chat completion to the serving endpoint with the
system message exactly "detailed thinking off" and the goal as the user message,
then checks the returned text against BOTH tenant contracts with
data.verifier.verify. The result is a 3x2 confusion matrix: how often each arm
produces output that satisfies Meridian rules, and how often it satisfies
Vantage rules.

How it measures it. Adapter selection is the `model` field on the request, which
is how vLLM routes to a LoRA with --enable-lora: "base" is the base model's
served name, "meridian" and "vantage" are adapter names. Requests are
non-streaming and run at fixed concurrency. Every response is appended to a raw
JSONL log AS IT ARRIVES, one row per request. The matrix is then computed by
reading that file back from disk, not from anything held in memory, and the
matrix JSON records the path of the raw log it was computed from. A number in
the matrix that has no corresponding row in the raw log cannot exist.

What a good result looks like. The diagonal is high and the off-diagonal is at
floor: the meridian arm passes Meridian rules and fails Vantage rules, the
vantage arm the reverse, and the base arm fails both. The base row is the
control. If the base model can satisfy a tenant contract on its own, the
adapters are not what produced the separation, so a base row above 10% on either
tenant prints a warning block.

Sealed set. The headline number has to come from a goal set that was fixed
before it was ever run against. --sealed refuses to run unless eval/SEALED.sha256
exists and matches the sha256 of the goals file, and refuses a second sealed run
in the same output directory unless --allow-rerun is passed. Write the hash once,
with --make-sealed-hash, before the first sealed run.

Model-free by construction. This file imports urllib.request and nothing else
for network work. No model-client library appears anywhere under eval/; see
scripts/check_no_model_imports.sh, which fails the build if one does.

Examples:
  python eval/separation.py --make-sealed-hash eval/goals_sealed.jsonl
  python eval/separation.py --endpoint http://127.0.0.1:8000 \
      --api-key-env MULTILORA_API_KEY --goals eval/goals_sealed.jsonl --sealed
  python eval/separation.py --endpoint $SCORING_URI --api-key-env AZURE_KEY \
      --goals eval/goals_dev.jsonl --arms base,meridian --max-concurrency 8
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.verifier import TENANTS, verify  # noqa: E402

SYSTEM_MESSAGE = "detailed thinking off"  # identical to the training system message
DEFAULT_ARMS = ("base", "meridian", "vantage")
DEFAULT_OUT_DIR = ROOT / "eval" / "logs"
SEALED_HASH_PATH = ROOT / "eval" / "SEALED.sha256"
BASE_LEAK_THRESHOLD_PCT = 10.0
REQUEST_TIMEOUT_S = 300.0
MAX_TOKENS = 2000


# --------------------------------------------------------------------------
# endpoint plumbing
# --------------------------------------------------------------------------


def chat_url(endpoint: str) -> str:
    """Normalise an endpoint to its chat-completions URL.

    Azure ML managed endpoints expose a scoring URI that may already carry the
    OpenAI path or may not. Accept both rather than making the caller guess.
    """
    url = endpoint.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url + "/v1/chat/completions"


def read_api_key(env_name: str):
    """Read the bearer key from the environment. Never accept it as a flag."""
    key = os.environ.get(env_name, "").strip()
    if not key:
        print(
            f"separation: env var {env_name} is unset or empty; sending no "
            f"Authorization header. Fine against a local mock, wrong against Azure.",
            file=sys.stderr,
        )
        return None
    return key


def post_chat(url: str, api_key, model: str, goal: str, timeout=REQUEST_TIMEOUT_S) -> dict:
    """One non-streaming chat completion. Returns a raw record, never raises."""
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": goal},
            ],
            "max_tokens": MAX_TOKENS,
            "stream": False,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            text = payload["choices"][0]["message"]["content"]
            return {
                "http_status": response.status,
                "latency_s": round(time.perf_counter() - started, 4),
                "text": text,
                "error": None,
            }
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        return {
            "http_status": exc.code,
            "latency_s": round(time.perf_counter() - started, 4),
            "text": "",
            "error": f"HTTP {exc.code}: {detail}",
        }
    except Exception as exc:  # noqa: BLE001 - a dead endpoint is a data point
        return {
            "http_status": 0,
            "latency_s": round(time.perf_counter() - started, 4),
            "text": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


# --------------------------------------------------------------------------
# goals + sealed set
# --------------------------------------------------------------------------


def read_goals(path: Path) -> list:
    """Read {"goal_id", "goal"} rows, skipping the synthetic-fixture marker."""
    goals = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: not valid JSON: {exc}")
            if "goal" not in row:
                continue  # provenance marker row
            goals.append({"goal_id": row.get("goal_id", lineno), "goal": row["goal"]})
    return goals


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def make_sealed_hash(goals_path: Path, hash_path: Path) -> int:
    """Write the sealed-set hash file. Refuses to overwrite an existing one."""
    if not goals_path.exists():
        print(f"separation: no goals file at {goals_path}", file=sys.stderr)
        return 2
    if hash_path.exists():
        print(
            f"separation: {hash_path} already exists and will not be overwritten.\n"
            f"            The sealed set is fixed once. Delete it by hand if you "
            f"really mean to re-seal, and say so in change_log.md.",
            file=sys.stderr,
        )
        return 2
    digest = sha256_file(goals_path)
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    # sha256sum output format, so `sha256sum -c eval/SEALED.sha256` also works.
    rel = os.path.relpath(goals_path, hash_path.parent.parent)
    hash_path.write_text(f"{digest}  {rel}\n", encoding="utf-8")
    print(f"separation: sealed {goals_path}")
    print(f"separation:   sha256 {digest}")
    print(f"separation:   wrote {hash_path}")
    return 0


def check_sealed(goals_path: Path, hash_path: Path, out_dir: Path, allow_rerun: bool) -> int:
    """Return 0 if a sealed run is allowed, else a nonzero exit code."""
    if not hash_path.exists():
        print(
            f"separation: --sealed requires {hash_path}, which does not exist.\n"
            f"            Seal the goal set first:\n"
            f"              python eval/separation.py --make-sealed-hash {goals_path}",
            file=sys.stderr,
        )
        return 2

    recorded = hash_path.read_text(encoding="utf-8").split()
    if not recorded:
        print(f"separation: {hash_path} is empty", file=sys.stderr)
        return 2
    expected, recorded_name = recorded[0], (recorded[1] if len(recorded) > 1 else "?")
    actual = sha256_file(goals_path)
    if actual != expected:
        print(
            f"separation: sealed-set mismatch, refusing to run.\n"
            f"            goals file : {goals_path}\n"
            f"            sha256 now : {actual}\n"
            f"            sha256 seal: {expected}  ({recorded_name})\n"
            f"            The sealed set changed after it was sealed. Either restore "
            f"the original file or run without --sealed and label the result as "
            f"unsealed.",
            file=sys.stderr,
        )
        return 2

    prior = sorted(out_dir.glob("separation_matrix_*sealed*.json")) if out_dir.exists() else []
    if prior and not allow_rerun:
        print(
            f"separation: a sealed run already exists in {out_dir}:\n"
            + "".join(f"              {p.name}\n" for p in prior)
            + f"            The sealed set is run once. Pass --allow-rerun to do it "
            f"again on purpose, and record why in change_log.md.",
            file=sys.stderr,
        )
        return 2
    return 0


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


class RawLog:
    """Append-only JSONL sink. Every request lands here before it is counted."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8")
        self._lock = threading.Lock()
        self.count = 0

    def write(self, row: dict) -> None:
        with self._lock:
            self._handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._handle.flush()
            self.count += 1

    def close(self) -> None:
        self._handle.close()


def run_requests(url, api_key, arms, served, goals, raw_log, max_concurrency, progress=True):
    """Fire arm x goal requests at fixed concurrency, logging each as it lands."""
    jobs = [(arm, goal) for arm in arms for goal in goals]
    done = [0]
    lock = threading.Lock()

    def one(job):
        arm, goal = job
        model = served[arm]
        result = post_chat(url, api_key, model, goal["goal"])
        row = {
            "arm": arm,
            "goal_id": goal["goal_id"],
            "model": model,
            "http_status": result["http_status"],
            "latency_s": result["latency_s"],
            "text": result["text"],
            "error": result["error"],
        }
        raw_log.write(row)
        with lock:
            done[0] += 1
            if progress:
                flag = "" if result["http_status"] == 200 else "  ERROR"
                print(
                    f"separation: [{done[0]:>4}/{len(jobs)}] {arm:<9} "
                    f"goal {goal['goal_id']:<5} http {result['http_status']} "
                    f"{result['latency_s']:.3f}s{flag}",
                    flush=True,
                )
        return row

    with ThreadPoolExecutor(max_workers=max(1, max_concurrency)) as pool:
        list(pool.map(one, jobs))
    return len(jobs)


# --------------------------------------------------------------------------
# matrix, computed from the raw log on disk
# --------------------------------------------------------------------------


def compute_matrix(raw_path: Path, arms) -> dict:
    """Read the raw log back off disk and score every row under both tenants."""
    counts = {
        arm: {"n": 0, "errors": 0, "meridian_pass": 0, "vantage_pass": 0} for arm in arms
    }
    with raw_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            arm = row.get("arm")
            if arm not in counts:
                counts[arm] = {"n": 0, "errors": 0, "meridian_pass": 0, "vantage_pass": 0}
            bucket = counts[arm]
            bucket["n"] += 1
            if row.get("http_status") != 200:
                bucket["errors"] += 1
            text = row.get("text") or ""
            for tenant in TENANTS:
                if verify(text, tenant)["ok"]:
                    bucket[f"{tenant}_pass"] += 1

    matrix = {}
    for arm, bucket in counts.items():
        n = bucket["n"]
        matrix[arm] = {
            "meridian_pass_rate": round(bucket["meridian_pass"] / n, 4) if n else None,
            "vantage_pass_rate": round(bucket["vantage_pass"] / n, 4) if n else None,
            "n": n,
            "errors": bucket["errors"],
            "meridian_pass_count": bucket["meridian_pass"],
            "vantage_pass_count": bucket["vantage_pass"],
        }
    return matrix


def _cell(passes, n) -> str:
    if not n:
        return "not measured"
    return f"{passes / n * 100:.1f}% ({passes}/{n})"


def print_matrix(matrix: dict, arms, raw_path: Path, stream=None) -> None:
    stream = stream or sys.stdout  # resolved at call time, not at def time
    header = ("arm", "passes Meridian rules", "passes Vantage rules", "n", "errors")
    rows = [header]
    for arm in arms:
        entry = matrix.get(arm)
        if not entry:
            rows.append((arm, "not measured", "not measured", "0", "0"))
            continue
        rows.append(
            (
                arm,
                _cell(entry["meridian_pass_count"], entry["n"]),
                _cell(entry["vantage_pass_count"], entry["n"]),
                str(entry["n"]),
                str(entry["errors"]),
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]

    def line(row):
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row))

    print("", file=stream)
    print("SEPARATION MATRIX", file=stream)
    print(f"source: {raw_path}", file=stream)
    print(line(header), file=stream)
    print("-" * len(line(header)), file=stream)
    for row in rows[1:]:
        print(line(row), file=stream)


def base_sanity_check(matrix: dict, stream=None) -> bool:
    """Loud warning if the base arm satisfies a tenant contract on its own."""
    stream = stream or sys.stdout  # resolved at call time, not at def time
    entry = matrix.get("base")
    if not entry or not entry["n"]:
        return False
    mer = (entry["meridian_pass_rate"] or 0) * 100
    van = (entry["vantage_pass_rate"] or 0) * 100
    if mer <= BASE_LEAK_THRESHOLD_PCT and van <= BASE_LEAK_THRESHOLD_PCT:
        return False
    bar = "!" * 74
    print("", file=stream)
    print(bar, file=stream)
    print("WARNING: the base arm satisfies a tenant contract without an adapter.", file=stream)
    print(f"         base passes Meridian rules {mer:.1f}%", file=stream)
    print(f"         base passes Vantage  rules {van:.1f}%", file=stream)
    print(f"         threshold is {BASE_LEAK_THRESHOLD_PCT:.0f}%.", file=stream)
    print("         The separation number is not attributable to the adapters", file=stream)
    print("         until this is explained. Check that the base arm is really", file=stream)
    print("         hitting the base served name and not a loaded adapter.", file=stream)
    print(bar, file=stream)
    return True


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def new_run_id() -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond // 1000:03d}Z"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="separation.py",
        description=(
            "Per-arm confusion matrix: how often each of base / meridian / vantage "
            "produces output that passes each tenant's deterministic contract."
        ),
    )
    parser.add_argument("--endpoint", help="scoring URI, with or without the /v1 path")
    parser.add_argument(
        "--api-key-env", default="MULTILORA_API_KEY",
        help="name of the env var holding the bearer key (never the key itself)",
    )
    parser.add_argument("--goals", help="goals JSONL, rows of {\"goal_id\", \"goal\"}")
    parser.add_argument(
        "--arms", default=",".join(DEFAULT_ARMS),
        help="comma-separated arms to run (default base,meridian,vantage)",
    )
    parser.add_argument(
        "--served-names", default=None,
        help='JSON object mapping arm -> model field value, e.g. \'{"base": "nemotron-nano-8b"}\'',
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="where logs are written")
    parser.add_argument("--max-concurrency", type=int, default=4, help="in-flight requests")
    parser.add_argument("--run-id", default=None, help="override the generated run id")
    parser.add_argument(
        "--sealed", action="store_true",
        help="require eval/SEALED.sha256 to match the goals file, and refuse a repeat run",
    )
    parser.add_argument(
        "--allow-rerun", action="store_true",
        help="permit a second sealed run in the same output directory",
    )
    parser.add_argument(
        "--make-sealed-hash", metavar="FILE", default=None,
        help="write eval/SEALED.sha256 for FILE and exit; refuses to overwrite",
    )
    parser.add_argument(
        "--sealed-hash-path", default=str(SEALED_HASH_PATH),
        help=f"location of the sealed hash file (default {SEALED_HASH_PATH})",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress per-request progress")
    args = parser.parse_args(argv)

    hash_path = Path(args.sealed_hash_path)

    if args.make_sealed_hash:
        return make_sealed_hash(Path(args.make_sealed_hash), hash_path)

    missing = [name for name in ("endpoint", "goals") if not getattr(args, name)]
    if missing:
        parser.error("missing required argument(s): " + ", ".join("--" + m for m in missing))

    goals_path = Path(args.goals)
    if not goals_path.exists():
        print(f"separation: no goals file at {goals_path}", file=sys.stderr)
        return 2

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if not arms:
        parser.error("--arms is empty")
    served = {arm: arm for arm in arms}
    if args.served_names:
        try:
            served.update(json.loads(args.served_names))
        except json.JSONDecodeError as exc:
            parser.error(f"--served-names is not valid JSON: {exc}")
    unmapped = [arm for arm in arms if arm not in served]
    if unmapped:
        parser.error("--served-names has no entry for: " + ", ".join(unmapped))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sealed:
        code = check_sealed(goals_path, hash_path, out_dir, args.allow_rerun)
        if code:
            return code

    goals = read_goals(goals_path)
    if not goals:
        print(f"separation: {goals_path} has no goal rows", file=sys.stderr)
        return 2

    run_id = args.run_id or new_run_id()
    tag = f"sealed_{run_id}" if args.sealed else run_id
    raw_path = out_dir / f"separation_raw_{tag}.jsonl"
    matrix_path = out_dir / f"separation_matrix_{tag}.json"
    url = chat_url(args.endpoint)
    api_key = read_api_key(args.api_key_env)

    print(f"separation: run {run_id}{' (SEALED)' if args.sealed else ''}")
    print(f"separation: endpoint {url}")
    print(f"separation: goals {goals_path} ({len(goals)} goals)")
    print(f"separation: arms {', '.join(f'{a}->{served[a]}' for a in arms)}")
    print(f"separation: concurrency {args.max_concurrency}")
    print(f"separation: raw log {raw_path}")

    raw_log = RawLog(raw_path)
    wall_start = time.time()
    try:
        expected = run_requests(
            url, api_key, arms, served, goals, raw_log,
            args.max_concurrency, progress=not args.quiet,
        )
    finally:
        raw_log.close()
    wall_s = time.time() - wall_start

    if raw_log.count != expected:
        print(
            f"separation: WARNING raw log holds {raw_log.count} rows, expected {expected}",
            file=sys.stderr,
        )

    matrix = compute_matrix(raw_path, arms)
    print_matrix(matrix, arms, raw_path)
    leaked = base_sanity_check(matrix)

    document = {
        "run_id": run_id,
        "sealed": args.sealed,
        "raw_log": str(raw_path),
        "goals_file": str(goals_path),
        "goals_sha256": sha256_file(goals_path),
        "n_goals": len(goals),
        "endpoint": url,
        "arms": arms,
        "served_names": {arm: served[arm] for arm in arms},
        "max_concurrency": args.max_concurrency,
        "system_message": SYSTEM_MESSAGE,
        "wall_time_s": round(wall_s, 3),
        "base_leak_warning": leaked,
        "matrix": matrix,
    }
    # Spec shape at the top level too, so a reader gets {arm: {...}} directly.
    document.update(matrix)
    matrix_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print("")
    print(f"separation: wrote {matrix_path}")
    print(f"separation: wall time {wall_s:.2f}s over {raw_log.count} requests")

    total_errors = sum(entry["errors"] for entry in matrix.values())
    return 1 if total_errors else 0


if __name__ == "__main__":
    sys.exit(main())
