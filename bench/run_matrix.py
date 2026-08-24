#!/usr/bin/env python3
"""Four-arm serving benchmark: arm definitions, metric contract, fallback driver.

STATUS: this is the Stage 2 orchestrator STUB. What is fixed and final here is
the CONTRACT — which four arms get run, and which metric row every arm reports.
What is provisional is the load driver: the stdlib streaming driver in this file
is the fallback, and at Stage 3 it is expected to be replaced by a real load
generator (see below) without changing the arms or the metric row.

What this measures. Throughput and latency of one vLLM server under four traffic
shapes, so the cost of serving two tenants from one GPU can be stated against the
cost of serving one:

  base-only             every request routes to the base model. The control.
  base-plus-one-lora    requests alternate base / adapter A. One adapter resident.
  two-lora-round-robin  requests alternate adapter A / adapter B, by request
                        index. Both adapters resident, but which one is in flight
                        at a given instant depends on scheduling luck.
  two-lora-interleaved  adapter is assigned by WORKER SLOT, not by request index,
                        so both adapters are guaranteed in flight simultaneously
                        and vLLM has to batch across them. This is the arm that
                        actually exercises multi-LoRA batching; round-robin can
                        degenerate into alternating homogeneous batches.

How it measures it. Fixed concurrency: N worker threads, each holding exactly one
streaming request open at a time, draining a shared queue of --requests-per-arm
requests. Every completed request appends a raw JSONL row the moment it finishes.
The metric row is then computed by reading that file back off disk. Wall time is
measured from the first request submitted to the last one returned, so
requests_per_sec is a real observed rate at the stated concurrency, not a
derived figure.

Metric row, identical for every arm:

  ttft_p50_s, ttft_p95_s            time to first SSE chunk with content
  itl_p50_s, itl_p95_s              per-request mean inter-token latency,
                                    (last_token_s - ttft_s) / (tokens - 1),
                                    then p50/p95 ACROSS requests
  tokens_per_sec_p50, _p95          per-request output_tokens / e2e_s
  tokens_per_sec_aggregate          sum(output_tokens) / wall_time_s
  requests_per_sec                  n_requests / wall_time_s at fixed concurrency
  e2e_p50_s, e2e_p95_s              request start to [DONE]
  wall_time_s                       first submit to last return
  overhead_vs_base_pct              (arm e2e_p50 / base-only e2e_p50 - 1) * 100

No bare means anywhere. Where a mean is unavoidable (per-request ITL) it is
labelled as one and the distribution is taken over requests.

PREFERRED LOAD GENERATORS, in order. Both are the industry-standard way to
produce these numbers and both report the same metric names, which is why the
contract above was written to match them:

  1. NVIDIA genai-perf (part of Triton perf tooling). Speaks the OpenAI chat
     endpoint, drives fixed concurrency, and reports TTFT / ITL / request
     throughput / output token throughput with full percentiles, including a
     true per-token-pair ITL distribution that this driver cannot produce from
     the raw schema below.
  2. vLLM's own benchmarks/benchmark_serving.py, which is what vLLM's published
     numbers come from, and which understands --model per request so the LoRA
     arms can be expressed directly.

This driver exists so the arms and the metric row are runnable and testable TODAY
against tools/mock_openai_server.py, with no GPU and no pip install. It is not a
better load generator than either of the above and is not meant to be.

Stdlib only. No model client, no pip install.

Examples:
  python bench/run_matrix.py --endpoint http://127.0.0.1:8000 \
      --arms base-only,two-lora-interleaved --requests-per-arm 24 --concurrency 8
  python bench/run_matrix.py --endpoint $SCORING_URI --api-key-env AZURE_KEY \
      --served-names '{"base": "nemotron-nano-8b"}' --requests-per-arm 200
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT / "bench" / "logs"

SYSTEM_MESSAGE = "detailed thinking off"
DEFAULT_PROMPT = "Cut unplanned downtime across our three sites by 30% within a year."
REQUEST_TIMEOUT_S = 300.0
MAX_TOKENS = 512

BASE_ARM = "base-only"
ARMS = (BASE_ARM, "base-plus-one-lora", "two-lora-round-robin", "two-lora-interleaved")

# The metric row schema. Every arm reports exactly these keys, in this order.
METRIC_FIELDS = (
    "arm",
    "concurrency",
    "n_requests",
    "errors",
    "model_mix",
    "ttft_p50_s",
    "ttft_p95_s",
    "itl_p50_s",
    "itl_p95_s",
    "tokens_per_sec_p50",
    "tokens_per_sec_p95",
    "tokens_per_sec_aggregate",
    "requests_per_sec",
    "e2e_p50_s",
    "e2e_p95_s",
    "wall_time_s",
    "overhead_vs_base_pct",
)


# --------------------------------------------------------------------------
# shared plumbing
# --------------------------------------------------------------------------


def chat_url(endpoint: str) -> str:
    url = endpoint.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url + "/v1/chat/completions"


def read_api_key(env_name: str):
    key = os.environ.get(env_name, "").strip()
    if not key:
        print(
            f"run_matrix: env var {env_name} is unset or empty; sending no "
            f"Authorization header.",
            file=sys.stderr,
        )
        return None
    return key


def percentile(values, q):
    """Linear-interpolation percentile. q in [0, 1]. None on an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


def new_run_id() -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond // 1000:03d}Z"


# --------------------------------------------------------------------------
# load driver  (# CHECK: replaced at Stage 3, see docstring)
# --------------------------------------------------------------------------
# CHECK: Stage 3 — if genai-perf is installable on the benchmark VM, replace
# stream_once + drive_arm with a genai-perf invocation per arm:
#   genai-perf profile -m <model> --endpoint-type chat --streaming \
#       --concurrency N --request-count R --url <endpoint>
# and parse its export JSON into the same METRIC_FIELDS row. The arms below are
# expressed as a per-request model assignment, which genai-perf cannot do in one
# process, so the two-lora arms become one genai-perf process per adapter run
# concurrently against the same server, sharing a wall clock.
# CHECK: Stage 3 alternative — vllm/benchmarks/benchmark_serving.py with
#   --backend openai-chat --lora-modules meridian vantage --request-rate inf
# already emits TTFT/ITL/throughput percentiles and understands per-request LoRA
# selection directly. Prefer this if the vLLM source tree is on the VM.
# CHECK: whichever wins, the raw per-request JSONL below must still be written,
# because the anti-fabrication rule is that summaries are computed from files.


def stream_once(url, api_key, model, prompt, timeout=REQUEST_TIMEOUT_S) -> dict:
    """One streaming request. Returns timings; never raises."""
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": MAX_TOKENS,
            "stream": True,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    start_wall = time.time()
    started = time.perf_counter()
    ttft = None
    last_token = None
    n_chunks = 0
    usage_tokens = None
    status = 0
    error = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            for line in response:
                text = line.decode("utf-8", "replace").strip()
                if not text.startswith("data:"):
                    continue
                payload = text[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage_tokens = chunk["usage"].get("completion_tokens")
                choices = chunk.get("choices") or [{}]
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if not content:
                    continue  # role-only or empty chunk: not a token
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - started
                last_token = now - started
                n_chunks += 1
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}"
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    return {
        "start": round(start_wall, 6),
        "http_status": status,
        "ttft_s": round(ttft, 6) if ttft is not None else None,
        "last_token_s": round(last_token, 6) if last_token is not None else None,
        "token_timestamps_count": n_chunks,
        "output_tokens": usage_tokens if usage_tokens is not None else n_chunks,
        "e2e_s": round(time.perf_counter() - started, 6),
        "error": error,
    }


def assigner(arm: str, served: dict):
    """Return f(request_index, worker_slot) -> model name, for one arm."""
    base = served["base"]
    lora_a = served["meridian"]
    lora_b = served["vantage"]
    if arm == "base-only":
        return lambda i, slot: base
    if arm == "base-plus-one-lora":
        return lambda i, slot: base if i % 2 == 0 else lora_a
    if arm == "two-lora-round-robin":
        return lambda i, slot: lora_a if i % 2 == 0 else lora_b
    if arm == "two-lora-interleaved":
        # By worker slot, so both adapters are in flight at the same instant.
        return lambda i, slot: lora_a if slot % 2 == 0 else lora_b
    raise ValueError(f"unknown arm {arm!r}")


def drive_arm(arm, url, api_key, served, n_requests, concurrency, prompt, raw_handle,
              lock, progress=True):
    """Fixed-concurrency streaming driver. Writes raw rows as they land."""
    pick = assigner(arm, served)
    pending = queue.Queue()
    for i in range(n_requests):
        pending.put(i)
    done = [0]

    def worker(slot):
        while True:
            try:
                index = pending.get_nowait()
            except queue.Empty:
                return
            model = pick(index, slot)
            result = stream_once(url, api_key, model, prompt)
            row = {
                "arm": arm,
                "model": model,
                "request_index": index,
                "worker_slot": slot,
                "concurrency": concurrency,
                "start": result["start"],
                "ttft_s": result["ttft_s"],
                "last_token_s": result["last_token_s"],
                "token_timestamps_count": result["token_timestamps_count"],
                "e2e_s": result["e2e_s"],
                "output_tokens": result["output_tokens"],
                "http_status": result["http_status"],
                "error": result["error"],
            }
            with lock:
                raw_handle.write(json.dumps(row) + "\n")
                raw_handle.flush()
                done[0] += 1
                if progress:
                    ttft = f"{result['ttft_s']:.4f}" if result["ttft_s"] is not None else "  n/a "
                    flag = "" if result["http_status"] == 200 else "  ERROR"
                    print(
                        f"run_matrix: [{arm}] {done[0]:>4}/{n_requests} slot {slot} "
                        f"{model:<10} ttft {ttft}s e2e {result['e2e_s']:.4f}s{flag}",
                        flush=True,
                    )

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(worker, slot) for slot in range(max(1, concurrency))]
        for future in futures:
            future.result()
    return round(time.perf_counter() - wall_start, 6)


# --------------------------------------------------------------------------
# metric row, computed from the raw log on disk
# --------------------------------------------------------------------------


def summarise_arm(raw_path: Path, arm: str, wall_time_s: float) -> dict:
    """Read the raw log back and build one metric row. Never uses in-memory state."""
    ttfts, itls, e2es, tps = [], [], [], []
    tokens_total = 0
    n_requests = 0
    errors = 0
    mix = {}

    with raw_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("arm") != arm:
                continue
            n_requests += 1
            mix[row.get("model")] = mix.get(row.get("model"), 0) + 1
            if row.get("http_status") != 200 or row.get("ttft_s") is None:
                errors += 1
                continue
            ttfts.append(row["ttft_s"])
            e2es.append(row["e2e_s"])
            tokens = row.get("output_tokens") or 0
            tokens_total += tokens
            if row["e2e_s"] > 0 and tokens:
                tps.append(tokens / row["e2e_s"])
            count = row.get("token_timestamps_count") or 0
            last = row.get("last_token_s")
            if count > 1 and last is not None:
                # Per-request MEAN inter-token latency. A true per-token-pair
                # distribution needs per-token timestamps; genai-perf reports
                # that and this fallback driver does not keep them.
                itls.append((last - row["ttft_s"]) / (count - 1))

    def r(value, digits=6):
        return round(value, digits) if value is not None else None

    return {
        "arm": arm,
        "concurrency": None,  # filled by the caller, it owns the run config
        "n_requests": n_requests,
        "errors": errors,
        "model_mix": mix,
        "ttft_p50_s": r(percentile(ttfts, 0.50)),
        "ttft_p95_s": r(percentile(ttfts, 0.95)),
        "itl_p50_s": r(percentile(itls, 0.50)),
        "itl_p95_s": r(percentile(itls, 0.95)),
        "tokens_per_sec_p50": r(percentile(tps, 0.50), 3),
        "tokens_per_sec_p95": r(percentile(tps, 0.95), 3),
        "tokens_per_sec_aggregate": r(tokens_total / wall_time_s, 3) if wall_time_s else None,
        "requests_per_sec": r(n_requests / wall_time_s, 3) if wall_time_s else None,
        "e2e_p50_s": r(percentile(e2es, 0.50)),
        "e2e_p95_s": r(percentile(e2es, 0.95)),
        "wall_time_s": r(wall_time_s, 3),
        "overhead_vs_base_pct": None,  # filled once the base arm is known
    }


def apply_overhead(rows: dict) -> None:
    """overhead_vs_base_pct against the base-only arm's e2e p50."""
    base = rows.get(BASE_ARM)
    baseline = base["e2e_p50_s"] if base else None
    for arm, row in rows.items():
        if not baseline:
            row["overhead_vs_base_pct"] = None
            continue
        if row["e2e_p50_s"] is None:
            row["overhead_vs_base_pct"] = None
            continue
        row["overhead_vs_base_pct"] = round((row["e2e_p50_s"] / baseline - 1) * 100, 3)


def print_rows(rows: dict, arms, raw_path: Path, stream=None) -> None:
    stream = stream or sys.stdout  # resolved at call time, not at def time
    show = (
        "arm", "n", "err", "ttft_p50_s", "ttft_p95_s", "itl_p50_s", "itl_p95_s",
        "e2e_p50_s", "e2e_p95_s", "tok/s_agg", "req/s", "vs_base_%",
    )
    keys = (
        "arm", "n_requests", "errors", "ttft_p50_s", "ttft_p95_s", "itl_p50_s",
        "itl_p95_s", "e2e_p50_s", "e2e_p95_s", "tokens_per_sec_aggregate",
        "requests_per_sec", "overhead_vs_base_pct",
    )

    def cell(value):
        if value is None:
            return "not measured"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    table = [show] + [tuple(cell(rows[a][k]) for k in keys) for a in arms if a in rows]
    widths = [max(len(r[i]) for r in table) for i in range(len(show))]

    def line(row):
        return "  ".join(str(c).rjust(widths[i]) for i, c in enumerate(row))

    print("", file=stream)
    print("BENCH MATRIX", file=stream)
    print(f"source: {raw_path}", file=stream)
    print(line(show), file=stream)
    print("-" * len(line(show)), file=stream)
    for row in table[1:]:
        print(line(row), file=stream)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_matrix.py",
        description=(
            "Four-arm serving benchmark. Fixes the arm definitions and the metric "
            "row; drives them with a stdlib streaming load driver until a real "
            "load generator replaces it at Stage 3."
        ),
    )
    parser.add_argument("--endpoint", required=True, help="scoring URI, with or without /v1")
    parser.add_argument(
        "--api-key-env", default="MULTILORA_API_KEY",
        help="name of the env var holding the bearer key (never the key itself)",
    )
    parser.add_argument(
        "--arms", default=",".join(ARMS),
        help="comma-separated arms; choices: " + ", ".join(ARMS),
    )
    parser.add_argument(
        "--served-names", default=None,
        help='JSON mapping base/meridian/vantage -> model field value',
    )
    parser.add_argument("--requests-per-arm", type=int, default=64, help="requests per arm")
    parser.add_argument("--concurrency", type=int, default=8, help="fixed in-flight requests")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="fixed prompt for every request")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="log directory")
    parser.add_argument("--run-id", default=None, help="override the generated run id")
    parser.add_argument("--quiet", action="store_true", help="suppress per-request lines")
    args = parser.parse_args(argv)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        parser.error("unknown arm(s): " + ", ".join(unknown) + "; choices: " + ", ".join(ARMS))
    if args.requests_per_arm < 1:
        parser.error("--requests-per-arm must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if "two-lora-interleaved" in arms and args.concurrency < 2:
        print(
            "run_matrix: WARNING two-lora-interleaved assigns adapters by worker "
            "slot, so at --concurrency 1 it degenerates to a single adapter.",
            file=sys.stderr,
        )

    served = {"base": "base", "meridian": "meridian", "vantage": "vantage"}
    if args.served_names:
        try:
            served.update(json.loads(args.served_names))
        except json.JSONDecodeError as exc:
            parser.error(f"--served-names is not valid JSON: {exc}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or new_run_id()
    raw_path = out_dir / f"matrix_raw_{run_id}.jsonl"
    summary_path = out_dir / f"matrix_summary_{run_id}.json"

    url = chat_url(args.endpoint)
    api_key = read_api_key(args.api_key_env)

    print(f"run_matrix: run {run_id}")
    print(f"run_matrix: endpoint {url}")
    print(f"run_matrix: arms {', '.join(arms)}")
    print(f"run_matrix: {args.requests_per_arm} requests/arm at concurrency {args.concurrency}")
    print(f"run_matrix: served names {json.dumps(served)}")
    print(f"run_matrix: raw log {raw_path}")
    print("run_matrix: driver = stdlib fallback (see docstring: genai-perf / "
          "benchmark_serving.py are the preferred generators)")

    lock = threading.Lock()
    wall = {}
    total_start = time.time()
    with raw_path.open("w", encoding="utf-8") as handle:
        for arm in arms:
            print(f"run_matrix: --- arm {arm} ---", flush=True)
            wall[arm] = drive_arm(
                arm, url, api_key, served, args.requests_per_arm, args.concurrency,
                args.prompt, handle, lock, progress=not args.quiet,
            )
    total_wall = time.time() - total_start

    rows = {}
    for arm in arms:
        row = summarise_arm(raw_path, arm, wall[arm])
        row["concurrency"] = args.concurrency
        rows[arm] = row
    apply_overhead(rows)
    print_rows(rows, arms, raw_path)

    missing = {
        arm: [k for k in METRIC_FIELDS if row.get(k) is None]
        for arm, row in rows.items()
    }
    document = {
        "run_id": run_id,
        "raw_log": str(raw_path),
        "endpoint": url,
        "arms": arms,
        "served_names": served,
        "requests_per_arm": args.requests_per_arm,
        "concurrency": args.concurrency,
        "prompt": args.prompt,
        "driver": "stdlib-fallback",
        "preferred_drivers": ["genai-perf", "vllm/benchmarks/benchmark_serving.py"],
        "metric_fields": list(METRIC_FIELDS),
        "overhead_baseline_arm": BASE_ARM,
        "overhead_definition": "(arm e2e_p50 / base-only e2e_p50 - 1) * 100",
        "itl_definition": "per-request mean (last_token_s - ttft_s)/(tokens-1), p50/p95 across requests",
        "total_wall_time_s": round(total_wall, 3),
        "unset_metric_fields": {a: m for a, m in missing.items() if m},
        "rows": rows,
    }
    summary_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print("")
    print(f"run_matrix: wrote {summary_path}")
    print(f"run_matrix: total wall time {total_wall:.2f}s")

    total_errors = sum(row["errors"] for row in rows.values())
    return 1 if total_errors else 0


if __name__ == "__main__":
    sys.exit(main())
