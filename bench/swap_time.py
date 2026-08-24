#!/usr/bin/env python3
"""Cold vs warm LoRA adapter swap time, measured in isolation.

What this measures. The extra time-to-first-token paid by the FIRST request that
routes to a LoRA adapter the server has not served yet (the cold swap: the
adapter has to be read off disk and moved onto the device), and the time paid by
every request after that (the warm swap: on vLLM this should be a pointer change
into an already-resident adapter, so roughly nothing).

Why this tool is hand-rolled when nothing else here is. A load generator reports
a TTFT distribution over many requests. The cold swap is a single event that
happens once per adapter per server lifetime, and it is buried inside that
distribution as one sample among hundreds. Isolating it needs control over
request ORDER, which load generators deliberately do not give you. This is the
only timing tool in this repo that is not delegated; bench/run_matrix.py hands
throughput measurement to a real load generator.

Method, in order:

  1. --n-warm streaming requests to --baseline-model, which is resident and
     never swaps. This is the warm floor: TTFT with no adapter work in it.
  2. ONE streaming request to --adapter. Its TTFT is the cold sample.
  3. --n-warm more streaming requests to --adapter. This is the same adapter
     once it is resident.

Reported:

  cold_ttft_s            step 2, a single sample
  adapter_warm_ttft      p50/p95 over step 3
  baseline_ttft          p50/p95 over step 1
  cold_swap_estimate_s   cold_ttft_s - adapter_warm_p50
  warm_swap_estimate_s   adapter_warm_p50 - baseline_p50   (expected near zero)

TTFT is measured from the moment the request is written to the moment the first
SSE chunk carrying non-empty `delta.content` arrives. A role-only opening chunk
does not count, because vLLM emits one before prefill has produced anything.

ASSUMPTION, and it is load-bearing: step 2 is a cold sample ONLY if --adapter has
never been requested since the server started. If anything has touched that
adapter first, the number is a warm sample wearing a cold label. The tool cannot
verify this from the client side, so it states the assumption in the console
output and in the summary JSON, and the operator has to honour it by restarting
the server before the run. n=1 by nature: a cold load happens once.

Raw per-request rows are written to bench/logs/ as they arrive; the summary is
computed by reading that file back and carries its path.

Stdlib only. No model client, no pip install.

Examples:
  python bench/swap_time.py --endpoint http://127.0.0.1:8000 --adapter meridian
  python bench/swap_time.py --endpoint $SCORING_URI --api-key-env AZURE_KEY \
      --adapter vantage --baseline-model nemotron-nano-8b --n-warm 30
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT / "bench" / "logs"

SYSTEM_MESSAGE = "detailed thinking off"
DEFAULT_PROMPT = "Cut unplanned downtime across our three sites by 30% within a year."
REQUEST_TIMEOUT_S = 300.0
MAX_TOKENS = 256


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
            f"swap_time: env var {env_name} is unset or empty; sending no "
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
    started = time.perf_counter()
    ttft = None
    token_times = []
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
                choices = chunk.get("choices") or [{}]
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if not content:
                    continue  # role-only or empty chunk: not a token
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - started
                token_times.append(now - started)
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}"
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    return {
        "http_status": status,
        "ttft_s": round(ttft, 6) if ttft is not None else None,
        "e2e_s": round(time.perf_counter() - started, 6),
        "output_tokens": len(token_times),
        "error": error,
    }


def new_run_id() -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond // 1000:03d}Z"


# --------------------------------------------------------------------------
# the measurement
# --------------------------------------------------------------------------

COLD_ASSUMPTION = (
    "The cold sample is valid ONLY if the adapter had never been requested since "
    "the server started. This tool cannot verify that from the client side. "
    "Restart the server before the run, or treat the cold number as warm."
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="swap_time.py",
        description=(
            "Isolate the first-request cost of loading a LoRA adapter (cold swap) "
            "from the per-request cost once it is resident (warm swap), by "
            "controlling request order."
        ),
    )
    parser.add_argument("--endpoint", required=True, help="scoring URI, with or without /v1")
    parser.add_argument(
        "--api-key-env", default="MULTILORA_API_KEY",
        help="name of the env var holding the bearer key (never the key itself)",
    )
    parser.add_argument("--adapter", required=True, help="adapter model name to measure")
    parser.add_argument(
        "--baseline-model", default="base",
        help="resident model that never swaps, used as the warm floor (default base)",
    )
    parser.add_argument("--n-warm", type=int, default=20, help="warm samples per phase")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="fixed short prompt")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="log directory")
    parser.add_argument("--run-id", default=None, help="override the generated run id")
    parser.add_argument("--quiet", action="store_true", help="suppress per-request lines")
    args = parser.parse_args(argv)

    if args.n_warm < 1:
        parser.error("--n-warm must be at least 1")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or new_run_id()
    raw_path = out_dir / f"swap_time_raw_{run_id}.jsonl"
    summary_path = out_dir / f"swap_time_summary_{run_id}.json"

    url = chat_url(args.endpoint)
    api_key = read_api_key(args.api_key_env)

    print(f"swap_time: run {run_id}")
    print(f"swap_time: endpoint {url}")
    print(f"swap_time: adapter {args.adapter}  baseline {args.baseline_model}  "
          f"n_warm {args.n_warm}")
    print(f"swap_time: raw log {raw_path}")
    print("")
    print("swap_time: ASSUMPTION " + COLD_ASSUMPTION)
    print("")

    # Phases run strictly in order. Order IS the measurement here.
    plan = (
        [("baseline_warm", args.baseline_model)] * args.n_warm
        + [("adapter_cold", args.adapter)]
        + [("adapter_warm", args.adapter)] * args.n_warm
    )

    wall_start = time.time()
    with raw_path.open("w", encoding="utf-8") as handle:
        for index, (phase, model) in enumerate(plan):
            result = stream_once(url, api_key, model, args.prompt)
            row = {
                "run_id": run_id,
                "seq": index,
                "phase": phase,
                "model": model,
                "http_status": result["http_status"],
                "ttft_s": result["ttft_s"],
                "e2e_s": result["e2e_s"],
                "output_tokens": result["output_tokens"],
                "error": result["error"],
            }
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if not args.quiet:
                ttft = f"{result['ttft_s']:.4f}s" if result["ttft_s"] is not None else "no-ttft"
                flag = "" if result["http_status"] == 200 else "  ERROR"
                print(
                    f"swap_time: [{index + 1:>3}/{len(plan)}] {phase:<13} {model:<12} "
                    f"ttft {ttft:>10}  e2e {result['e2e_s']:.4f}s{flag}",
                    flush=True,
                )
    wall_s = time.time() - wall_start

    # Summary is computed from the file on disk, not from anything in memory.
    phases = {"baseline_warm": [], "adapter_cold": [], "adapter_warm": []}
    errors = 0
    rows = 0
    with raw_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows += 1
            if row["http_status"] != 200 or row["ttft_s"] is None:
                errors += 1
                continue
            phases.setdefault(row["phase"], []).append(row["ttft_s"])

    cold_list = phases["adapter_cold"]
    cold = cold_list[0] if cold_list else None
    warm_p50 = percentile(phases["adapter_warm"], 0.50)
    warm_p95 = percentile(phases["adapter_warm"], 0.95)
    base_p50 = percentile(phases["baseline_warm"], 0.50)
    base_p95 = percentile(phases["baseline_warm"], 0.95)

    cold_swap = (cold - warm_p50) if (cold is not None and warm_p50 is not None) else None
    warm_swap = (warm_p50 - base_p50) if (warm_p50 is not None and base_p50 is not None) else None

    def r(value):
        return round(value, 6) if value is not None else None

    summary = {
        "run_id": run_id,
        "raw_log": str(raw_path),
        "endpoint": url,
        "adapter": args.adapter,
        "baseline_model": args.baseline_model,
        "n_warm": args.n_warm,
        "prompt": args.prompt,
        "rows_in_raw_log": rows,
        "errors": errors,
        "wall_time_s": round(wall_s, 3),
        "cold_assumption": COLD_ASSUMPTION,
        "ttft_definition": "request start to first SSE chunk with non-empty delta.content",
        "cold_ttft_s": r(cold),
        "cold_ttft_n": len(cold_list),
        "adapter_warm_ttft_p50_s": r(warm_p50),
        "adapter_warm_ttft_p95_s": r(warm_p95),
        "adapter_warm_n": len(phases["adapter_warm"]),
        "baseline_ttft_p50_s": r(base_p50),
        "baseline_ttft_p95_s": r(base_p95),
        "baseline_n": len(phases["baseline_warm"]),
        "cold_swap_estimate_s": r(cold_swap),
        "warm_swap_estimate_s": r(warm_swap),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    def fmt(value, unit="s"):
        return "not measured" if value is None else f"{value:.4f}{unit}"

    print("")
    print("SWAP TIME")
    print(f"source: {raw_path}")
    print(f"  baseline {args.baseline_model:<14} ttft p50 {fmt(base_p50)}   "
          f"p95 {fmt(base_p95)}   n={summary['baseline_n']}")
    print(f"  adapter  {args.adapter:<14} ttft p50 {fmt(warm_p50)}   "
          f"p95 {fmt(warm_p95)}   n={summary['adapter_warm_n']}")
    print(f"  adapter  {args.adapter:<14} ttft cold {fmt(cold)}                  n=1")
    print(f"  cold_swap_estimate  = cold - adapter_p50    = {fmt(cold_swap)}")
    print(f"  warm_swap_estimate  = adapter_p50 - base_p50 = {fmt(warm_swap)}")
    print("")
    print(f"swap_time: wrote {summary_path}")
    print(f"swap_time: wall time {wall_s:.2f}s over {rows} requests, {errors} errors")
    if errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
