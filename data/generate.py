#!/usr/bin/env python3
"""Staged data pipeline for the two-tenant multi-LoRA project.

Four stages, run independently so each one can be inspected and re-run:

  goals    write N tenant-neutral leadership-intent statements as JSONL
  outputs  for each goal x each tenant, ask the model for the tenant JSON
  filter   run every output through data.verifier.verify, split kept/rejected
  package  turn kept rows into chat-format training JSONL, one file per tenant

Only the `outputs` stage talks to the network. It has two providers:

  --provider anthropic  (default) the anthropic SDK, imported lazily inside the
                        stage so nothing else in the file needs it installed.
  --provider openai     any OpenAI-compatible chat-completions endpoint, spoken
                        with stdlib urllib.request and no new dependency. That
                        covers Azure OpenAI / Azure AI Foundry deployments —
                        which is how generation gets paid for out of Azure
                        credit instead of an Anthropic key — and it also covers
                        tools/mock_openai_server.py, which is what makes this
                        stage self-testable offline with no key at all.

Everything else is stdlib, so the filter and package stages stay runnable with
no API key and no network.

Examples:
  python data/generate.py goals --n-goals 200 --seed 7
  python data/generate.py outputs --model claude-sonnet-5
  python data/generate.py outputs --provider openai --model gpt-4o-mini \\
      --base-url https://my-resource.openai.azure.com \\
      --api-key-env AZURE_OPENAI_API_KEY --azure-api-version 2024-10-21
  python data/generate.py filter --input data/generated/outputs.jsonl
  python data/generate.py package --goals data/generated/goals.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.verifier import TENANTS, VOCAB, verify  # noqa: E402

DEFAULT_MODEL = "claude-sonnet-5"  # CHECK: model id
DEFAULT_OUT_DIR = ROOT / "data" / "generated"

MAX_TOKENS = 2000
# Sampling parameters (temperature/top_p/top_k) are rejected with a 400 by the
# current Claude 5 model family, so no temperature is sent; pass --temperature
# only for older models that still accept it.

SYSTEM_TRAINING_MESSAGE = "detailed thinking off"


# --------------------------------------------------------------------------
# tenant prompts
# --------------------------------------------------------------------------

TENANT_PROFILE = {
    "meridian": {
        "name": "Meridian Industrial",
        "blurb": (
            "a regulated industrial manufacturer. House voice is formal, "
            "process-led and audit-aware. Plans move through gate reviews and "
            "leave a documented trail."
        ),
        "schema": """{
  "initiative": "str",
  "work_packages": [
    {"id": "WP-01", "title": "str", "owner_role": "str",
     "duration_quarters": 1, "deliverable": "str"}
  ],
  "approval_chain": [
    {"gate": "G1", "name": "str", "approver_role": "str", "criteria": "str"}
  ],
  "compliance_notes": ["str"],
  "risks": [{"risk": "str", "mitigation": "str", "severity": "high|medium|low"}],
  "timeline_horizon": "str"
}""",
        "rules": [
            "3 to 6 work_packages; ids run WP-01, WP-02, ... consecutively from WP-01",
            "duration_quarters is a positive integer",
            "2 to 4 approval_chain gates; ids run G1, G2, ... consecutively from G1",
            "compliance_notes holds at least one non-empty string",
            "at least one risk; severity is exactly high, medium or low",
            "no keys beyond the ones shown; every string is non-empty",
        ],
    },
    "vantage": {
        "name": "Vantage Cloud",
        "blurb": (
            "a fast-moving SaaS company. House voice is terse and outcome-led. "
            "Plans are sprints, owners and shipped increments."
        ),
        "schema": """{
  "initiative": "str",
  "okrs": [{"objective": "str", "key_results": ["str"]}],
  "sprint_plan": [
    {"sprint": 1, "weeks": "1-2", "focus": "str", "owner": "str", "ships": "str"}
  ],
  "blockers": ["str"],
  "success_metric": "str",
  "timeline_weeks": 12
}""",
        "rules": [
            "1 or 2 okrs; each has at least one non-empty key result",
            "3 to 6 sprint_plan entries; sprint is an integer counting up from 1",
            "weeks looks like \"1-2\" or \"3\"",
            "blockers is a list of strings and may be empty",
            "timeline_weeks is a positive integer, not a string",
            "no keys beyond the ones shown; every string is non-empty",
        ],
    },
}


def _readable_terms(patterns: list) -> str:
    """Turn the verifier regexes into plain words for the prompt."""
    words = {
        r"\bwork packages?\b": "work package",
        r"\bgate reviews?\b": "gate review",
        r"\bgates?\b": "gate",
        r"\bdeviations?\b": "deviation",
        r"\bsign[- ]?offs?\b": "sign-off",
        r"\bdeliverables?\b": "deliverable",
        r"\bcontrols?\b": "control",
        r"\bnon[- ]?conformances?\b": "non-conformance",
        r"\bsprints?\b": "sprint",
        r"\bokrs?\b": "OKR",
        r"\bship(s|ped|ping)?\b": "ship",
        r"\bowners?\b": "owner",
        r"\bblockers?\b": "blocker",
        r"\biterat(e|es|ed|ing|ion|ions)\b": "iterate",
        r"\bweeks?\b": "week",
        r"\bmetrics?\b": "metric",
        r"\bquarters?\b": "quarter",
    }
    return ", ".join(words.get(p, p) for p in patterns)


def build_prompt(tenant: str, goal: str) -> str:
    """The user prompt for one goal and one tenant."""
    profile = TENANT_PROFILE[tenant]
    rules = "\n".join(f"- {r}" for r in profile["rules"])
    own = _readable_terms(VOCAB[tenant]["own"])
    forbidden = _readable_terms(VOCAB[tenant]["forbidden"])
    return f"""You are the planning function inside {profile['name']}, {profile['blurb']}

Convert the leadership intent below into an execution plan as a single JSON
object with exactly this shape:

{profile['schema']}

Schema rules:
{rules}

House vocabulary. Use at least two distinct terms from this list in the prose
values: {own}

Never use any of these terms anywhere in the output, in any grammatical form:
{forbidden}

Leadership intent:
{goal}"""


GENERATOR_SYSTEM_PROMPT = (
    "You produce structured execution plans. Output ONLY the JSON object. "
    "No markdown fences, no commentary before or after, no explanation. "
    "The first character of your response is { and the last is }."
)


# --------------------------------------------------------------------------
# jsonl helpers
# --------------------------------------------------------------------------

NOTE_KEY = "_note"


def read_jsonl(path: Path) -> list:
    """Read a JSONL file, dropping the synthetic-fixture marker row."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: not valid JSON: {exc}")
            if NOTE_KEY in row and "text" not in row and "goal" not in row:
                continue  # provenance marker, not a data row
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# stage: goals
# --------------------------------------------------------------------------

GOAL_TEMPLATES = [
    "Cut unplanned downtime across our sites by {pct}% within {horizon}.",
    "Bring a second source online for our most constrained component within {horizon}.",
    "Reduce the cost of serving our largest accounts by {pct}% without hurting reliability.",
    "Get our onboarding time for new customers down by {pct}% in {horizon}.",
    "Move the reporting stack off the legacy platform within {horizon}.",
    "Raise on-time delivery to {high}% across every product family.",
    "Halve the time it takes to close the books each period.",
    "Make security review stop being the bottleneck on new work within {horizon}.",
    "Increase throughput on our slowest line by {pct}% without new capital spend.",
    "Consolidate our three overlapping tools into one within {horizon}.",
    "Cut the defect escape rate by {pct}% before the next audit cycle.",
    "Stand up predictive maintenance on the equipment that fails most often.",
    "Reduce customer-reported incidents by {pct}% in {horizon}.",
    "Get a self-serve path to first value working within {horizon}.",
    "Bring energy use per unit down {pct}% at the two highest-consuming sites.",
    "Shorten the cycle from design freeze to first production output by {pct}%.",
    "Make our capacity forecast accurate enough to plan hiring against.",
    "Reduce dependence on the single vendor behind our core workflow.",
    "Improve first-contact resolution to {high}% in the support organisation.",
    "Cut the time from raw data to a decision-ready number by {pct}%.",
    "Replace the manual approval chain on spend under a threshold within {horizon}.",
    "Get every site reporting the same operational numbers the same way.",
    "Reduce rework on the highest-volume product family by {pct}%.",
    "Make disaster recovery something we can prove, not something we assume.",
    "Lift utilisation of the newest equipment to {high}% of rated capacity.",
    "Take {pct}% out of the cost of our compliance reporting process.",
    "Get new hires productive in half the time they take today.",
    "Bring the top three sources of customer churn under active control.",
    "Reduce inventory tied up in slow-moving stock by {pct}%.",
    "Make our forecast-to-plan handoff work without a spreadsheet.",
    "Cut the lead time on custom orders by {pct}% within {horizon}.",
    "Get quality data flowing from the floor without manual entry.",
    "Reduce the number of systems that hold customer records to one.",
    "Improve throughput of the review process by {pct}% without adding headcount.",
    "Make capacity constraints visible before they bite, not after.",
    "Reduce our exposure to a single-site failure within {horizon}.",
    "Bring the cost per transaction down {pct}% at current volume.",
    "Get a working plan for entering the adjacent market within {horizon}.",
    "Cut the time we spend reconciling numbers between systems by {pct}%.",
    "Make the handoff between engineering and operations repeatable.",
]

HORIZONS = [
    "six months",
    "a year",
    "two quarters",
    "the next planning period",
    "eighteen months",
    "nine months",
]


def stage_goals(args) -> int:
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    n = args.n_goals

    pool = list(GOAL_TEMPLATES)
    rng.shuffle(pool)
    goals = []
    for i in range(n):
        template = pool[i % len(pool)]
        goal = template.format(
            pct=rng.choice([10, 15, 20, 25, 30, 40, 50]),
            high=rng.choice([95, 96, 97, 98]),
            horizon=rng.choice(HORIZONS),
        )
        goals.append({"goal_id": i + 1, "goal": goal})

    path = out_dir / "goals.jsonl"
    write_jsonl(path, goals)
    print(f"goals: wrote {len(goals)} goals to {path} (seed={args.seed})")
    return 0


# --------------------------------------------------------------------------
# stage: outputs
# --------------------------------------------------------------------------
#
# Two providers, one request shape: the same system prompt, the same single user
# message, the same max_tokens, and no sampling parameters unless --temperature
# is passed. Only the transport differs.
#
# Which generator produced a row is deliberately not load-bearing. Every row
# goes through data.verifier.verify in the filter stage, which is deterministic,
# has no model client and no network, so swapping the generator changes cost and
# throughput, never what is allowed into training.


PROVIDERS = ("anthropic", "openai")

DEFAULT_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

# A key is mandatory for a real endpoint. It is optional against a loopback
# address, because that is the offline mock in tools/ and it has no auth.
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", ""})

REQUEST_TIMEOUT_S = 300.0
MAX_ATTEMPTS = 3
# Fixed, not jittered: this is a batch job, not a latency-sensitive path, and a
# fixed schedule keeps a failing run's wall-clock cost predictable. Sleeps are
# taken BETWEEN attempts, so MAX_ATTEMPTS=3 uses the first two entries.
RETRY_SLEEPS_S = (2.0, 4.0, 8.0)


def is_local_url(url: str) -> bool:
    """True if the URL points at loopback, where a missing key is acceptable."""
    try:
        host = (urllib.parse.urlsplit(url).hostname or "").lower()
    except ValueError:
        return False
    return host in LOCAL_HOSTS or host.startswith("127.")


def read_api_key(env_name: str, provider: str, base_url):
    """Return (key, exit_code). The key is read from the environment only.

    There is no --api-key flag and there will not be one: a key on a command
    line ends up in shell history, in `ps`, and in CI logs.
    """
    key = os.environ.get(env_name, "").strip()
    if key:
        return key, None
    if provider == "openai" and base_url and is_local_url(base_url):
        print(
            f"outputs: {env_name} is unset; {base_url} is loopback, so no auth "
            f"header is sent. Fine against the mock, wrong against Azure."
        )
        return None, None
    print(f"{env_name} is not set; the outputs stage cannot run.", file=sys.stderr)
    return None, 2


def openai_chat_url(base_url: str, model: str, azure_api_version=None) -> str:
    """The chat-completions URL for one request.

    Two shapes, both documented on learn.microsoft.com (checked 2026-08-25):

    Azure OpenAI, deployment route — used when --azure-api-version is given:
        POST {base}/openai/deployments/{deployment}/chat/completions
             ?api-version={VER}
      with the key in an `api-key` header. Verified against
      https://learn.microsoft.com/en-us/azure/ai-foundry/openai/reference
      ("POST https://YOUR_RESOURCE_NAME.openai.azure.com/openai/deployments/
       YOUR_DEPLOYMENT_NAME/chat/completions?api-version=2024-06-01", and
      "all API requests must include the API Key in the `api-key` HTTP header").
      So --base-url is the bare resource endpoint here, e.g.
      https://my-resource.openai.azure.com — no /v1 suffix.

    Plain OpenAI — used when --azure-api-version is omitted:
        POST {base}/chat/completions
      with `Authorization: Bearer KEY`. This also reaches Azure's newer v1 API
      by passing --base-url https://my-resource.openai.azure.com/openai/v1,
      which the same docs describe as OpenAI-client compatible with no
      api-version parameter.

    # CHECK: the path segment is the *deployment* name, which Azure does not
    # force to equal the model id. This function reuses --model for both, which
    # is right only when the deployment was named after the model. Verify on the
    # live resource before a paid run; a mismatch is a 404 (DeploymentNotFound).
    # CHECK: which api-version string the target resource accepts. Versions are
    # dated, are retired on a published schedule, and the docs list several as
    # current. Nothing here can pick one without the live resource.
    # CHECK: Azure AI Foundry models that are NOT Azure OpenAI (serverless
    # Foundry Models — Mistral, DeepSeek, Llama) are documented on a different
    # route, "POST /chat/completions?api-version=..." under a /models base with
    # `Authorization: Bearer`, not /openai/deployments/. For those, omit
    # --azure-api-version and give the full base path in --base-url; the exact
    # base path was not confirmed from the docs.
    """
    base = base_url.rstrip("/")
    if azure_api_version:
        deployment = urllib.parse.quote(model, safe="")
        version = urllib.parse.quote(azure_api_version, safe="")
        return f"{base}/openai/deployments/{deployment}/chat/completions?api-version={version}"
    return f"{base}/chat/completions"


def openai_headers(api_key, azure_api_version=None) -> dict:
    """Auth header for one request. Azure key auth is `api-key`, not Bearer."""
    headers = {"Content-Type": "application/json"}
    if not api_key:
        return headers
    if azure_api_version:
        # CHECK: `api-key` is the documented header for Azure OpenAI key auth
        # (Entra ID uses `Authorization: Bearer <token>` instead). The docs' own
        # OpenAI-client examples for the v1 route send a key as Bearer, so the
        # two headers are not interchangeable across the two routes and this
        # pairing — api-key with the deployment route — is the documented one.
        headers["api-key"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def post_openai_chat(url, headers, body_bytes, timeout=REQUEST_TIMEOUT_S):
    """One chat completion. Returns (text, error). Never raises.

    Retries HTTP 429 and 5xx up to MAX_ATTEMPTS times on the fixed schedule.
    Anything else — a 400, a 404, a bad body — is returned as an error on the
    first try. An exhausted or non-retryable failure is a failed row, not a
    failed run: the caller records it and keeps going.
    """
    last_error = "no attempt was made"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        request = urllib.request.Request(url, data=body_bytes, headers=headers, method="POST")
        retryable = False
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            text = payload["choices"][0]["message"]["content"]
            return (text or "").strip(), None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            last_error = f"HTTP {exc.code}: {detail}"
            retryable = exc.code == 429 or exc.code >= 500
        except urllib.error.URLError as exc:
            last_error = f"URLError: {exc.reason}"
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            last_error = f"unexpected response shape: {type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001 - a dead endpoint is a data point
            last_error = f"{type(exc).__name__}: {exc}"

        if not retryable or attempt == MAX_ATTEMPTS:
            break
        sleep_s = RETRY_SLEEPS_S[min(attempt, len(RETRY_SLEEPS_S)) - 1]
        print(
            f"outputs:   attempt {attempt}/{MAX_ATTEMPTS} failed ({last_error[:120]}); "
            f"retrying in {sleep_s:.0f}s"
        )
        time.sleep(sleep_s)
    return "", last_error


def build_anthropic_caller(args):
    """Return (caller, exit_code). caller(tenant, goal) -> (text, error)."""
    env_name = args.api_key_env or DEFAULT_KEY_ENV["anthropic"]
    _, code = read_api_key(env_name, "anthropic", None)
    if code is not None:
        return None, code

    try:
        import anthropic  # noqa: PLC0415 - lazy on purpose, only this stage needs it
    except ImportError:
        print("the anthropic package is not installed; pip install anthropic", file=sys.stderr)
        return None, 2

    # The SDK reads ANTHROPIC_API_KEY itself; an alternate env name is passed on.
    client_kwargs = {}
    if env_name != "ANTHROPIC_API_KEY":
        client_kwargs["api_key"] = os.environ[env_name].strip()
    client = anthropic.Anthropic(**client_kwargs)

    request_kwargs = {}
    if args.temperature is not None:
        request_kwargs["temperature"] = args.temperature

    def call(tenant: str, goal: str):
        try:
            response = client.messages.create(
                model=args.model,
                max_tokens=MAX_TOKENS,
                system=GENERATOR_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_prompt(tenant, goal)}],
                **request_kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - one bad row must not kill the run
            return "", f"{type(exc).__name__}: {exc}"
        text = "".join(block.text for block in response.content if block.type == "text")
        return text.strip(), None

    return call, 0


def build_openai_caller(args):
    """Return (caller, exit_code). caller(tenant, goal) -> (text, error)."""
    if not args.base_url:
        print(
            "--base-url is required with --provider openai (e.g. "
            "https://my-resource.openai.azure.com or http://127.0.0.1:8000/v1)",
            file=sys.stderr,
        )
        return None, 2

    env_name = args.api_key_env or DEFAULT_KEY_ENV["openai"]
    api_key, code = read_api_key(env_name, "openai", args.base_url)
    if code is not None:
        return None, code

    url = openai_chat_url(args.base_url, args.model, args.azure_api_version)
    headers = openai_headers(api_key, args.azure_api_version)
    style = "azure" if args.azure_api_version else "openai"
    print(f"outputs: provider openai ({style} style) -> {url}")

    def call(tenant: str, goal: str):
        body = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(tenant, goal)},
            ],
            "max_tokens": MAX_TOKENS,
            "stream": False,
        }
        if args.temperature is not None:
            body["temperature"] = args.temperature
        return post_openai_chat(url, headers, json.dumps(body).encode("utf-8"))

    return call, 0


def stage_outputs(args) -> int:
    if args.provider not in PROVIDERS:  # argparse already guards this
        print(f"unknown provider {args.provider!r}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    goals_path = Path(args.input) if args.input else out_dir / "goals.jsonl"
    if not goals_path.exists():
        print(f"no goals file at {goals_path}; run the goals stage first", file=sys.stderr)
        return 2

    builder = build_openai_caller if args.provider == "openai" else build_anthropic_caller
    call, code = builder(args)
    if call is None:
        return code

    goals = read_jsonl(goals_path)
    rows = []
    errors = 0
    for goal_row in goals:
        for tenant in TENANTS:
            text, error = call(tenant, goal_row["goal"])
            row = {
                "goal_id": goal_row["goal_id"],
                "goal": goal_row["goal"],
                "tenant": tenant,
                "text": text,
                "model": args.model,
                "provider": args.provider,
            }
            if error:
                errors += 1
                row["error"] = error
                print(f"outputs: goal {goal_row['goal_id']} {tenant} ERROR {error[:200]}")
            else:
                print(f"outputs: goal {goal_row['goal_id']} {tenant} ({len(text)} chars)")
            rows.append(row)

    path = out_dir / "outputs.jsonl"
    write_jsonl(path, rows)
    print(
        f"outputs: wrote {len(rows)} outputs to {path} "
        f"({len(rows) - errors} ok, {errors} errors, provider={args.provider}, "
        f"model={args.model})"
    )
    return 0


# --------------------------------------------------------------------------
# stage: filter
# --------------------------------------------------------------------------


def _reason(result: dict) -> str:
    if any(e.startswith("json parse error") for e in result["errors"]):
        return "parse"
    if not result["schema_ok"] and not result["vocab_ok"]:
        return "schema+vocab"
    if not result["schema_ok"]:
        return "schema"
    return "vocab"


def stage_filter(args) -> int:
    out_dir = Path(args.out_dir)
    input_path = Path(args.input) if args.input else out_dir / "outputs.jsonl"
    if not input_path.exists():
        print(f"no outputs file at {input_path}", file=sys.stderr)
        return 2

    rows = read_jsonl(input_path)
    kept, rejected = [], []
    per_tenant = {t: {"total": 0, "kept": 0, "rejected": 0} for t in TENANTS}
    reason_counts: dict = {}

    for row in rows:
        tenant = row.get("tenant")
        if tenant not in TENANTS:
            rejected.append(
                {**row, "reason": "unknown-tenant", "errors": [f"unknown tenant {tenant!r}"]}
            )
            reason_counts["unknown-tenant"] = reason_counts.get("unknown-tenant", 0) + 1
            continue

        per_tenant[tenant]["total"] += 1
        result = verify(row.get("text", ""), tenant)
        if result["ok"]:
            kept.append(row)
            per_tenant[tenant]["kept"] += 1
        else:
            reason = _reason(result)
            rejected.append({**row, "reason": reason, "errors": result["errors"]})
            per_tenant[tenant]["rejected"] += 1
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    total = len(rows)
    rate = (len(rejected) / total * 100) if total else 0.0
    summary = {
        "input": str(input_path),
        "total": total,
        "kept": len(kept),
        "rejected": len(rejected),
        "rejection_rate_pct": round(rate, 2),
        "per_tenant": per_tenant,
        "reasons": reason_counts,
    }

    write_jsonl(out_dir / "kept.jsonl", kept)
    write_jsonl(out_dir / "rejected.jsonl", rejected)
    (out_dir / "filter_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(f"filter: input {input_path}")
    print(f"filter: total {total}  kept {len(kept)}  rejected {len(rejected)}  "
          f"rejection rate {rate:.2f}%")
    for tenant in TENANTS:
        stats = per_tenant[tenant]
        print(f"filter:   {tenant}: total {stats['total']} kept {stats['kept']} "
              f"rejected {stats['rejected']}")
    if reason_counts:
        for reason, count in sorted(reason_counts.items()):
            print(f"filter:   reason {reason}: {count}")
    for row in rejected:
        print(f"filter:   REJECT goal_id={row.get('goal_id')} tenant={row.get('tenant')} "
              f"reason={row.get('reason')}")
        for err in row.get("errors", []):
            print(f"filter:     {err}")
    print(f"filter: wrote {out_dir / 'kept.jsonl'}, {out_dir / 'rejected.jsonl'}, "
          f"{out_dir / 'filter_summary.json'}")
    return 0


# --------------------------------------------------------------------------
# stage: package
# --------------------------------------------------------------------------


def stage_package(args) -> int:
    out_dir = Path(args.out_dir)
    kept_path = Path(args.input) if args.input else out_dir / "kept.jsonl"
    if not kept_path.exists():
        print(f"no kept file at {kept_path}; run the filter stage first", file=sys.stderr)
        return 2
    kept = read_jsonl(kept_path)

    goal_index = {}
    goals_path = Path(args.goals) if args.goals else out_dir / "goals.jsonl"
    if goals_path.exists():
        for row in read_jsonl(goals_path):
            goal_index[row["goal_id"]] = row["goal"]

    counts = {t: 0 for t in TENANTS}
    missing_goal = 0
    for tenant in TENANTS:
        rows = []
        for row in kept:
            if row.get("tenant") != tenant:
                continue
            goal = row.get("goal") or goal_index.get(row.get("goal_id"))
            if not goal:
                missing_goal += 1
                continue
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_TRAINING_MESSAGE},
                        {"role": "user", "content": goal},
                        {"role": "assistant", "content": row["text"]},
                    ]
                }
            )
        path = out_dir / f"train_{tenant}.jsonl"
        write_jsonl(path, rows)
        counts[tenant] = len(rows)
        print(f"package: {tenant}: {len(rows)} rows -> {path}")

    if missing_goal:
        print(f"package: skipped {missing_goal} kept rows with no goal text "
              f"(looked in {goals_path})")
    print(f"package: total {sum(counts.values())} training rows")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

STAGES = {
    "goals": stage_goals,
    "outputs": stage_outputs,
    "filter": stage_filter,
    "package": stage_package,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="generate.py",
        description="Staged data pipeline: goals -> outputs -> filter -> package.",
    )
    parser.add_argument("stage", choices=sorted(STAGES), help="pipeline stage to run")
    parser.add_argument("--n-goals", type=int, default=50, help="goals stage: how many goals")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="outputs stage: model id; with --provider openai on Azure this is "
             "the deployment name that goes in the URL path",
    )
    parser.add_argument("--seed", type=int, default=0, help="goals stage: random seed")
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="outputs stage: sampling temperature; omit for Claude 5 models, which reject it",
    )
    parser.add_argument(
        "--provider", choices=PROVIDERS, default="anthropic",
        help="outputs stage: which API to call (default anthropic)",
    )
    parser.add_argument(
        "--base-url",
        help="outputs stage, --provider openai: endpoint base, e.g. "
             "https://my-resource.openai.azure.com (with --azure-api-version), "
             "https://my-resource.openai.azure.com/openai/v1, or "
             "http://127.0.0.1:8000/v1 for the mock. Required for openai.",
    )
    parser.add_argument(
        "--api-key-env",
        help="outputs stage: env var holding the API key (default "
             "ANTHROPIC_API_KEY for anthropic, OPENAI_API_KEY for openai). The "
             "key is only ever read from the environment, never from a flag.",
    )
    parser.add_argument(
        "--azure-api-version",
        help="outputs stage, --provider openai: send Azure OpenAI style — "
             "POST {base-url}/openai/deployments/{model}/chat/completions"
             "?api-version=VER with an api-key header. Omit for plain OpenAI "
             "style (POST {base-url}/chat/completions, Authorization: Bearer).",
    )
    parser.add_argument(
        "--out-dir", default=str(DEFAULT_OUT_DIR), help="directory for stage outputs"
    )
    parser.add_argument(
        "--input", help="filter/package stage: read from this file instead of --out-dir"
    )
    parser.add_argument(
        "--goals", help="package stage: goals JSONL used to resolve goal text by goal_id"
    )
    args = parser.parse_args(argv)
    return STAGES[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
