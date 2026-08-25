#!/usr/bin/env python3
"""Self-test for the `outputs` stage's OpenAI-compatible provider.

What this proves. `data/generate.py outputs --provider openai` can drive a real
OpenAI-shaped chat-completions endpoint end to end — request built, key read
from the environment, response text pulled out of choices[0].message.content,
rows written, errors counted — and the rows it produces land correctly on the
other side of the deterministic filter.

Why it can be proved offline. tools/mock_openai_server.py speaks the same wire
protocol as the endpoint that will actually be paid for out of Azure credit, and
its bodies are fixed: model "meridian" returns a valid Meridian Industrial plan,
model "vantage" a valid Vantage Cloud plan. So the filter's verdict on this
data is known before the test runs — each tenant's own plan passes its own
contract and fails the other one — and a wrong result is a bug in the pipeline,
never a finding about a model.

NOTHING HERE IS A RESULT ABOUT A MODEL. No model is called, no credit is spent,
and no row produced by this file is training data.

Runs under pytest or as a plain script:

  python -m pytest tests/test_generate_openai_provider.py -q
  python tests/test_generate_openai_provider.py
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOALS_FIXTURE = ROOT / "data" / "fixtures" / "dryrun_goals.jsonl"
TMP_DIR = ROOT / "tests" / ".tmp_generate"

N_GOALS = 3
KEY_ENV_SET = "MULTILORA_MOCK_KEY"      # present, so the auth header path runs
KEY_ENV_ABSENT = "MULTILORA_ABSENT_KEY"  # absent, so the loopback allowance runs

MOCK_TTFT_MS = 5.0
MOCK_ITL_MS = 1.0


# --------------------------------------------------------------------------
# loading generate.py by path, so `data` never has to be a package here
# --------------------------------------------------------------------------


def load_tool(alias: str, relpath: str):
    path = ROOT / relpath
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


generate = load_tool("tool_generate", "data/generate.py")


def run_stage(argv):
    """Call generate.main(), echo its stdout, and hand back both."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = generate.main(argv)
    text = buffer.getvalue()
    sys.stdout.write(text)
    sys.stdout.flush()
    return code, text


# --------------------------------------------------------------------------
# mock server control (same shape as tests/test_tooling.py, kept standalone so
# neither self-test can break the other)
# --------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class MockServer:
    def __init__(self, ttft_ms=MOCK_TTFT_MS, itl_ms=MOCK_ITL_MS):
        self.port = free_port()
        self.proc = None
        self.args = (ttft_ms, itl_ms)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        ttft_ms, itl_ms = self.args
        self.proc = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "tools" / "mock_openai_server.py"),
                "--port", str(self.port),
                "--ttft-ms", str(ttft_ms),
                "--itl-ms", str(itl_ms),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.time() + 15
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    "mock server exited during startup:\n" + (self.proc.stdout.read() or "")
                )
            try:
                with urllib.request.urlopen(self.url + "/v1/models", timeout=1) as response:
                    if response.status == 200:
                        print(f"selftest: mock server up on {self.url}")
                        return self
            except Exception:  # noqa: BLE001 - not up yet
                time.sleep(0.05)
        raise RuntimeError(f"mock server did not come up on {self.url}")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


SERVER = None
GOALS_3 = None
_RUNS: dict = {}


def make_goals_file() -> Path:
    """The first N_GOALS goals of the dry-run fixture, marker row kept."""
    path = TMP_DIR / "goals_3.jsonl"
    lines, taken = [], 0
    with GOALS_FIXTURE.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "goal" not in row:
                lines.append(line)  # the SYNTHETIC FIXTURE provenance marker
                continue
            if taken < N_GOALS:
                lines.append(line)
                taken += 1
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def setup_module(module=None):
    global SERVER, GOALS_3
    os.environ[KEY_ENV_SET] = "selftest-not-a-real-key"
    os.environ.pop(KEY_ENV_ABSENT, None)
    _RUNS.clear()
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    GOALS_3 = make_goals_file()
    assert len(generate.read_jsonl(GOALS_3)) == N_GOALS
    SERVER = MockServer().start()


def teardown_module(module=None):
    global SERVER
    if SERVER:
        SERVER.stop()
        SERVER = None
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)


# --------------------------------------------------------------------------
# the two generation runs, shared by the tests below
# --------------------------------------------------------------------------


def outputs_run(model: str, key_env: str):
    """Run the outputs stage against the mock once per model. Cached."""
    if model in _RUNS:
        return _RUNS[model]
    out_dir = TMP_DIR / f"out_{model}"
    code, text = run_stage(
        [
            "outputs",
            "--provider", "openai",
            "--base-url", SERVER.url + "/v1",
            "--api-key-env", key_env,
            "--model", model,
            "--input", str(GOALS_3),
            "--out-dir", str(out_dir),
        ]
    )
    rows = generate.read_jsonl(out_dir / "outputs.jsonl") if code == 0 else []
    _RUNS[model] = (code, text, rows)
    return _RUNS[model]


# --------------------------------------------------------------------------
# 1. the stage itself
# --------------------------------------------------------------------------


def test_outputs_stage_runs_against_the_mock_endpoint():
    # Run A carries a key, so the Authorization header path is exercised.
    code, text, rows = outputs_run("meridian", KEY_ENV_SET)
    assert code == 0, f"outputs stage exited {code}"
    assert len(rows) == len(generate.TENANTS) * N_GOALS == 6, f"{len(rows)} rows"
    assert sorted(r["tenant"] for r in rows) == ["meridian"] * 3 + ["vantage"] * 3
    assert sorted({r["goal_id"] for r in rows}) == [1, 2, 3]
    for row in rows:
        assert row["text"].strip(), f"empty text on {row['goal_id']}/{row['tenant']}"
        assert "error" not in row, f"unexpected error: {row.get('error')}"
        assert row["provider"] == "openai" and row["model"] == "meridian"
    assert "0 errors" in text, text
    assert f"{SERVER.url}/v1/chat/completions" in text, "the URL should be printed"

    # Run B has no key at all. Loopback, so that is allowed and stated.
    code_b, text_b, rows_b = outputs_run("vantage", KEY_ENV_ABSENT)
    assert code_b == 0, f"keyless loopback run exited {code_b}"
    assert len(rows_b) == 6 and all(r["text"].strip() for r in rows_b)
    assert not any("error" in r for r in rows_b)
    assert "0 errors" in text_b
    assert f"{KEY_ENV_ABSENT} is unset" in text_b and "loopback" in text_b


# --------------------------------------------------------------------------
# 2. the rows survive the deterministic filter, per tenant
# --------------------------------------------------------------------------


def test_filter_keeps_each_tenants_own_mock_output():
    _, _, meridian_rows = outputs_run("meridian", KEY_ENV_SET)
    _, _, vantage_rows = outputs_run("vantage", KEY_ENV_ABSENT)

    # Each tenant's rows taken from the run whose adapter name is that tenant:
    # the mock returns that tenant's own valid plan.
    combined = [r for r in meridian_rows if r["tenant"] == "meridian"]
    combined += [r for r in vantage_rows if r["tenant"] == "vantage"]
    assert len(combined) == 6

    combined_path = TMP_DIR / "combined_outputs.jsonl"
    generate.write_jsonl(combined_path, combined)
    out_dir = TMP_DIR / "filter_combined"
    code, _ = run_stage(
        ["filter", "--input", str(combined_path), "--out-dir", str(out_dir)]
    )
    assert code == 0
    summary = json.loads((out_dir / "filter_summary.json").read_text(encoding="utf-8"))
    assert summary["total"] == 6, summary
    assert summary["kept"] == 6, summary
    assert summary["rejected"] == 0, summary
    for tenant in ("meridian", "vantage"):
        stats = summary["per_tenant"][tenant]
        assert stats == {"total": 3, "kept": 3, "rejected": 0}, (tenant, stats)

    # The other direction, from the same data: run A's vantage-tenant rows hold a
    # Meridian plan and must be rejected. If they were not, "kept 6" above would
    # mean the filter accepts anything.
    cross_path = TMP_DIR / "cross_outputs.jsonl"
    generate.write_jsonl(cross_path, meridian_rows)
    cross_dir = TMP_DIR / "filter_cross"
    code, _ = run_stage(["filter", "--input", str(cross_path), "--out-dir", str(cross_dir)])
    assert code == 0
    cross = json.loads((cross_dir / "filter_summary.json").read_text(encoding="utf-8"))
    assert cross["per_tenant"]["meridian"] == {"total": 3, "kept": 3, "rejected": 0}, cross
    assert cross["per_tenant"]["vantage"] == {"total": 3, "kept": 0, "rejected": 3}, cross


# --------------------------------------------------------------------------
# 3. request shape: azure vs plain openai (no network)
# --------------------------------------------------------------------------


def test_url_and_header_shapes():
    azure = generate.openai_chat_url(
        "https://my-resource.openai.azure.com/", "gpt-4o-mini", "2024-10-21"
    )
    assert azure == (
        "https://my-resource.openai.azure.com/openai/deployments/gpt-4o-mini"
        "/chat/completions?api-version=2024-10-21"
    ), azure

    plain = generate.openai_chat_url("http://127.0.0.1:8000/v1", "gpt-4o-mini", None)
    assert plain == "http://127.0.0.1:8000/v1/chat/completions", plain

    azure_headers = generate.openai_headers("k", "2024-10-21")
    assert azure_headers["api-key"] == "k" and "Authorization" not in azure_headers

    plain_headers = generate.openai_headers("k", None)
    assert plain_headers["Authorization"] == "Bearer k" and "api-key" not in plain_headers

    keyless = generate.openai_headers(None, None)
    assert "Authorization" not in keyless and "api-key" not in keyless

    assert generate.is_local_url("http://127.0.0.1:8000/v1")
    assert generate.is_local_url("http://localhost:9000/v1")
    assert not generate.is_local_url("https://my-resource.openai.azure.com")


# --------------------------------------------------------------------------
# 4. a missing key for a real endpoint is a clean refusal (no network)
# --------------------------------------------------------------------------


def test_openai_provider_refusals():
    out_dir = TMP_DIR / "refused"
    code, _ = run_stage(
        ["outputs", "--provider", "openai", "--input", str(GOALS_3), "--out-dir", str(out_dir)]
    )
    assert code == 2, "openai with no --base-url should exit 2"

    code, _ = run_stage(
        [
            "outputs",
            "--provider", "openai",
            "--base-url", "https://my-resource.openai.azure.com",
            "--api-key-env", KEY_ENV_ABSENT,
            "--input", str(GOALS_3),
            "--out-dir", str(out_dir),
        ]
    )
    assert code == 2, "a missing key for a non-loopback endpoint should exit 2"
    assert not (out_dir / "outputs.jsonl").exists(), "a refused run wrote a file"


# --------------------------------------------------------------------------
# 5. a failing request is a failed row, not a failed run
# --------------------------------------------------------------------------


def test_http_errors_are_recorded_not_fatal():
    # A route the mock does not serve: every request 404s. 404 is not retryable,
    # so this costs six round trips and no sleeps.
    out_dir = TMP_DIR / "out_errors"
    code, text = run_stage(
        [
            "outputs",
            "--provider", "openai",
            "--base-url", SERVER.url + "/no-such-route",
            "--api-key-env", KEY_ENV_SET,
            "--model", "meridian",
            "--input", str(GOALS_3),
            "--out-dir", str(out_dir),
        ]
    )
    assert code == 0, "a failing endpoint must not abort the run"
    rows = generate.read_jsonl(out_dir / "outputs.jsonl")
    assert len(rows) == 6, f"{len(rows)} rows"
    assert all(r["text"] == "" and r["error"].startswith("HTTP 404") for r in rows), rows[0]
    assert "0 ok, 6 errors" in text, text


# --------------------------------------------------------------------------
# plain runner
# --------------------------------------------------------------------------

TESTS = [
    test_outputs_stage_runs_against_the_mock_endpoint,
    test_filter_keeps_each_tenants_own_mock_output,
    test_url_and_header_shapes,
    test_openai_provider_refusals,
    test_http_errors_are_recorded_not_fatal,
]


def main() -> int:
    setup_module()
    failures = []
    try:
        for test in TESTS:
            print("")
            print("=" * 78)
            print(f"selftest: {test.__name__}")
            print("=" * 78)
            try:
                test()
            except AssertionError as exc:
                failures.append((test.__name__, f"AssertionError: {exc}"))
                print(f"selftest: FAIL {test.__name__}: {exc}")
            except Exception as exc:  # noqa: BLE001
                failures.append((test.__name__, f"{type(exc).__name__}: {exc}"))
                print(f"selftest: ERROR {test.__name__}: {exc}")
            else:
                print(f"selftest: PASS {test.__name__}")
    finally:
        teardown_module()

    print("")
    print("-" * 78)
    print(f"selftest: {len(TESTS) - len(failures)}/{len(TESTS)} passed")
    for name, detail in failures:
        print(f"selftest:   FAIL {name}: {detail}")
    print("selftest: " + ("PASS" if not failures else "FAIL"))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
