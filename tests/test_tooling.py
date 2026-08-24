#!/usr/bin/env python3
"""Self-test for the measurement tooling, against a local mock server.

What this proves. Every measurement tool in this repo produces the right number
when the right answer is already known. The mock server in tools/ has fixed
timings and fixed response bodies, so the correct output of each tool is
arithmetic, not opinion:

  separation.py   the mock returns a valid Meridian plan for model "meridian",
                  a valid Vantage plan for "vantage" and non-JSON prose for
                  "base", so the confusion matrix MUST be 100/0, 0/100, 0/0.
                  Anything else is a bug in the harness, not a finding.
  swap_time.py    the mock adds exactly --cold-first-request-ms to the first
                  request for each non-base model, so cold_swap_estimate must
                  come back at that value and warm_swap_estimate at zero.
  run_matrix.py   every field in the metric row must be populated.
  economics.py    N=20 must give 320 GB against 17.6 GB.
  make_report.py  the report must carry those numbers and cite the file each
                  one came from.

NOTHING HERE IS A RESULT ABOUT A MODEL. No model is called, no adapter exists
yet, and every number produced by this file is synthetic by construction. The
report it generates, reports/iter_00.md, says so at the top.

Runs under pytest or as a plain script:

  python -m pytest tests/test_tooling.py -q
  python tests/test_tooling.py
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
FIXTURE_GOALS = ROOT / "tests" / "fixtures" / "tooling_goals.jsonl"
TMP_DIR = ROOT / "tests" / ".tmp"
EVAL_LOGS = ROOT / "eval" / "logs"
BENCH_LOGS = ROOT / "bench" / "logs"
REPORTS = ROOT / "reports"
REAL_SEALED_PATH = ROOT / "eval" / "SEALED.sha256"

RUN_ID = "selftest"
MOCK_TTFT_MS = 80.0
MOCK_ITL_MS = 10.0
MOCK_COLD_MS = 400.0

SEPARATION_MATRIX = EVAL_LOGS / f"separation_matrix_{RUN_ID}.json"
SEPARATION_RAW = EVAL_LOGS / f"separation_raw_{RUN_ID}.jsonl"
SWAP_SUMMARY = BENCH_LOGS / f"swap_time_summary_{RUN_ID}.json"
SWAP_RAW = BENCH_LOGS / f"swap_time_raw_{RUN_ID}.jsonl"
BENCH_SUMMARY = BENCH_LOGS / f"matrix_summary_{RUN_ID}.json"
BENCH_RAW = BENCH_LOGS / f"matrix_raw_{RUN_ID}.jsonl"
ECONOMICS_MD = BENCH_LOGS / f"economics_{RUN_ID}.md"
ITER_00 = REPORTS / "iter_00.md"

ITER_00_BANNER = (
    "TOOLING SELF-TEST against local mock server — no real model, numbers are "
    "synthetic by construction."
)


# --------------------------------------------------------------------------
# loading the tools by path, so `eval` never has to be importable as a package
# --------------------------------------------------------------------------


def load_tool(alias: str, relpath: str):
    path = ROOT / relpath
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


separation = load_tool("tool_separation", "eval/separation.py")
swap_time = load_tool("tool_swap_time", "bench/swap_time.py")
run_matrix = load_tool("tool_run_matrix", "bench/run_matrix.py")
economics = load_tool("tool_economics", "bench/economics.py")
make_report = load_tool("tool_make_report", "scripts/make_report.py")


def run_tool(module, argv):
    """Call a tool's main(), echo its stdout, and hand back both."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = module.main(argv)
    text = buffer.getvalue()
    sys.stdout.write(text)
    sys.stdout.flush()
    return code, text


# --------------------------------------------------------------------------
# mock server control
# --------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class MockServer:
    """The mock as a child process, so 'never requested yet' really means it."""

    def __init__(self, ttft_ms=MOCK_TTFT_MS, itl_ms=MOCK_ITL_MS, cold_ms=MOCK_COLD_MS):
        self.port = free_port()
        self.proc = None
        self.args = (ttft_ms, itl_ms, cold_ms)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        ttft_ms, itl_ms, cold_ms = self.args
        self.proc = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "tools" / "mock_openai_server.py"),
                "--port", str(self.port),
                "--ttft-ms", str(ttft_ms),
                "--itl-ms", str(itl_ms),
                "--cold-first-request-ms", str(cold_ms),
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
                        print(
                            f"selftest: mock server up on {self.url} "
                            f"(ttft={ttft_ms}ms itl={itl_ms}ms cold={cold_ms}ms)"
                        )
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

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


SERVER = None


def setup_module(module=None):
    """One shared server for the tests that do not need a virgin adapter."""
    global SERVER
    os.environ["MULTILORA_API_KEY"] = "selftest-not-a-real-key"
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_LOGS.mkdir(parents=True, exist_ok=True)
    BENCH_LOGS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    SERVER = MockServer().start()


def teardown_module(module=None):
    global SERVER
    if SERVER:
        SERVER.stop()
        SERVER = None
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    # The repo must never carry a sealed hash created by a test.
    assert not REAL_SEALED_PATH.exists(), (
        f"{REAL_SEALED_PATH} exists after the self-test; a test created a real "
        f"sealed hash and did not clean it up"
    )


# --------------------------------------------------------------------------
# 1. separation
# --------------------------------------------------------------------------


def test_separation_matrix_is_the_known_answer():
    code, out = run_tool(
        separation,
        [
            "--endpoint", SERVER.url,
            "--api-key-env", "MULTILORA_API_KEY",
            "--goals", str(FIXTURE_GOALS),
            "--out-dir", str(EVAL_LOGS),
            "--max-concurrency", "4",
            "--run-id", RUN_ID,
            "--quiet",
        ],
    )
    assert code == 0, f"separation.py exited {code}"

    document = json.loads(SEPARATION_MATRIX.read_text(encoding="utf-8"))
    matrix = document["matrix"]

    assert matrix["meridian"]["meridian_pass_rate"] == 1.0, matrix["meridian"]
    assert matrix["meridian"]["vantage_pass_rate"] == 0.0, matrix["meridian"]
    assert matrix["vantage"]["vantage_pass_rate"] == 1.0, matrix["vantage"]
    assert matrix["vantage"]["meridian_pass_rate"] == 0.0, matrix["vantage"]
    assert matrix["base"]["meridian_pass_rate"] == 0.0, matrix["base"]
    assert matrix["base"]["vantage_pass_rate"] == 0.0, matrix["base"]
    for arm in ("base", "meridian", "vantage"):
        assert matrix[arm]["errors"] == 0, matrix[arm]

    # raw row count = arms x goals, and the summary was computed from that file
    goals = separation.read_goals(FIXTURE_GOALS)
    assert len(goals) == 6, f"fixture should hold 6 goals, has {len(goals)}"
    rows = [l for l in SEPARATION_RAW.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 3 * len(goals) == 18, f"raw log has {len(rows)} rows"
    assert document["raw_log"] == str(SEPARATION_RAW)

    # the printed confusion matrix carries the same cells
    assert "100.0% (6/6)" in out
    assert "0.0% (0/6)" in out
    assert "passes Meridian rules" in out and "passes Vantage rules" in out
    # base arm is at floor, so no leak warning
    assert document["base_leak_warning"] is False
    assert "WARNING" not in out


def test_separation_sealed_refuses_without_hash_file():
    assert not REAL_SEALED_PATH.exists(), (
        "this test asserts the behaviour when eval/SEALED.sha256 is absent, and "
        "it is present"
    )
    code, out = run_tool(
        separation,
        [
            "--endpoint", SERVER.url,
            "--goals", str(FIXTURE_GOALS),
            "--out-dir", str(TMP_DIR / "sealed_no_hash"),
            "--sealed",
            "--quiet",
        ],
    )
    assert code == 2, f"expected refusal (exit 2), got {code}"
    # refusal means refusal: no requests were sent and no log was written
    stray = list((TMP_DIR / "sealed_no_hash").glob("separation_*")) \
        if (TMP_DIR / "sealed_no_hash").exists() else []
    assert stray == [], f"a refused sealed run still wrote {stray}"


def test_separation_sealed_roundtrip_and_tamper():
    # A copy under tests/, so the real eval/SEALED.sha256 is never created.
    goals_copy = TMP_DIR / "sealed_goals.jsonl"
    shutil.copyfile(FIXTURE_GOALS, goals_copy)
    hash_path = TMP_DIR / "SEALED.sha256"
    out_dir = TMP_DIR / "sealed_out"

    code, _ = run_tool(
        separation,
        ["--make-sealed-hash", str(goals_copy), "--sealed-hash-path", str(hash_path)],
    )
    assert code == 0 and hash_path.exists()

    # sealing twice is refused
    code, _ = run_tool(
        separation,
        ["--make-sealed-hash", str(goals_copy), "--sealed-hash-path", str(hash_path)],
    )
    assert code == 2, "re-sealing an existing hash file should be refused"

    sealed_args = [
        "--endpoint", SERVER.url,
        "--api-key-env", "MULTILORA_API_KEY",
        "--goals", str(goals_copy),
        "--out-dir", str(out_dir),
        "--sealed",
        "--sealed-hash-path", str(hash_path),
        "--max-concurrency", "4",
        "--quiet",
    ]
    code, out = run_tool(separation, sealed_args)
    assert code == 0, f"sealed run on the sealed file should work, exited {code}"
    produced = list(out_dir.glob("separation_matrix_*sealed*.json"))
    assert len(produced) == 1, f"expected one sealed matrix, got {produced}"

    # a second sealed run in the same out-dir is refused
    code, _ = run_tool(separation, sealed_args)
    assert code == 2, "a repeat sealed run should be refused without --allow-rerun"

    # a tampered goals file is refused
    with goals_copy.open("a", encoding="utf-8") as handle:
        handle.write('{"goal_id": 99, "goal": "an extra goal added after sealing"}\n')
    code, out = run_tool(
        separation,
        [
            "--endpoint", SERVER.url,
            "--goals", str(goals_copy),
            "--out-dir", str(TMP_DIR / "sealed_tampered"),
            "--sealed",
            "--sealed-hash-path", str(hash_path),
            "--quiet",
        ],
    )
    assert code == 2, "a goals file changed after sealing should be refused"

    hash_path.unlink()
    assert not REAL_SEALED_PATH.exists()


# --------------------------------------------------------------------------
# 2. swap time
# --------------------------------------------------------------------------


def test_swap_time_recovers_the_injected_cold_penalty():
    # A dedicated server: the cold sample is only valid if the adapter has never
    # been requested since start, which is exactly what swap_time.py asserts.
    with MockServer() as fresh:
        code, out = run_tool(
            swap_time,
            [
                "--endpoint", fresh.url,
                "--api-key-env", "MULTILORA_API_KEY",
                "--adapter", "meridian",
                "--baseline-model", "base",
                "--n-warm", "8",
                "--out", str(BENCH_LOGS),
                "--run-id", RUN_ID,
                "--quiet",
            ],
        )
    assert code == 0, f"swap_time.py exited {code}"

    summary = json.loads(SWAP_SUMMARY.read_text(encoding="utf-8"))
    cold_swap = summary["cold_swap_estimate_s"]
    warm_swap = summary["warm_swap_estimate_s"]
    injected = MOCK_COLD_MS / 1000.0

    assert cold_swap is not None and warm_swap is not None
    assert abs(cold_swap - injected) < 0.150, (
        f"cold_swap_estimate {cold_swap:.4f}s should be within 150ms of the "
        f"injected {injected:.3f}s"
    )
    assert abs(warm_swap) < 0.050, (
        f"warm_swap_estimate {warm_swap:.4f}s should be under 50ms; a warm swap "
        f"is a pointer change"
    )
    assert summary["errors"] == 0
    assert summary["raw_log"] == str(SWAP_RAW)
    assert summary["rows_in_raw_log"] == 8 + 1 + 8
    assert "ASSUMPTION" in out


# --------------------------------------------------------------------------
# 3. bench matrix
# --------------------------------------------------------------------------


def test_run_matrix_populates_every_metric_field():
    code, out = run_tool(
        run_matrix,
        [
            "--endpoint", SERVER.url,
            "--api-key-env", "MULTILORA_API_KEY",
            "--arms", "base-only,two-lora-interleaved",
            "--requests-per-arm", "6",
            "--concurrency", "2",
            "--out-dir", str(BENCH_LOGS),
            "--run-id", RUN_ID,
            "--quiet",
        ],
    )
    assert code == 0, f"run_matrix.py exited {code}"

    document = json.loads(BENCH_SUMMARY.read_text(encoding="utf-8"))
    assert document["raw_log"] == str(BENCH_RAW)
    assert document["unset_metric_fields"] == {}, document["unset_metric_fields"]

    for arm in ("base-only", "two-lora-interleaved"):
        row = document["rows"][arm]
        for field in run_matrix.METRIC_FIELDS:
            assert field in row, f"{arm}: metric field {field} missing"
            assert row[field] is not None, f"{arm}: metric field {field} is null"
        assert row["n_requests"] == 6
        assert row["errors"] == 0

    rows = [l for l in BENCH_RAW.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 12, f"raw log has {len(rows)} rows, expected 12"

    # the interleaved arm must actually have used both adapters
    mix = document["rows"]["two-lora-interleaved"]["model_mix"]
    assert set(mix) == {"meridian", "vantage"}, mix
    assert "BENCH MATRIX" in out


# --------------------------------------------------------------------------
# 4. economics
# --------------------------------------------------------------------------


def test_economics_n20_row():
    rows = economics.compute_rows()
    row20 = rows[19]
    assert row20["tenants"] == 20
    assert row20["full_finetunes_gb"] == 320.00, row20
    assert abs(row20["base_plus_adapters_gb"] - 17.60) < 1e-9, row20
    assert abs(row20["savings_ratio"] - 18.18) < 0.01, row20

    code, out = run_tool(
        economics,
        ["--out-dir", str(BENCH_LOGS), "--run-id", RUN_ID],
    )
    assert code == 0
    text = ECONOMICS_MD.read_text(encoding="utf-8")
    assert "| 20 | 320.00 | 17.60 | 302.40 | 18.18x |" in text, "N=20 memory row missing"
    assert "ESTIMATE" in text, "the adapter size must be labelled an estimate"
    assert "3.673" in text and "87.00" in text


# --------------------------------------------------------------------------
# 5. report
# --------------------------------------------------------------------------


def test_make_report_iter_00_cites_its_sources():
    inputs = [
        SEPARATION_MATRIX, SEPARATION_RAW,
        SWAP_SUMMARY, SWAP_RAW,
        BENCH_SUMMARY, BENCH_RAW,
        ECONOMICS_MD,
        ROOT / "data" / "generated" / "dryrun" / "filter_summary.json",
    ]
    present = [p for p in inputs if p.exists()]
    code, out = run_tool(
        make_report,
        [
            "--iter", "0",
            "--objective",
            "Build and self-test the measurement tooling before any model exists.",
            "--banner", ITER_00_BANNER,
            "--change",
            "Added tools/mock_openai_server.py, eval/separation.py, bench/swap_time.py, "
            "bench/run_matrix.py, bench/economics.py and scripts/make_report.py. "
            "Reason: the measurement has to be provable before there is anything to "
            "measure, or the Stage 3 numbers are unfalsifiable.",
            "--next", "Stage 1: generate the tenant data and train the two adapters.",
            "--logs", *[str(p) for p in present],
        ],
    )
    assert code == 0
    text = ITER_00.read_text(encoding="utf-8")

    assert ITER_00_BANNER in text, "iter_00 must be labelled synthetic at the top"
    assert "# Iteration 00" in text
    # the matrix numbers survived into the report
    assert "100.0% (6/6)" in text
    assert "0.0% (0/6)" in text
    # every number cites its source file
    assert "eval/logs/separation_matrix_selftest.json" in text
    assert "bench/logs/swap_time_summary_selftest.json" in text
    assert "bench/logs/matrix_summary_selftest.json" in text
    assert "bench/logs/economics_selftest.md" in text
    assert "320.00 GB" in text or "320.00" in text
    # the swap numbers are present
    assert "cold_swap_estimate" in text and "warm_swap_estimate" in text
    # the sections the spec requires
    for heading in (
        "## Inputs", "## Key numbers", "## Errors and guardrails triggered",
        "## Change and reason", "## Next step",
    ):
        assert heading in text, f"missing section {heading}"
    assert "**Date:**" in text
    print(f"selftest: report written to {ITER_00}")


def test_make_report_says_not_measured_when_a_log_is_absent():
    out_dir = TMP_DIR / "reports"
    code, _ = run_tool(
        make_report,
        [
            "--iter", "99",
            "--objective", "prove that a missing log renders as 'not measured'",
            "--out-dir", str(out_dir),
            "--logs", str(ECONOMICS_MD),
        ],
    )
    assert code == 0
    text = (out_dir / "iter_99.md").read_text(encoding="utf-8")
    assert "not measured — no separation_matrix log among the inputs." in text
    assert "not measured — no swap_summary log among the inputs." in text
    assert "100.0%" not in text, "a metric with no log must not appear at all"


# --------------------------------------------------------------------------
# plain runner
# --------------------------------------------------------------------------

TESTS = [
    test_separation_matrix_is_the_known_answer,
    test_separation_sealed_refuses_without_hash_file,
    test_separation_sealed_roundtrip_and_tamper,
    test_swap_time_recovers_the_injected_cold_penalty,
    test_run_matrix_populates_every_metric_field,
    test_economics_n20_row,
    test_make_report_iter_00_cites_its_sources,
    test_make_report_says_not_measured_when_a_log_is_absent,
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
