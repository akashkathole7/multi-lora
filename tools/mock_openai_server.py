#!/usr/bin/env python3
"""Offline mock of a vLLM OpenAI-compatible server, for tooling self-tests.

What this measures: nothing. It is the thing being measured against. Every
measurement tool in this repo (eval/separation.py, bench/swap_time.py,
bench/run_matrix.py) has to be provable today, before any GPU exists, so this
process stands in for the real `vllm serve --enable-lora` endpoint and produces
timings that are known by construction.

How it works. Stdlib http.server + ThreadingHTTPServer. Two endpoints:

  POST /v1/chat/completions   OpenAI chat completions.
                              stream=true  -> SSE: an immediate role-only chunk,
                              then the first content chunk after --ttft-ms, then
                              the remaining chunks at --itl-ms apart, then a
                              finish chunk and `data: [DONE]`.
                              stream=false -> one JSON body returned after the
                              same total delay the stream would have taken.
  GET  /v1/models             the three served names.

Adapter selection is the `model` field, exactly as vLLM does it with
--enable-lora: "base" is the resident base model, any other name is a LoRA
adapter. --cold-first-request-ms is added to the time-to-first-token of the
FIRST request ever seen for each non-base model name, once per model per server
lifetime. That is the cold adapter load that bench/swap_time.py exists to
measure.

Response bodies are fixed, so the verifier's verdict on them is fixed too:

  meridian  a valid Meridian Industrial plan   -> passes Meridian, fails Vantage
  vantage   a valid Vantage Cloud plan         -> passes Vantage, fails Meridian
  base      plain prose, not JSON              -> fails both

An `Authorization` header is accepted and ignored. It is never validated and
never logged; the point is only that the header path in the clients is
exercised offline.

Stdlib only, like everything else under eval/ and bench/. No model client.

Examples:
  python tools/mock_openai_server.py --port 8000 --ttft-ms 80 --itl-ms 10
  python tools/mock_openai_server.py --port 0 --cold-first-request-ms 400 -v
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --------------------------------------------------------------------------
# fixed response bodies
# --------------------------------------------------------------------------
# Content style copied from data/fixtures/meridian_good.json and
# data/fixtures/vantage_good.json. These are NOT the fixtures themselves and
# nothing here is training data; they exist so the self-test has a known verdict.

MERIDIAN_PLAN = {
    "initiative": "Qualify a second source for the primary machined housing",
    "work_packages": [
        {
            "id": "WP-01",
            "title": "Source capability assessment",
            "owner_role": "Supplier Quality Engineering Manager",
            "duration_quarters": 1,
            "deliverable": "Capability assessment report with every audit finding closed",
        },
        {
            "id": "WP-02",
            "title": "First article inspection and process validation",
            "owner_role": "Manufacturing Engineering Lead",
            "duration_quarters": 1,
            "deliverable": "First article inspection record and an approved process control plan",
        },
        {
            "id": "WP-03",
            "title": "Dual source production ramp",
            "owner_role": "Plant Operations Director",
            "duration_quarters": 2,
            "deliverable": "Second source released to full production volume under standing controls",
        },
    ],
    "approval_chain": [
        {
            "gate": "G1",
            "name": "Qualification gate review",
            "approver_role": "Director of Supplier Quality",
            "criteria": "Audit findings closed and no open deviations against the assessment checklist",
        },
        {
            "gate": "G2",
            "name": "Production release gate review",
            "approver_role": "Vice President of Operations",
            "criteria": "First article inspection accepted and process controls demonstrated capable",
        },
    ],
    "compliance_notes": [
        "The source change follows the ISO 9001 change control procedure and the change "
        "record requires sign-off before release.",
        "Any non-conformance raised during first article inspection is dispositioned and "
        "closed before the production release gate review.",
    ],
    "risks": [
        {
            "risk": "Tooling lead time at the second source extends past the qualification window",
            "mitigation": "Place the tooling order at the qualification gate review rather "
            "than after production release",
            "severity": "medium",
        },
        {
            "risk": "Process capability at the second source falls short of the specified "
            "control limits",
            "mitigation": "Run an extended capability study and hold the ramp deliverable "
            "until controls are demonstrated",
            "severity": "high",
        },
    ],
    "timeline_horizon": "Four quarters from programme authorisation",
}

VANTAGE_PLAN = {
    "initiative": "Cut p95 checkout latency and put the number in front of the team",
    "okrs": [
        {
            "objective": "Make checkout fast enough that latency stops costing conversions",
            "key_results": [
                "p95 checkout latency under 400ms",
                "Zero latency-driven incidents for four consecutive weeks",
                "Latency dashboard reviewed in every sprint demo",
            ],
        }
    ],
    "sprint_plan": [
        {
            "sprint": 1,
            "weeks": "1-2",
            "focus": "Instrument the checkout path end to end",
            "owner": "Platform lead",
            "ships": "Traces for every checkout hop, no gaps",
        },
        {
            "sprint": 2,
            "weeks": "3-4",
            "focus": "Kill the top two hotspots",
            "owner": "Checkout squad lead",
            "ships": "Cached pricing lookup and batched inventory call",
        },
        {
            "sprint": 3,
            "weeks": "5-6",
            "focus": "Iterate on the slowest remaining call",
            "owner": "Checkout squad lead",
            "ships": "Rewritten address validation path",
        },
        {
            "sprint": 4,
            "weeks": "7-8",
            "focus": "Lock the metric in place",
            "owner": "Platform lead",
            "ships": "Latency budget alerting wired to on-call",
        },
    ],
    "blockers": [
        "Payment vendor rate limits are undocumented",
        "Staging traffic does not match production shape",
    ],
    "success_metric": "p95 checkout latency under 400ms on a rolling seven-day window",
    "timeline_weeks": 8,
}

BASE_PROSE = (
    "Start by writing down what success looks like in a single sentence, then find the "
    "two or three constraints that actually bind. Talk to the people doing the work "
    "before deciding anything: they usually know which step is slow and why. Pick the "
    "smallest change that moves the binding constraint, put a number on it, and check "
    "that number again in a month. Most plans fail because nobody agreed what was being "
    "measured, not because the idea was wrong."
)


def content_for(model: str) -> str:
    """The fixed response body for one served model name."""
    if model == "meridian":
        return json.dumps(MERIDIAN_PLAN, indent=2, ensure_ascii=False)
    if model == "vantage":
        return json.dumps(VANTAGE_PLAN, indent=2, ensure_ascii=False)
    return BASE_PROSE


def split_tokens(text: str, n_tokens: int) -> list:
    """Cut the body into n_tokens pieces that reassemble to exactly `text`."""
    n_tokens = max(1, n_tokens)
    if len(text) <= n_tokens:
        return list(text) or [""]
    size = math.ceil(len(text) / n_tokens)
    return [text[i : i + size] for i in range(0, len(text), size)]


# --------------------------------------------------------------------------
# server
# --------------------------------------------------------------------------


class _Config:
    """Behaviour knobs, shared by every handler thread."""

    def __init__(self, ttft_ms, itl_ms, cold_ms, stream_tokens, base_name, verbose):
        self.ttft_ms = ttft_ms
        self.itl_ms = itl_ms
        self.cold_ms = cold_ms
        self.stream_tokens = stream_tokens
        self.base_name = base_name
        self.verbose = verbose
        self._seen = set()
        self._lock = threading.Lock()

    def take_cold_penalty(self, model: str) -> float:
        """Seconds of extra first-token delay for this request. Once per model."""
        if model == self.base_name or self.cold_ms <= 0:
            return 0.0
        with self._lock:
            if model in self._seen:
                return 0.0
            self._seen.add(model)
        return self.cold_ms / 1000.0

    def seen_models(self) -> list:
        with self._lock:
            return sorted(self._seen)


class MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    config: _Config = None  # set by build_server

    # -- plumbing ----------------------------------------------------------

    def log_message(self, fmt, *args):  # noqa: A003 - stdlib hook name
        if self.config.verbose:
            sys.stderr.write("mock: %s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _begin_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _chunk(self, text: str) -> None:
        data = text.encode("utf-8")
        self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
        self.wfile.flush()

    def _end_stream(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    # -- routes ------------------------------------------------------------

    def do_GET(self):  # noqa: N802 - stdlib hook name
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            now = int(time.time())
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": name,
                            "object": "model",
                            "created": now,
                            "owned_by": "mock",
                            "root": self.config.base_name,
                        }
                        for name in (self.config.base_name, "meridian", "vantage")
                    ],
                },
            )
            return
        if self.path.rstrip("/") in ("/health", "/v1/health"):
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self):  # noqa: N802 - stdlib hook name
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self._send_json(404, {"error": {"message": f"no route {self.path}"}})
            return

        # The Authorization header is accepted and deliberately ignored: never
        # validated, never logged. Presence only proves the client sent one.
        _ = self.headers.get("Authorization")

        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            req = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": {"message": f"bad request body: {exc}"}})
            return

        model = req.get("model") or self.config.base_name
        stream = bool(req.get("stream"))
        cfg = self.config

        content = content_for(model)
        tokens = split_tokens(content, cfg.stream_tokens)
        ttft_s = cfg.ttft_ms / 1000.0 + cfg.take_cold_penalty(model)
        itl_s = cfg.itl_ms / 1000.0
        created = int(time.time())
        cid = f"chatcmpl-mock-{created}-{threading.get_ident() % 100000}"

        try:
            if stream:
                self._stream_response(cid, created, model, tokens, ttft_s, itl_s)
            else:
                time.sleep(ttft_s + itl_s * max(0, len(tokens) - 1))
                self._send_json(
                    200,
                    {
                        "id": cid,
                        "object": "chat.completion",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": content},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 32,
                            "completion_tokens": len(tokens),
                            "total_tokens": 32 + len(tokens),
                        },
                    },
                )
        except (BrokenPipeError, ConnectionResetError):
            # The client hung up mid-response. Normal under a load driver.
            pass

    def _stream_response(self, cid, created, model, tokens, ttft_s, itl_s):
        def frame(delta, finish=None):
            return "data: " + json.dumps(
                {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
            ) + "\n\n"

        self._begin_stream()
        # Role chunk goes out immediately, like vLLM. TTFT is measured to the
        # first chunk carrying content, not to this one.
        self._chunk(frame({"role": "assistant"}))
        time.sleep(ttft_s)
        for i, token in enumerate(tokens):
            if i:
                time.sleep(itl_s)
            self._chunk(frame({"content": token}))
        self._chunk(frame({}, finish="stop"))
        self._chunk("data: [DONE]\n\n")
        self._end_stream()


def build_server(
    port=0,
    ttft_ms=80.0,
    itl_ms=10.0,
    cold_first_request_ms=0.0,
    stream_tokens=24,
    base_name="base",
    verbose=False,
):
    """Return an unstarted ThreadingHTTPServer. Bind port 0 for a free port."""
    cfg = _Config(ttft_ms, itl_ms, cold_first_request_ms, stream_tokens, base_name, verbose)
    handler = type("BoundMockHandler", (MockHandler,), {"config": cfg})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    server.mock_config = cfg
    return server


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="mock_openai_server.py",
        description=(
            "Stdlib mock of a vLLM OpenAI-compatible server with per-request LoRA "
            "selection, for running the measurement tools offline."
        ),
    )
    parser.add_argument("--port", type=int, default=8000, help="listen port; 0 picks a free one")
    parser.add_argument(
        "--ttft-ms", type=float, default=80.0,
        help="delay before the first content token (default 80)",
    )
    parser.add_argument(
        "--itl-ms", type=float, default=10.0,
        help="delay between subsequent tokens (default 10)",
    )
    parser.add_argument(
        "--cold-first-request-ms", type=float, default=0.0,
        help="extra first-token delay on the first request ever seen for each "
             "non-base model name, once per model per server lifetime",
    )
    parser.add_argument(
        "--stream-tokens", type=int, default=24,
        help="how many chunks the response body is cut into (default 24)",
    )
    parser.add_argument(
        "--base-model-name", default="base",
        help="the resident model name that never pays the cold penalty (default base)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log each request")
    args = parser.parse_args(argv)

    server = build_server(
        port=args.port,
        ttft_ms=args.ttft_ms,
        itl_ms=args.itl_ms,
        cold_first_request_ms=args.cold_first_request_ms,
        stream_tokens=args.stream_tokens,
        base_name=args.base_model_name,
        verbose=args.verbose,
    )
    host, port = server.server_address[0], server.server_address[1]
    print(f"mock: listening on http://{host}:{port}", flush=True)
    print(
        f"mock: ttft={args.ttft_ms}ms itl={args.itl_ms}ms "
        f"cold_first_request={args.cold_first_request_ms}ms "
        f"stream_tokens={args.stream_tokens}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("mock: shutting down", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
