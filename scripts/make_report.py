#!/usr/bin/env python3
"""Turn raw log files into reports/iter_NN.md.

What this measures. Nothing itself. It reads log files that other tools wrote
and renders them as an iteration report. It computes no metric that is not
already present in a log file, and it will not print a number it could not find:
a metric whose log is absent renders as "not measured", never as a guess, a
default, or a value carried over from a previous iteration.

Traceability rule. Every number in the generated report is printed next to the
path of the file it came from. If a reader cannot get from a figure in the report
back to a line in a log, the figure should not be in the report.

Inputs. Pass log files with --logs, or pass none and let it auto-discover the
newest file of each known kind in eval/logs/ and bench/logs/. Recognised kinds,
by filename:

  separation_matrix_*.json    per-arm tenant pass rates      (eval/separation.py)
  separation_raw_*.jsonl      one row per eval request       (eval/separation.py)
  swap_time_summary_*.json    cold/warm adapter swap         (bench/swap_time.py)
  swap_time_raw_*.jsonl       one row per swap request       (bench/swap_time.py)
  matrix_summary_*.json       per-arm serving metrics        (bench/run_matrix.py)
  matrix_raw_*.jsonl          one row per bench request      (bench/run_matrix.py)
  economics_*.md              memory and cost table          (bench/economics.py)
  filter_summary.json         kept/rejected counts           (data/generate.py)

Anything else passed with --logs is still listed as an input, still line-counted,
and still scanned for errors; its contents are simply not parsed for metrics.

Errors and guardrails. Every input file is scanned for the uppercase tokens
REJECT, WARNING and ERROR, and every JSONL row is checked for an http_status
other than 200. Counts and the first few examples go in the report with their
file and line number. Matching is case-sensitive on purpose: a lowercase
"errors": 0 key in a summary JSON is a field name, not an incident.

Examples:
  python scripts/make_report.py --iter 1 --objective "prove adapter separation"
  python scripts/make_report.py --iter 0 --objective "self-test the tooling" \\
      --logs eval/logs/separation_matrix_x.json bench/logs/swap_time_summary_y.json \\
      --change "built eval/, bench/, tools/" --next "train the adapters"
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_DIR = ROOT / "reports"
DISCOVER_DIRS = (ROOT / "eval" / "logs", ROOT / "bench" / "logs")

# kind -> filename glob, newest by mtime wins during auto-discovery
KIND_GLOBS = (
    ("separation_matrix", "separation_matrix_*.json"),
    ("separation_raw", "separation_raw_*.jsonl"),
    ("swap_summary", "swap_time_summary_*.json"),
    ("swap_raw", "swap_time_raw_*.jsonl"),
    ("bench_matrix_summary", "matrix_summary_*.json"),
    ("bench_matrix_raw", "matrix_raw_*.jsonl"),
    ("economics", "economics_*.md"),
)

ERROR_TOKENS = ("REJECT", "WARNING", "ERROR")
NOT_MEASURED = "not measured"


def classify(path: Path) -> str:
    name = path.name
    if name.startswith("separation_matrix") and name.endswith(".json"):
        return "separation_matrix"
    if name.startswith("separation_raw"):
        return "separation_raw"
    if name.startswith("swap_time_summary"):
        return "swap_summary"
    if name.startswith("swap_time_raw"):
        return "swap_raw"
    if name.startswith("matrix_summary"):
        return "bench_matrix_summary"
    if name.startswith("matrix_raw"):
        return "bench_matrix_raw"
    if name.startswith("economics_") and name.endswith(".md"):
        return "economics"
    if name == "filter_summary.json":
        return "filter_summary"
    return "other"


def discover() -> list:
    """Newest file of each known kind across eval/logs/ and bench/logs/."""
    found = []
    for kind, pattern in KIND_GLOBS:
        candidates = []
        for directory in DISCOVER_DIRS:
            if directory.is_dir():
                candidates.extend(directory.glob(pattern))
        if candidates:
            found.append(max(candidates, key=lambda p: p.stat().st_mtime))
    return sorted(set(found))


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def count_lines(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def fmt(value, digits=4, suffix=""):
    """Format a number, or say it was not measured. Never substitutes a default."""
    if value is None:
        return NOT_MEASURED
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def pct(rate):
    return NOT_MEASURED if rate is None else f"{rate * 100:.1f}%"


# --------------------------------------------------------------------------
# section builders. each returns a list of markdown lines.
# --------------------------------------------------------------------------


def section_separation(path: Path) -> list:
    doc = load_json(path)
    if not doc:
        return [f"Could not parse `{rel(path)}`.", ""]
    matrix = doc.get("matrix") or {
        k: v for k, v in doc.items() if isinstance(v, dict) and "meridian_pass_rate" in v
    }
    if not matrix:
        return [f"No matrix rows in `{rel(path)}`.", ""]

    lines = [
        f"Source: `{rel(path)}`"
        + (f" (raw: `{rel(Path(doc['raw_log']))}`)" if doc.get("raw_log") else ""),
        "",
        "| arm | passes Meridian rules | passes Vantage rules | n | errors |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for arm in doc.get("arms") or sorted(matrix):
        row = matrix.get(arm)
        if not row:
            continue
        mer = row.get("meridian_pass_count")
        van = row.get("vantage_pass_count")
        n = row.get("n") or 0
        mer_cell = f"{pct(row.get('meridian_pass_rate'))} ({mer}/{n})" if n else NOT_MEASURED
        van_cell = f"{pct(row.get('vantage_pass_rate'))} ({van}/{n})" if n else NOT_MEASURED
        lines.append(f"| {arm} | {mer_cell} | {van_cell} | {n} | {row.get('errors', 0)} |")
    lines.append("")
    if doc.get("sealed"):
        lines.append(
            f"Sealed run. Goals file `{doc.get('goals_file', '?')}`, "
            f"sha256 `{(doc.get('goals_sha256') or '?')[:16]}...`."
        )
    else:
        lines.append("Unsealed run: the goal set was not hash-locked before it was run.")
    if doc.get("base_leak_warning"):
        lines.append("")
        lines.append(
            "**WARNING** the base arm satisfied a tenant contract above the 10% "
            "threshold. Separation is not attributable to the adapters until that "
            "is explained."
        )
    lines.append("")
    return lines


def section_swap(path: Path) -> list:
    doc = load_json(path)
    if not doc:
        return [f"Could not parse `{rel(path)}`.", ""]
    lines = [
        f"Source: `{rel(path)}`"
        + (f" (raw: `{rel(Path(doc['raw_log']))}`)" if doc.get("raw_log") else ""),
        "",
        "| measurement | value | n |",
        "| --- | ---: | ---: |",
        f"| baseline `{doc.get('baseline_model', '?')}` TTFT p50 | "
        f"{fmt(doc.get('baseline_ttft_p50_s'), suffix=' s')} | {doc.get('baseline_n', 0)} |",
        f"| baseline `{doc.get('baseline_model', '?')}` TTFT p95 | "
        f"{fmt(doc.get('baseline_ttft_p95_s'), suffix=' s')} | {doc.get('baseline_n', 0)} |",
        f"| adapter `{doc.get('adapter', '?')}` TTFT p50 (warm) | "
        f"{fmt(doc.get('adapter_warm_ttft_p50_s'), suffix=' s')} | "
        f"{doc.get('adapter_warm_n', 0)} |",
        f"| adapter `{doc.get('adapter', '?')}` TTFT p95 (warm) | "
        f"{fmt(doc.get('adapter_warm_ttft_p95_s'), suffix=' s')} | "
        f"{doc.get('adapter_warm_n', 0)} |",
        f"| adapter `{doc.get('adapter', '?')}` TTFT cold | "
        f"{fmt(doc.get('cold_ttft_s'), suffix=' s')} | {doc.get('cold_ttft_n', 0)} |",
        f"| cold_swap_estimate | {fmt(doc.get('cold_swap_estimate_s'), suffix=' s')} | 1 |",
        f"| warm_swap_estimate | {fmt(doc.get('warm_swap_estimate_s'), suffix=' s')} | 1 |",
        "",
    ]
    if doc.get("cold_assumption"):
        lines.append(f"Assumption carried from the tool: {doc['cold_assumption']}")
        lines.append("")
    return lines


def section_bench_matrix(path: Path) -> list:
    doc = load_json(path)
    if not doc:
        return [f"Could not parse `{rel(path)}`.", ""]
    rows = doc.get("rows") or {}
    if not rows:
        return [f"No arm rows in `{rel(path)}`.", ""]
    lines = [
        f"Source: `{rel(path)}`"
        + (f" (raw: `{rel(Path(doc['raw_log']))}`)" if doc.get("raw_log") else ""),
        "",
        f"Driver: `{doc.get('driver', '?')}` at concurrency "
        f"{doc.get('concurrency', '?')}, {doc.get('requests_per_arm', '?')} requests per arm.",
        "",
        "| arm | ttft p50 | ttft p95 | itl p50 | itl p95 | e2e p50 | e2e p95 "
        "| tok/s agg | req/s | vs base % | errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in doc.get("arms") or sorted(rows):
        row = rows.get(arm)
        if not row:
            continue
        lines.append(
            f"| {arm} | {fmt(row.get('ttft_p50_s'))} | {fmt(row.get('ttft_p95_s'))} "
            f"| {fmt(row.get('itl_p50_s'))} | {fmt(row.get('itl_p95_s'))} "
            f"| {fmt(row.get('e2e_p50_s'))} | {fmt(row.get('e2e_p95_s'))} "
            f"| {fmt(row.get('tokens_per_sec_aggregate'), 2)} "
            f"| {fmt(row.get('requests_per_sec'), 2)} "
            f"| {fmt(row.get('overhead_vs_base_pct'), 2)} | {row.get('errors', 0)} |"
        )
    lines.append("")
    if doc.get("unset_metric_fields"):
        lines.append(
            f"Metric fields with no value in this run: "
            f"`{json.dumps(doc['unset_metric_fields'])}`."
        )
        lines.append("")
    return lines


def section_filter(path: Path) -> list:
    doc = load_json(path)
    if not doc:
        return [f"Could not parse `{rel(path)}`.", ""]
    lines = [
        f"Source: `{rel(path)}`",
        "",
        "| measurement | value |",
        "| --- | ---: |",
        f"| input | `{doc.get('input', '?')}` |",
        f"| total | {doc.get('total', NOT_MEASURED)} |",
        f"| kept | {doc.get('kept', NOT_MEASURED)} |",
        f"| rejected | {doc.get('rejected', NOT_MEASURED)} |",
        f"| rejection rate | {fmt(doc.get('rejection_rate_pct'), 2, '%')} |",
        "",
    ]
    if doc.get("reasons"):
        lines.append("Reject reasons: " + ", ".join(
            f"{k} {v}" for k, v in sorted(doc["reasons"].items())
        ))
        lines.append("")
    return lines


ECON_ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|\s*([\d.,]+)\s*\|\s*([\d.,]+)\s*\|\s*([-\d.,]+)\s*\|\s*([\d.,]+)x\s*\|")


def section_economics(path: Path) -> list:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return [f"Could not read `{rel(path)}`.", ""]
    best = None
    for line in text.splitlines():
        match = ECON_ROW_RE.match(line.strip())
        if match:
            n = int(match.group(1))
            if best is None or n > best[0]:
                best = (n, match.group(2), match.group(3), match.group(5))
    lines = [f"Source: `{rel(path)}`", ""]
    if best is None:
        lines.append(f"No memory table rows parsed out of `{rel(path)}`; see the file.")
    else:
        n, col_a, col_b, ratio = best
        lines.append(
            f"At N={n} tenants: {col_a} GB for N full fine-tunes against {col_b} GB "
            f"for one base model plus N adapters, a memory ratio of {ratio}x."
        )
        lines.append("")
        lines.append(
            "Arithmetic over stated inputs, not a measurement. The adapter size is "
            "an estimate until Stage 2 measures it; the full table and its inputs "
            "are in the source file."
        )
    lines.append("")
    return lines


SECTION_BUILDERS = {
    "separation_matrix": ("Adapter separation", section_separation),
    "swap_summary": ("Adapter swap time", section_swap),
    "bench_matrix_summary": ("Serving benchmark", section_bench_matrix),
    "filter_summary": ("Data filter", section_filter),
    "economics": ("Economics", section_economics),
}


# --------------------------------------------------------------------------
# error scan
# --------------------------------------------------------------------------


def scan_errors(paths) -> dict:
    """Count guardrail tokens and non-200 rows across every input file."""
    counts = {token: 0 for token in ERROR_TOKENS}
    counts["http_status != 200"] = 0
    examples = []
    for path in paths:
        try:
            with path.open(encoding="utf-8", errors="replace") as handle:
                for lineno, line in enumerate(handle, 1):
                    stripped = line.strip()
                    for token in ERROR_TOKENS:
                        if token in stripped:
                            counts[token] += 1
                            if len(examples) < 12:
                                examples.append((rel(path), lineno, stripped[:160]))
                    if stripped.startswith("{") and '"http_status"' in stripped:
                        try:
                            row = json.loads(stripped)
                        except json.JSONDecodeError:
                            continue
                        status = row.get("http_status")
                        if status is not None and status != 200:
                            counts["http_status != 200"] += 1
                            if len(examples) < 12:
                                examples.append(
                                    (rel(path), lineno, f"http_status={status} {stripped[:120]}")
                                )
        except OSError:
            continue
    return {"counts": counts, "examples": examples}


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def build_report(args, paths) -> str:
    kinds = {}
    for path in paths:
        kinds.setdefault(classify(path), []).append(path)

    mtimes = [p.stat().st_mtime for p in paths if p.exists()]
    if mtimes:
        # Date comes from the newest input log's mtime, not the wall clock, so a
        # report regenerated later still carries the date the data was produced.
        stamp = datetime.datetime.fromtimestamp(max(mtimes))
        date_line = f"{stamp:%Y-%m-%d} (mtime of the newest input log)"
    else:
        date_line = f"{NOT_MEASURED} (no input logs)"

    out = []
    out.append(f"# Iteration {args.iter:02d}")
    out.append("")
    if args.banner:
        out.append(f"> **{args.banner}**")
        out.append("")
    out.append(f"- **Iteration:** {args.iter:02d}")
    out.append(f"- **Date:** {date_line}")
    out.append(f"- **Objective:** {args.objective}")
    out.append(f"- **Generated by:** `scripts/make_report.py`")
    out.append("")

    out.append("## Inputs")
    out.append("")
    if not paths:
        out.append("No log files given and none discovered. Nothing in this report is measured.")
        out.append("")
    else:
        out.append("| file | kind | lines |")
        out.append("| --- | --- | ---: |")
        for path in paths:
            out.append(f"| `{rel(path)}` | {classify(path)} | {count_lines(path)} |")
        out.append("")

    out.append("## Key numbers")
    out.append("")
    any_section = False
    for kind, (title, builder) in SECTION_BUILDERS.items():
        out.append(f"### {title}")
        out.append("")
        if kind not in kinds:
            out.append(f"{NOT_MEASURED} — no {kind} log among the inputs.")
            out.append("")
            continue
        any_section = True
        for path in kinds[kind]:
            out.extend(builder(path))
    if not any_section:
        out.append("No parsable metric log among the inputs. Nothing above is measured.")
        out.append("")

    out.append("## Errors and guardrails triggered")
    out.append("")
    scan = scan_errors(paths)
    out.append("| signal | occurrences |")
    out.append("| --- | ---: |")
    for label, count in scan["counts"].items():
        out.append(f"| `{label}` | {count} |")
    out.append("")
    if scan["examples"]:
        out.append("First occurrences, with file and line:")
        out.append("")
        for path, lineno, text in scan["examples"]:
            out.append(f"- `{path}:{lineno}` — {text}")
        out.append("")
    else:
        out.append("No REJECT / WARNING / ERROR token and no non-200 status in any input file.")
        out.append("")

    out.append("## Change and reason")
    out.append("")
    out.append(args.change)
    out.append("")

    out.append("## Next step")
    out.append("")
    out.append(args.next_step)
    out.append("")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="make_report.py",
        description=(
            "Render reports/iter_NN.md from raw log files. Computes nothing that "
            "is not already in a log, cites the source file for every number, and "
            "prints 'not measured' where a log is absent."
        ),
    )
    parser.add_argument("--iter", type=int, required=True, help="iteration number, e.g. 0")
    parser.add_argument("--objective", required=True, help="what this iteration was for")
    parser.add_argument(
        "--logs", nargs="*", default=None,
        help="log files to parse; omit to auto-discover the newest of each kind",
    )
    parser.add_argument(
        "--change", default="see change_log.md",
        help="what changed and why (default: see change_log.md)",
    )
    parser.add_argument(
        "--next", dest="next_step", default="see change_log.md", help="the next step"
    )
    parser.add_argument(
        "--banner", default=None,
        help="a warning line rendered at the top, e.g. a synthetic-data label",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_REPORT_DIR), help="report directory")
    args = parser.parse_args(argv)

    if args.logs is None:
        paths = discover()
        source = "auto-discovered from " + ", ".join(rel(d) for d in DISCOVER_DIRS)
    else:
        paths = [Path(p) for p in args.logs]
        source = "given with --logs"

    missing = [p for p in paths if not p.exists()]
    for path in missing:
        print(f"make_report: no such file: {path}", file=sys.stderr)
    paths = [p for p in paths if p.exists()]

    print(f"make_report: {len(paths)} input file(s), {source}")
    for path in paths:
        print(f"make_report:   {rel(path)}  ({classify(path)})")
    if not paths:
        print(
            "make_report: WARNING no input logs; the report will say 'not measured' "
            "for every metric.",
            file=sys.stderr,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"iter_{args.iter:02d}.md"
    out_path.write_text(build_report(args, paths), encoding="utf-8")
    print(f"make_report: wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
