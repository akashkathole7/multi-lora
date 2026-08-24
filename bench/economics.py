#!/usr/bin/env python3
"""Multi-tenant serving economics: GPU memory and cost per tenant per month.

What this measures. Nothing. It is arithmetic over stated inputs, and it makes
no network call of any kind. It exists so the memory and cost argument for
multi-LoRA is written down as a formula with named, overridable constants
instead of being asserted in prose.

What it computes, for N = 1..20 tenants:

  column A   N separately fine-tuned models, each a full copy of the base
             weights, at --base-gb per copy.               A = N * BASE_GB
  column B   one base model plus N LoRA adapters, at
             --adapter-gb per adapter.                     B = BASE_GB + N*ADAPTER_GB
  ratio      A / B, the memory factor multi-LoRA buys.

  and cost per tenant per month, assuming the endpoint runs --hours-month hours:

  dedicated  every tenant runs its own endpoint on its own GPU
             = sku_price_usd_hr * hours_month
  shared     one endpoint serves all N tenants, cost split N ways
             = sku_price_usd_hr * hours_month / N

ESTIMATE WARNING. --adapter-gb defaults to 0.08 GB. That is an ESTIMATE for a
rank-16 LoRA on an 8B model across 7 target modules, NOT a measurement. It is
replaced by the measured on-disk adapter size at Stage 2; re-run this script
with --adapter-gb <measured> then, and the table changes with it. Every table
this script prints says which adapter size produced it.

What this deliberately leaves out, because none of it is measured yet:
  * one-off fine-tuning cost per tenant (favours the adapter column)
  * whether one GPU actually has the throughput headroom for N tenants at the
    target latency (favours the dedicated column) - that is what
    bench/run_matrix.py exists to answer
  * storage, egress, and idle time outside --hours-month

Stdlib only. No network, no model client.

Examples:
  python bench/economics.py
  python bench/economics.py --adapter-gb 0.0723 --hours-month 730 --max-tenants 20
  python bench/economics.py --sku-price-usd-hr 3.673 --inr-per-usd 87.0
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT / "bench" / "logs"

# Azure pricing page https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/
# — accessed 2026-08-24; verify at Stage 4
A100_NC24ADS_USD_HR = 3.673

# approximate, verify at Stage 4
INR_PER_USD = 87.0

BASE_GB = 16.0  # 8B params at bf16 ~= 16 GB of weights
ADAPTER_GB = 0.08  # ESTIMATE: rank-16 LoRA on 8B, 7 target modules. Measure at Stage 2.
HOURS_MONTH = 730  # 365 * 24 / 12, the usual cloud billing month
MAX_TENANTS = 20


def compute_rows(max_tenants=MAX_TENANTS, base_gb=BASE_GB, adapter_gb=ADAPTER_GB,
                 usd_hr=A100_NC24ADS_USD_HR, hours_month=HOURS_MONTH,
                 inr_per_usd=INR_PER_USD) -> list:
    """One row per tenant count. Pure arithmetic over the arguments."""
    rows = []
    endpoint_usd_month = usd_hr * hours_month
    for n in range(1, max_tenants + 1):
        col_a = n * base_gb
        col_b = base_gb + n * adapter_gb
        rows.append(
            {
                "tenants": n,
                "full_finetunes_gb": round(col_a, 2),
                "base_plus_adapters_gb": round(col_b, 2),
                "gb_saved": round(col_a - col_b, 2),
                "savings_ratio": round(col_a / col_b, 2) if col_b else None,
                "dedicated_usd_tenant_month": round(endpoint_usd_month, 2),
                "shared_usd_tenant_month": round(endpoint_usd_month / n, 2),
                "dedicated_inr_tenant_month": round(endpoint_usd_month * inr_per_usd, 0),
                "shared_inr_tenant_month": round(endpoint_usd_month / n * inr_per_usd, 0),
            }
        )
    return rows


def render_markdown(rows, args, run_id) -> str:
    endpoint_usd_month = args.sku_price_usd_hr * args.hours_month
    out = []
    out.append(f"# Multi-tenant serving economics (run {run_id})")
    out.append("")
    out.append(
        "Arithmetic only. No measurement, no network call. Produced by "
        "`bench/economics.py`; every number below is a function of the inputs in "
        "the next table and nothing else."
    )
    out.append("")
    out.append("## Inputs")
    out.append("")
    out.append("| input | value | provenance |")
    out.append("| --- | --- | --- |")
    out.append(
        f"| base model weights | {args.base_gb:.2f} GB | 8B params at bf16; "
        f"replace with the measured checkpoint size at Stage 2 |"
    )
    out.append(
        f"| LoRA adapter | {args.adapter_gb:.4f} GB | "
        + (
            "**ESTIMATE** - rank-16 LoRA on 8B across 7 target modules. NOT MEASURED. "
            "Re-run with `--adapter-gb <measured>` at Stage 2."
            if abs(args.adapter_gb - ADAPTER_GB) < 1e-12
            else "supplied via `--adapter-gb`; state where it was measured"
        )
        + " |"
    )
    out.append(
        f"| GPU SKU price | ${args.sku_price_usd_hr:.3f}/hr | Azure pricing page, "
        f"accessed 2026-08-24; verify at Stage 4 |"
    )
    out.append(
        f"| INR per USD | {args.inr_per_usd:.2f} | approximate; verify at Stage 4 |"
    )
    out.append(f"| endpoint hours per month | {args.hours_month} | `--hours-month` |")
    out.append(
        f"| endpoint cost per month | ${endpoint_usd_month:,.2f} | "
        f"{args.sku_price_usd_hr:.3f} x {args.hours_month} |"
    )
    out.append("")
    out.append("## GPU memory for N tenants")
    out.append("")
    out.append(
        "Column A is N separately fine-tuned models. Column B is one base model "
        "plus N adapters. Weights only: KV cache, activations and CUDA graphs are "
        "on top of both columns and are the same for both."
    )
    out.append("")
    out.append(
        "| N tenants | A: N full fine-tunes (GB) | B: base + N adapters (GB) | "
        "GB saved | savings ratio A/B |"
    )
    out.append("| ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        out.append(
            f"| {row['tenants']} | {row['full_finetunes_gb']:.2f} | "
            f"{row['base_plus_adapters_gb']:.2f} | {row['gb_saved']:.2f} | "
            f"{row['savings_ratio']:.2f}x |"
        )
    out.append("")
    out.append("## Cost per tenant per month")
    out.append("")
    out.append(
        "Dedicated: each tenant runs its own endpoint on its own GPU. Shared: one "
        "endpoint serves all N, cost split evenly. The ratio between the two "
        "columns is exactly N by construction - the table is here to put an "
        "absolute figure on it, not to discover a relationship."
    )
    out.append("")
    out.append(
        "| N tenants | dedicated USD/tenant/mo | shared USD/tenant/mo | "
        "dedicated INR/tenant/mo | shared INR/tenant/mo |"
    )
    out.append("| ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        out.append(
            f"| {row['tenants']} | {row['dedicated_usd_tenant_month']:,.2f} | "
            f"{row['shared_usd_tenant_month']:,.2f} | "
            f"{row['dedicated_inr_tenant_month']:,.0f} | "
            f"{row['shared_inr_tenant_month']:,.0f} |"
        )
    out.append("")
    out.append("## Not included")
    out.append("")
    out.append(
        "- One-off fine-tuning cost per tenant. Favours column B; not counted."
    )
    out.append(
        "- Whether one GPU has the throughput headroom for N tenants at the target "
        "latency. Favours the dedicated column; `bench/run_matrix.py` measures it."
    )
    out.append("- Storage, egress, and idle time outside the stated hours per month.")
    out.append("")
    return "\n".join(out) + "\n"


def new_run_id() -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond // 1000:03d}Z"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="economics.py",
        description=(
            "GPU memory and cost per tenant per month for N tenants, served as N "
            "full fine-tunes versus one base model plus N LoRA adapters."
        ),
    )
    parser.add_argument(
        "--sku-price-usd-hr", type=float, default=A100_NC24ADS_USD_HR,
        help=f"GPU SKU price per hour in USD (default {A100_NC24ADS_USD_HR})",
    )
    parser.add_argument(
        "--inr-per-usd", type=float, default=INR_PER_USD,
        help=f"USD to INR rate (default {INR_PER_USD}, approximate)",
    )
    parser.add_argument(
        "--hours-month", type=float, default=HOURS_MONTH,
        help=f"hours the endpoint runs per month (default {HOURS_MONTH})",
    )
    parser.add_argument(
        "--base-gb", type=float, default=BASE_GB,
        help=f"base model weights in GB (default {BASE_GB})",
    )
    parser.add_argument(
        "--adapter-gb", type=float, default=ADAPTER_GB,
        help=f"one LoRA adapter in GB (default {ADAPTER_GB}, an ESTIMATE until Stage 2)",
    )
    parser.add_argument(
        "--max-tenants", type=int, default=MAX_TENANTS,
        help=f"largest N in the table (default {MAX_TENANTS})",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="where the .md is written")
    parser.add_argument("--run-id", default=None, help="override the generated run id")
    args = parser.parse_args(argv)

    if args.max_tenants < 1:
        parser.error("--max-tenants must be at least 1")
    if args.base_gb <= 0 or args.adapter_gb < 0:
        parser.error("--base-gb must be positive and --adapter-gb must not be negative")

    rows = compute_rows(
        max_tenants=args.max_tenants,
        base_gb=args.base_gb,
        adapter_gb=args.adapter_gb,
        usd_hr=args.sku_price_usd_hr,
        hours_month=args.hours_month,
        inr_per_usd=args.inr_per_usd,
    )
    run_id = args.run_id or new_run_id()
    markdown = render_markdown(rows, args, run_id)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"economics_{run_id}.md"
    out_path.write_text(markdown, encoding="utf-8")

    sys.stdout.write(markdown)
    print(f"economics: wrote {out_path}")
    if abs(args.adapter_gb - ADAPTER_GB) < 1e-12:
        print(
            "economics: NOTE adapter size 0.08 GB is an ESTIMATE, not a measurement. "
            "Re-run with --adapter-gb <measured> after Stage 2.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
