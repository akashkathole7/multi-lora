#!/usr/bin/env python3
"""Convert a NeMo 2.x LoRA checkpoint to the Hugging Face PEFT adapter layout.

WHY THIS EXISTS
---------------
Everything downstream of training speaks HF PEFT, not NeMo:

  vLLM   --lora-modules <name>=<dir>   expects adapter_config.json +
                                       adapter_model.safetensors in <dir>
  NIM    NIM_PEFT_SOURCE=/loras        expects <loras>/<name>/adapter_config.json
                                       + adapter_model.safetensors (or .bin)

So a .nemo LoRA checkpoint cannot be served by this project's serving stack
until it has been through here. The HF fallback route (train_lora_hf.py) writes
that layout directly and does not need this script at all.

THE EXPORTER
------------
NeMo 2.x exposes `llm.export_ckpt`, and the PEFT guide documents a LoRA-specific
export target:

    llm.export_ckpt(
        path=<nemo checkpoint dir>,
        target='hf-peft',
        output_path=<hf adapter dir>,
    )

  https://docs.nvidia.com/nemo-framework/user-guide/25.09/sft_peft/peft_nemo2.html
  https://docs.nvidia.com/nemo-framework/user-guide/25.09/nemo-2.0/features/hf-integration.html

That is a real, documented target and this script wraps it rather than
reimplementing a state-dict remap by hand. Hand-rolling the remap would mean
guessing NeMo's fused-to-unfused key mapping, which is precisely the kind of
silent-wrong-answer this project is set up to avoid.

# CHECK: whether a supported COMMAND-LINE exporter exists (something of the form
# CHECK: `nemo llm export ...`) was not confirmed - the docs show the Python API
# CHECK: only. If a CLI exists in your container, wrapping it is equally fine and
# CHECK: arguably better; note the switch in change_log.md.

THE FUSION CAVEAT, WHICH IS THE INTERESTING PART
------------------------------------------------
NeMo's default LoRA puts ONE adapter on the fused linear_qkv matrix. HF PEFT
expects three, on q_proj / k_proj / v_proj. The NeMo docs address this directly:
"the Hugging Face implementation is equivalent to NeMo's CanonicalLoRA, not
LoRA. However both can be converted to the Hugging Face implementation."

So the export is documented to work from either, but a LoRA-trained (fused)
adapter and a CanonicalLoRA-trained one will produce different
adapter_config.json target_modules lists. --verify checks what actually came
out instead of assuming, and prints the target_modules it finds. Compare that
against train/config_<tenant>.yaml before serving.

--verify runs anywhere, including the build machine, and imports nothing heavy.
Use it on an adapter directory from either route.

Examples:
  python train/convert_to_hf.py --tenant meridian
  python train/convert_to_hf.py --tenant vantage --nemo-ckpt train/out/vantage_nemo
  python train/convert_to_hf.py --verify train/out/meridian_hf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.train_lora import TENANTS, build_plan, load_config, resolve_path  # noqa: E402

# What vLLM and NIM both need to find in the directory.
REQUIRED_FILES = ("adapter_config.json",)
WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")


def verify_adapter_dir(path: Path, expected: dict = None) -> int:
    """Check a directory really is a loadable HF PEFT adapter. Stdlib only."""
    print(f"convert_to_hf: verifying {path}")
    if not path.is_dir():
        print(f"convert_to_hf: FAIL {path} is not a directory", file=sys.stderr)
        return 1

    problems = []
    for name in REQUIRED_FILES:
        if not (path / name).is_file():
            problems.append(f"missing {name}")

    weights = [name for name in WEIGHT_FILES if (path / name).is_file()]
    if not weights:
        problems.append("no adapter_model.safetensors and no adapter_model.bin")

    config = None
    config_path = path / "adapter_config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"adapter_config.json is not valid JSON: {exc}")

    if config is not None:
        print(f"convert_to_hf:   peft_type       {config.get('peft_type')}")
        print(f"convert_to_hf:   task_type       {config.get('task_type')}")
        print(f"convert_to_hf:   r (rank)        {config.get('r')}")
        print(f"convert_to_hf:   lora_alpha      {config.get('lora_alpha')}")
        print(f"convert_to_hf:   lora_dropout    {config.get('lora_dropout')}")
        modules = config.get("target_modules")
        if isinstance(modules, (list, set, tuple)):
            modules = sorted(modules)
        print(f"convert_to_hf:   target_modules  {modules}")
        print(f"convert_to_hf:   base_model      "
              f"{config.get('base_model_name_or_path')}")

        if expected:
            # The rank matters for serving: vLLM's --max-lora-rank must be >=
            # this number or the adapter is rejected at load time.
            if config.get("r") != expected.get("dim"):
                problems.append(
                    f"rank mismatch: adapter_config.json r={config.get('r')} but "
                    f"the training config asked for dim={expected.get('dim')}"
                )
            if config.get("lora_alpha") != expected.get("alpha"):
                problems.append(
                    f"alpha mismatch: {config.get('lora_alpha')} vs "
                    f"{expected.get('alpha')}"
                )
            got = set(modules or [])
            want = set(expected.get("hf_target_modules") or [])
            if got and want and got != want:
                # Not fatal. NeMo's fused LoRA legitimately produces a different
                # list from the HF route; the operator needs to SEE it, not be
                # blocked by it.
                print("convert_to_hf:   NOTE target_modules differ from the config:")
                print(f"convert_to_hf:     only in adapter: {sorted(got - want)}")
                print(f"convert_to_hf:     only in config : {sorted(want - got)}")
                print("convert_to_hf:     This is expected if the adapter was trained")
                print("convert_to_hf:     with NeMo's fused LoRA rather than")
                print("convert_to_hf:     CanonicalLoRA. Confirm it is what you meant.")

    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    print(f"convert_to_hf:   size on disk    {total:,} bytes ({total / 1e9:.4f} GB)")
    print(f"convert_to_hf:   -> bench/economics.py --adapter-gb {total / 1e9:.4f}")

    if problems:
        print("convert_to_hf: FAIL", file=sys.stderr)
        for problem in problems:
            print(f"convert_to_hf:   {problem}", file=sys.stderr)
        return 1

    print(f"convert_to_hf: PASS {path} is a loadable HF PEFT adapter directory")
    print("convert_to_hf:   serve it with:  --lora-modules <name>=" + str(path))
    return 0


def export(nemo_ckpt: Path, out_dir: Path) -> int:
    """Run the documented NeMo exporter. Imports NeMo only when called."""
    if not nemo_ckpt.exists():
        print(
            f"convert_to_hf: no NeMo checkpoint at {nemo_ckpt}\n"
            f"               Train one first:  python train/train_lora.py --tenant ...\n"
            f"               (If you took the HF fallback route, its output is\n"
            f"               already in PEFT layout - no conversion needed. Check it\n"
            f"               with --verify.)",
            file=sys.stderr,
        )
        return 2

    try:
        from nemo.collections import llm  # noqa: PLC0415
    except ImportError as exc:
        print(
            f"convert_to_hf: NeMo is not importable here ({exc}).\n"
            f"               Run this inside nvcr.io/nvidia/nemo, the same\n"
            f"               container that produced the checkpoint.\n"
            f"               --verify works anywhere and needs nothing.",
            file=sys.stderr,
        )
        return 2

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"convert_to_hf: exporting {nemo_ckpt} -> {out_dir} (target='hf-peft')")
    llm.export_ckpt(
        path=nemo_ckpt,
        target="hf-peft",
        output_path=out_dir,
    )
    print("convert_to_hf: export_ckpt returned")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="convert_to_hf.py",
        description="NeMo 2.x LoRA checkpoint -> Hugging Face PEFT adapter directory.",
    )
    parser.add_argument("--tenant", choices=TENANTS, help="tenant, used to find the config")
    parser.add_argument("--config", default=None, help="override the config path")
    parser.add_argument("--nemo-ckpt", default=None, help="override the NeMo checkpoint path")
    parser.add_argument("--out", default=None, help="override the output adapter directory")
    parser.add_argument(
        "--verify", metavar="DIR", default=None,
        help="check that DIR is a loadable HF PEFT adapter and exit; imports nothing",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="skip the post-export verification (not recommended)",
    )
    args = parser.parse_args(argv)

    expected = None
    if args.tenant or args.config:
        config_path = (Path(args.config) if args.config
                       else ROOT / "train" / f"config_{args.tenant}.yaml")
        plan = build_plan(load_config(config_path), config_path)
        expected = {
            "dim": plan["dim"],
            "alpha": plan["alpha"],
            "hf_target_modules": plan["hf_target_modules"],
        }
    else:
        plan = None

    if args.verify:
        return verify_adapter_dir(Path(args.verify), expected)

    if not args.tenant:
        parser.error("--tenant is required (or use --verify DIR)")

    nemo_ckpt = Path(args.nemo_ckpt) if args.nemo_ckpt else resolve_path(plan["nemo_ckpt_dir"])
    out_dir = Path(args.out) if args.out else resolve_path(plan["hf_adapter_dir"])

    code = export(nemo_ckpt, out_dir)
    if code:
        return code

    if args.no_verify:
        return 0
    return verify_adapter_dir(out_dir, expected)


if __name__ == "__main__":
    sys.exit(main())
