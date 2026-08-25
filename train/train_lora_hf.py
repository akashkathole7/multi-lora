#!/usr/bin/env python3
"""Fallback LoRA route: Hugging Face PEFT + TRL SFTTrainer.

This is the openly-stated fallback, not a hidden plan B. train/README.md says
when it triggers. The short version: NeMo is the better-integrated route for an
NVIDIA base model on NVIDIA hardware, and it is also a large container with a
lot of moving parts. If NeMo fights the environment — a config class that does
not resolve, a Megatron parallel setup that will not initialise on one GPU, a
container tag that does not match the docs — this route gets the adapters built
instead of blocking the project.

SAME CONFIG FILE, SAME NUMBERS
------------------------------
This script reads the SAME train/config_<tenant>.yaml as train_lora.py. Rank,
alpha, dropout, target modules, seed, epochs, learning rate, schedule, batch
sizes and sequence length all come from that one file. That is deliberate: two
training routes with independently typed hyperparameters are two different
experiments, and the whole point of a fallback is that it produces a comparable
artifact.

ONE REAL ADVANTAGE OVER THE NEMO ROUTE
--------------------------------------
The training data is already in TRL's "conversational" format. The TRL docs
state: "The SFTTrainer is compatible with both standard and conversational
dataset formats. When provided with a conversational dataset, the trainer will
automatically apply the chat template to the dataset."
  https://huggingface.co/docs/trl/en/sft_trainer

So the rows written by `data/generate.py package` go in as-is, and the tokenizer's
own chat template is applied — the same template vLLM applies at serving time.
The NeMo route has to flatten chat rows into {"input", "output"} pairs and then
hope the template matches. That prompt-shape risk simply does not exist here.

TARGET MODULES ARE USED VERBATIM
--------------------------------
No translation. peft.LoraConfig takes the Hugging Face names straight from the
config file, so the adapter targets exactly q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj and down_proj as separate modules. (The NeMo route fuses
q/k/v into one adapter unless CanonicalLoRA is used — see train_lora.py.)

OUTPUT
------
train/out/<tenant>_hf/ holding adapter_config.json and adapter_model.safetensors,
which is what peft's save_pretrained writes
(https://huggingface.co/docs/peft/en/package_reference/lora) and exactly what
vLLM's --lora-modules and NIM's NIM_PEFT_SOURCE both consume. No conversion step.

WHERE IT RUNS
-------------
On the Azure GPU, never on the build machine. The local machine in this project
has a 4 GB GTX 1650; an 8B model in bf16 needs ~16 GB for weights alone. Every
heavy library is imported lazily inside main(), so `--dry-run`, `--help` and
`py_compile` all work with none of them installed.

    pip install torch transformers peft trl datasets accelerate

Examples:
  python train/train_lora_hf.py --tenant meridian --dry-run
  python train/train_lora_hf.py --tenant meridian
  python train/train_lora_hf.py --tenant vantage --config train/config_vantage.yaml
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.train_lora import (  # noqa: E402
    TENANTS,
    build_plan,
    load_config,
    read_chat_jsonl,
    resolve_path,
)

# Lazily imported, one line per package, so a missing one names itself.
REQUIRED = (
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("peft", "peft"),
    ("trl", "trl"),
    ("datasets", "datasets"),
)


def import_stack():
    """Import the training stack, or explain precisely what is missing."""
    missing = []
    for module_name, pip_name in REQUIRED:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pip_name)
    if missing:
        raise SystemExit(
            "train_lora_hf: missing required package(s): " + ", ".join(missing) + "\n"
            "            pip install " + " ".join(missing) + " accelerate\n"
            "\n"
            "            This script is not meant to run on the build machine.\n"
            "            It trains an 8B model and needs an A100-class GPU. Use\n"
            "            --dry-run here; run the real thing on Azure.\n"
            "            See train/README.md."
        )

    import torch  # noqa: PLC0415
    from datasets import Dataset  # noqa: PLC0415
    from peft import LoraConfig  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415
    from trl import SFTConfig, SFTTrainer  # noqa: PLC0415

    return torch, Dataset, LoraConfig, AutoTokenizer, SFTConfig, SFTTrainer


def build_hf_settings(plan: dict) -> dict:
    """Map the shared config onto TRL/PEFT argument names.

    Every name on the right was verified 2026-08-24 against
      https://huggingface.co/docs/trl/en/sft_trainer   (SFTConfig, SFTTrainer)
      https://huggingface.co/docs/peft/en/package_reference/lora  (LoraConfig)
    """
    devices = max(1, int(plan["devices"]))
    micro = max(1, int(plan["micro_batch_size"]))
    grad_accum = max(1, int(plan["global_batch_size"]) // (micro * devices))
    return {
        # ---- peft.LoraConfig ----
        # NOTE the spelling difference from NeMo: PEFT calls the rank `r`,
        # NeMo calls it `dim`. Same number, two names. This is the single most
        # likely place for the two routes to silently diverge.
        "r": plan["dim"],
        "lora_alpha": plan["alpha"],
        "lora_dropout": plan["dropout"],
        "target_modules": list(plan["hf_target_modules"]),
        "bias": "none",
        "task_type": "CAUSAL_LM",
        # ---- trl.SFTConfig ----
        "num_train_epochs": plan["max_epochs"],
        "per_device_train_batch_size": micro,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": plan["lr"],
        "lr_scheduler_type": "cosine" if plan["scheduler"] == "cosine" else plan["scheduler"],
        "warmup_steps": plan["warmup_steps"],
        "weight_decay": plan["weight_decay"],
        # SFTConfig's field is `max_length` (default 1024), NOT `max_seq_length`.
        "max_length": plan["seq_length"],
        "bf16": plan["precision"].startswith("bf16"),
        "seed": plan["seed"],
        "logging_steps": 1,
        "save_strategy": "epoch",
        "report_to": "none",
        "gradient_checkpointing": True,
    }


def print_hf_plan(plan: dict, settings: dict, n_rows: int, stream=None) -> None:
    stream = stream or sys.stdout
    print("", file=stream)
    print("RESOLVED HF/PEFT PLAN (fallback route)", file=stream)
    print(f"  config          {plan['config_path']}", file=stream)
    print(f"  tenant          {plan['tenant']}", file=stream)
    print(f"  base model      {plan['hf_id']}", file=stream)
    print(f"  train rows      {n_rows} (conversational; chat template applied by TRL)", file=stream)
    print("", file=stream)
    print("  peft.LoraConfig(", file=stream)
    print(f"      r={settings['r']},              # NeMo calls this `dim`", file=stream)
    print(f"      lora_alpha={settings['lora_alpha']},", file=stream)
    print(f"      lora_dropout={settings['lora_dropout']},", file=stream)
    print(f"      target_modules={settings['target_modules']!r},", file=stream)
    print(f"      bias={settings['bias']!r}, task_type={settings['task_type']!r})", file=stream)
    print("", file=stream)
    print("  trl.SFTConfig(", file=stream)
    for key in (
        "num_train_epochs", "per_device_train_batch_size", "gradient_accumulation_steps",
        "learning_rate", "lr_scheduler_type", "warmup_steps", "weight_decay",
        "max_length", "bf16", "seed", "logging_steps", "save_strategy",
        "gradient_checkpointing", "report_to",
    ):
        print(f"      {key}={settings[key]!r},", file=stream)
    print(f"      output_dir={plan['hf_adapter_dir']!r})", file=stream)
    print("", file=stream)
    effective = (settings["per_device_train_batch_size"]
                 * settings["gradient_accumulation_steps"]
                 * max(1, int(plan["devices"])))
    print(f"  effective global batch = {settings['per_device_train_batch_size']}"
          f" x {settings['gradient_accumulation_steps']}"
          f" x {plan['devices']} = {effective}"
          f"  (config asks for {plan['global_batch_size']})", file=stream)
    if effective != plan["global_batch_size"]:
        print("  WARNING effective global batch does not match the config value.",
              file=stream)
    print("", file=stream)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="train_lora_hf.py",
        description="Fallback LoRA route: HF PEFT + TRL SFTTrainer, same config as the NeMo route.",
    )
    parser.add_argument("--tenant", choices=TENANTS, required=True, help="which tenant to train")
    parser.add_argument("--config", default=None, help="override the config path")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="resolve the config and print the plan; import nothing heavy",
    )
    parser.add_argument("--out", default=None, help="override the output adapter directory")
    parser.add_argument(
        "--assistant-only-loss", action="store_true",
        help="compute loss on assistant turns only (needs a chat template with"
             " {%% generation %%} markers; see the CHECK note in the source)",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config) if args.config else ROOT / "train" / f"config_{args.tenant}.yaml"
    config = load_config(config_path)
    if config.get("tenant") != args.tenant:
        print(
            f"train_lora_hf: --tenant is {args.tenant!r} but {config_path} says "
            f"{config.get('tenant')!r}. Refusing to run.",
            file=sys.stderr,
        )
        return 2

    plan = build_plan(config, config_path)
    settings = build_hf_settings(plan)
    out_dir = Path(args.out) if args.out else resolve_path(plan["hf_adapter_dir"])

    log_dir = resolve_path(plan["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_log = log_dir / f"train_hf_{args.tenant}_{stamp}.log"

    train_file = resolve_path(plan["train_file"])
    if not train_file.exists():
        print(
            f"train_lora_hf: no training data at {train_file}.\n"
            f"               Run: python data/generate.py package",
            file=sys.stderr,
        )
        return 2

    chat_rows = read_chat_jsonl(train_file)
    if not chat_rows:
        print(f"train_lora_hf: {train_file} has no chat rows", file=sys.stderr)
        return 2

    with run_log.open("w", encoding="utf-8") as handle:
        def emit(text=""):
            print(text)
            handle.write(text + "\n")

        emit(f"train_lora_hf: run log {run_log}")
        print_hf_plan(plan, settings, len(chat_rows))
        print_hf_plan(plan, settings, len(chat_rows), stream=handle)

        # Sanity check the data before spending GPU time on it. Every row must
        # carry the project's system message; a row that does not would train
        # the adapter on a prompt shape that never occurs at inference.
        expected_system = config.get("data", {}).get("system_message", "detailed thinking off")
        bad = 0
        for row in chat_rows:
            roles = [m.get("role") for m in row.get("messages", [])]
            systems = [m.get("content") for m in row.get("messages", [])
                       if m.get("role") == "system"]
            if roles != ["system", "user", "assistant"] or systems != [expected_system]:
                bad += 1
        if bad:
            emit(f"train_lora_hf: WARNING {bad}/{len(chat_rows)} rows are not "
                 f"[system,user,assistant] with system == {expected_system!r}")
        else:
            emit(f"train_lora_hf: all {len(chat_rows)} rows are "
                 f"[system,user,assistant] with the expected system message")

        if args.dry_run:
            emit("train_lora_hf: --dry-run, nothing imported and nothing trained.")
            emit(f"train_lora_hf: would write {out_dir}/adapter_config.json")
            emit(f"train_lora_hf: would write {out_dir}/adapter_model.safetensors")
            return 0

        torch, Dataset, LoraConfig, AutoTokenizer, SFTConfig, SFTTrainer = import_stack()

        emit(f"train_lora_hf: torch {torch.__version__}, cuda available "
             f"{torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            emit("train_lora_hf: no CUDA device visible. Refusing to start an 8B")
            emit("               fine-tune on CPU - it would not finish. Run this")
            emit("               on the Azure GPU.")
            return 2

        # Rows go in as-is. TRL sees a `messages` column, recognises the
        # conversational format and applies the tokenizer's chat template.
        dataset = Dataset.from_list([{"messages": row["messages"]} for row in chat_rows])
        emit(f"train_lora_hf: dataset {len(dataset)} rows, columns {dataset.column_names}")

        tokenizer = AutoTokenizer.from_pretrained(plan["hf_id"])
        if tokenizer.pad_token is None:
            # SFTTrainer: "A padding token, tokenizer.pad_token, must be set. If
            # the processing class has not set a padding token,
            # tokenizer.eos_token will be used as the default."
            tokenizer.pad_token = tokenizer.eos_token
            emit("train_lora_hf: pad_token was unset; using eos_token")

        peft_config = LoraConfig(
            r=settings["r"],
            lora_alpha=settings["lora_alpha"],
            lora_dropout=settings["lora_dropout"],
            target_modules=settings["target_modules"],
            bias=settings["bias"],
            task_type=settings["task_type"],
        )

        # CHECK: --assistant-only-loss is off by default. TRL documents it as
        # CHECK: conversational-only AND as requiring the chat template to
        # CHECK: contain {% generation %} / {% endgeneration %} markers, which
        # CHECK: it auto-patches only "for known model families (e.g. Qwen3)".
        # CHECK: Whether the Nemotron Nano template carries those markers is
        # CHECK: UNVERIFIED. Training on the full sequence (the default here)
        # CHECK: is the safe behaviour: it also teaches the prompt, which is
        # CHECK: harmless with a fixed system message and a single user turn.
        # CHECK: Turn the flag on once the template has been inspected, and
        # CHECK: record what it contained in change_log.md.
        sft_config = SFTConfig(
            output_dir=str(out_dir),
            num_train_epochs=settings["num_train_epochs"],
            per_device_train_batch_size=settings["per_device_train_batch_size"],
            gradient_accumulation_steps=settings["gradient_accumulation_steps"],
            learning_rate=settings["learning_rate"],
            lr_scheduler_type=settings["lr_scheduler_type"],
            warmup_steps=settings["warmup_steps"],
            weight_decay=settings["weight_decay"],
            max_length=settings["max_length"],
            bf16=settings["bf16"],
            seed=settings["seed"],
            logging_steps=settings["logging_steps"],
            save_strategy=settings["save_strategy"],
            report_to=settings["report_to"],
            gradient_checkpointing=settings["gradient_checkpointing"],
            assistant_only_loss=args.assistant_only_loss,
            model_init_kwargs={"dtype": torch.bfloat16},
        )

        trainer = SFTTrainer(
            model=plan["hf_id"],
            args=sft_config,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

        emit("train_lora_hf: starting SFTTrainer.train()")
        result = trainer.train()
        emit(f"train_lora_hf: train() returned {result}")

        out_dir.mkdir(parents=True, exist_ok=True)
        # save_pretrained on the PEFT-wrapped model writes the adapter only -
        # adapter_config.json + adapter_model.safetensors - not the base weights.
        trainer.model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        emit(f"train_lora_hf: adapter written to {out_dir}")

        # Report the real on-disk size. This is the number that retires the
        # 0.08 GB ESTIMATE in bench/economics.py and in ARCHITECTURE.md.
        total = 0
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                size = path.stat().st_size
                total += size
                emit(f"train_lora_hf:   {path.name:<40} {size:>12,} bytes")
        emit(f"train_lora_hf: ADAPTER SIZE ON DISK = {total:,} bytes "
             f"({total / 1e9:.4f} GB)")
        emit("train_lora_hf: feed that to bench/economics.py --adapter-gb "
             f"{total / 1e9:.4f}")

        summary = {
            "tenant": plan["tenant"],
            "route": "hf-peft-trl",
            "config": str(config_path),
            "base_model": plan["hf_id"],
            "train_rows": len(dataset),
            "adapter_dir": str(out_dir),
            "adapter_bytes": total,
            "adapter_gb": round(total / 1e9, 6),
            "settings": settings,
        }
        summary_path = log_dir / f"train_hf_{args.tenant}_{stamp}.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        emit(f"train_lora_hf: wrote {summary_path}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
