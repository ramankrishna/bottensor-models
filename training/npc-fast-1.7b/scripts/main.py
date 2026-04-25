"""
NPC Fast — unified CLI.

    python main.py train                        # train from scratch
    python main.py train --resume_from_checkpoint path/to/ckpt
    python main.py eval --checkpoint path/to/ckpt
    python main.py export --checkpoint path/to/ckpt
    python main.py status
    python main.py preview
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from config import (
    BASE_MODEL,
    CHECKPOINT_DIR,
    CURRICULUM,
    DATASETS_REGISTRY,
    EVAL_RESULTS_PATH,
    HF_REPO_FP16,
    HF_REPO_GGUF,
    HF_REPO_GPTQ,
    MODEL_NAME,
    TRAINING_CONFIG,
)
from data.curriculum import describe_schedule

LOG = logging.getLogger("npc-fast")

HERE = Path(__file__).resolve().parent


# ---------------- train ----------------

def cmd_train(args: argparse.Namespace) -> None:
    from train import run
    run(
        model_name=args.model_name,
        resume_from_checkpoint=args.resume_from_checkpoint,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        target_step=args.target_step,
    )


# ---------------- eval ----------------

def _run_module(mod: str, argv: list[str]) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.call([sys.executable, "-m", mod, *argv], env=env, cwd=str(HERE))


def cmd_eval(args: argparse.Namespace) -> None:
    ck = args.checkpoint
    rc = 0
    rc |= _run_module("eval.benchmarks", ["--model_path", ck])
    rc |= _run_module("eval.context_eval", ["--model_path", ck,
                                             "--max_context", str(args.max_context)])
    rc |= _run_module("eval.router_eval", ["--model_path", ck])
    rc |= _run_module("eval.perplexity", ["--model_path", ck])
    if rc != 0:
        sys.exit(rc)
    print(f"\nEval results → {EVAL_RESULTS_PATH}")


# ---------------- export ----------------

def cmd_export(args: argparse.Namespace) -> None:
    ck = args.checkpoint
    gptq_dir = args.output_root / "npc-fast-1.7b-gptq"
    gguf_dir = args.output_root / "npc-fast-1.7b-gguf"

    rc = 0
    if "gptq" in args.artifacts:
        rc |= _run_module("export.quantize", ["--model_path", ck,
                                               "--output_dir", str(gptq_dir)])
    if "gguf" in args.artifacts:
        rc |= _run_module("export.gguf", ["--model_path", ck,
                                           "--output_dir", str(gguf_dir)])
    if rc != 0:
        sys.exit(rc)

    if args.push:
        if "bf16" in args.artifacts:
            _run_module("export.push_hf", ["--model_path", ck, "--repo", HF_REPO_FP16])
        if "gptq" in args.artifacts:
            _run_module("export.push_hf", ["--model_path", str(gptq_dir), "--repo", HF_REPO_GPTQ])
        if "gguf" in args.artifacts:
            _run_module("export.push_hf", ["--model_path", str(gguf_dir), "--repo", HF_REPO_GGUF])


# ---------------- status ----------------

def cmd_status(_args: argparse.Namespace) -> None:
    print(f"=== {MODEL_NAME} ===")
    print(f"Base model: {BASE_MODEL}")
    print(f"Output dir: {CHECKPOINT_DIR}")
    print(f"Datasets registry: {DATASETS_REGISTRY}")
    print()
    print("Training config:")
    for k, v in TRAINING_CONFIG.items():
        print(f"  {k:28s} {v}")
    print()
    print(describe_schedule())
    print()

    # Checkpoint inventory
    ckpts = sorted(CHECKPOINT_DIR.glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1)
    if not ckpts:
        print("No checkpoints yet.")
    else:
        print("Checkpoints:")
        for p in ckpts:
            print(f"  {p.name}")
        final = CHECKPOINT_DIR / "final"
        if final.exists():
            print(f"  final/     <- last saved full model")

    if EVAL_RESULTS_PATH.exists():
        print()
        print(f"Latest eval: {EVAL_RESULTS_PATH}")
        try:
            data = json.loads(EVAL_RESULTS_PATH.read_text())
            for section, body in data.items():
                if isinstance(body, dict):
                    keys = [f"{k}={v}" for k, v in body.items() if isinstance(v, (int, float))]
                    print(f"  [{section}] " + ", ".join(keys[:6]))
        except json.JSONDecodeError:
            pass


# ---------------- preview ----------------

def cmd_preview(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from data.loader import preview
    preview(DATASETS_REGISTRY, k=args.k)


# ---------------- CLI wiring ----------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="npc-fast", description=f"{MODEL_NAME} CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Full-weight continual pre-training")
    t.add_argument("--model_name", default=BASE_MODEL)
    t.add_argument("--output_dir", default=str(CHECKPOINT_DIR))
    t.add_argument("--max_steps", type=int, default=None)
    t.add_argument("--target_step", type=int, default=None,
                   help="Stop when global_step reaches this. Use for curriculum-resume.")
    t.add_argument("--learning_rate", type=float, default=None)
    t.add_argument("--resume_from_checkpoint", default=None)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval", help="Run the full eval suite")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--max_context", type=int, default=131_072)
    e.set_defaults(func=cmd_eval)

    x = sub.add_parser("export", help="Quantize + (optionally) push to HF")
    x.add_argument("--checkpoint", required=True)
    x.add_argument("--output_root", type=Path, default=HERE / "output")
    x.add_argument("--artifacts", nargs="+",
                   default=["gptq", "gguf", "bf16"],
                   choices=["bf16", "gptq", "gguf"])
    x.add_argument("--push", action="store_true", help="Upload to HuggingFace")
    x.set_defaults(func=cmd_export)

    s = sub.add_parser("status", help="Show training progress and config")
    s.set_defaults(func=cmd_status)

    pv = sub.add_parser("preview", help="Show a few random examples per dataset")
    pv.add_argument("--k", type=int, default=5)
    pv.set_defaults(func=cmd_preview)

    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
