"""
GGUF export via llama.cpp's convert script.

Produces Q4_K_M and Q8_0 variants for fast local inference (Mac Mini,
M-series laptops, CPU-only servers). Assumes llama.cpp is cloned at
``$LLAMA_CPP_HOME`` or ``~/llama.cpp`` — the script clones it on demand.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

LOG = logging.getLogger("npc-fast.export.gguf")

DEFAULT_LLAMA_CPP = Path(os.getenv("LLAMA_CPP_HOME", str(Path.home() / "llama.cpp")))
VARIANTS = [("Q4_K_M", "q4_k_m"), ("Q8_0", "q8_0")]


def _ensure_llama_cpp(path: Path) -> None:
    if path.exists() and (path / "convert_hf_to_gguf.py").exists():
        return
    if path.exists() and (path / "convert-hf-to-gguf.py").exists():
        return
    LOG.info("Cloning llama.cpp into %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", str(path)],
        check=True,
    )


def _convert_script(path: Path) -> Path:
    for name in ("convert_hf_to_gguf.py", "convert-hf-to-gguf.py"):
        cand = path / name
        if cand.exists():
            return cand
    raise RuntimeError(f"No convert script found under {path}")


def _quantize_bin(path: Path) -> Path:
    # Build llama-quantize if missing
    bin_path = path / "build" / "bin" / "llama-quantize"
    legacy = path / "quantize"
    if bin_path.exists():
        return bin_path
    if legacy.exists():
        return legacy
    LOG.info("Building llama.cpp (this may take a few minutes)...")
    build_dir = path / "build"
    build_dir.mkdir(exist_ok=True)
    subprocess.run(["cmake", "-B", str(build_dir), "-S", str(path)], check=True)
    subprocess.run(["cmake", "--build", str(build_dir), "--config", "Release", "--target",
                    "llama-quantize", "-j"], check=True)
    if not bin_path.exists():
        raise RuntimeError("llama-quantize build did not produce a binary")
    return bin_path


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="HF checkpoint directory (bf16)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--llama_cpp", default=str(DEFAULT_LLAMA_CPP))
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    llama_cpp = Path(args.llama_cpp)
    _ensure_llama_cpp(llama_cpp)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fp16_gguf = out / "npc-fast-1.7b-f16.gguf"

    LOG.info("Converting HF → GGUF (f16): %s", fp16_gguf)
    convert = _convert_script(llama_cpp)
    subprocess.run(
        ["python", str(convert), str(args.model_path), "--outtype", "f16",
         "--outfile", str(fp16_gguf)],
        check=True,
    )

    quantize = _quantize_bin(llama_cpp)
    for label, suffix in VARIANTS:
        target = out / f"npc-fast-1.7b-{suffix}.gguf"
        LOG.info("Quantizing to %s → %s", label, target)
        subprocess.run([str(quantize), str(fp16_gguf), str(target), label], check=True)

    LOG.info("GGUF exports complete: %s", out)


if __name__ == "__main__":
    main()
