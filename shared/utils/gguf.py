"""
llama.cpp GGUF build / convert / quantize wrapper.

Idempotent at every step: clones llama.cpp if missing, builds the
``llama-quantize`` and ``llama-cli`` targets if their binaries don't
exist, converts the merged HF model to FP16 GGUF if not present, and
quantizes to each target level (Q4_K_M / Q5_K_M / Q8_0 by default)
only if the output isn't already there.

CPU-only build by design — pods often have CUDA drivers but no
``nvcc``, and ``llama-quantize`` is CPU-only anyway. Override via
``cmake_extra_args`` if you want a GPU-enabled build.

Two known footguns this wrapper handles
---------------------------------------
1. The convert script is named ``convert_hf_to_gguf.py`` in modern
   checkouts and ``convert-hf-to-gguf.py`` in older ones; we try both.
2. The CPU smoke test on a 7B 4-bit quant takes 3–5 min and the
   ``timeout=180`` we'd default to often hits before legitimate output
   completes — set ``GGUF_SKIP_SMOKE=1`` (or pass ``smoke=False``) to
   skip when you're confident in the conversion.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [gguf] {msg}", flush=True)


def _run(
    cmd: list[str] | str,
    cwd: str | Path | None = None,
    env: dict | None = None,
) -> None:
    if isinstance(cmd, list):
        _log(f"$ {' '.join(map(str, cmd))}")
    else:
        _log(f"$ {cmd}")
    r = subprocess.run(cmd, cwd=cwd, env=env, shell=isinstance(cmd, str), check=False)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}")


# ─────────────────────────────────────────────────────────────────────
# llama.cpp build
# ─────────────────────────────────────────────────────────────────────
DEFAULT_LLAMA_CPP_REPO = "https://github.com/ggerganov/llama.cpp"


def ensure_llama_cpp(
    llama_cpp_dir: str | Path,
    *,
    repo_url: str = DEFAULT_LLAMA_CPP_REPO,
    cmake_extra_args: Sequence[str] = (),
) -> Path:
    """
    Clone + build llama.cpp at ``llama_cpp_dir``. Returns the build's
    ``bin/`` directory (where ``llama-quantize`` and ``llama-cli`` live).
    """
    llama_cpp_dir = Path(llama_cpp_dir)
    if not llama_cpp_dir.exists():
        _log(f"cloning {repo_url} → {llama_cpp_dir}")
        _run(["git", "clone", "--depth", "1", repo_url, str(llama_cpp_dir)])

    build_dir = llama_cpp_dir / "build"
    bin_dir = build_dir / "bin"
    quantize_bin = bin_dir / "llama-quantize"
    cli_bin = bin_dir / "llama-cli"

    if quantize_bin.exists() and cli_bin.exists():
        _log(f"already built at {build_dir}")
        return bin_dir

    _log("building llama.cpp (CPU-only by default — pass cmake_extra_args for GPU)")
    cmake_args = ["cmake", "-B", "build", *cmake_extra_args]
    _run(cmake_args, cwd=llama_cpp_dir)
    _run(
        ["cmake", "--build", "build", "--config", "Release", "-j", "4",
         "--target", "llama-quantize", "llama-cli"],
        cwd=llama_cpp_dir,
    )

    if not quantize_bin.exists():
        raise RuntimeError(f"build produced no llama-quantize at {quantize_bin}")
    return bin_dir


def ensure_python_deps(llama_cpp_dir: str | Path) -> None:
    """
    Install llama.cpp's ``requirements.txt`` (or just ``gguf`` if it's
    not present) into the current Python env.
    """
    llama_cpp_dir = Path(llama_cpp_dir)
    req = llama_cpp_dir / "requirements.txt"
    if req.exists():
        _log(f"installing llama.cpp Python deps from {req}")
        _run([sys.executable, "-m", "pip", "install", "--quiet", "-r", str(req)])
    else:
        _run([sys.executable, "-m", "pip", "install", "--quiet", "gguf"])


# ─────────────────────────────────────────────────────────────────────
# Convert + quantize
# ─────────────────────────────────────────────────────────────────────
def convert_to_fp16(
    merged_dir: str | Path,
    out_path: str | Path,
    llama_cpp_dir: str | Path,
) -> Path:
    """
    Run ``convert_hf_to_gguf.py`` to produce an FP16 GGUF. Skips work
    if ``out_path`` already exists with non-trivial size (>100 MB).
    """
    merged_dir = Path(merged_dir)
    out_path = Path(out_path)
    llama_cpp_dir = Path(llama_cpp_dir)

    if out_path.exists() and out_path.stat().st_size > 100 * 1024 * 1024:
        _log(f"f16 GGUF exists ({out_path.stat().st_size/1e9:.2f} GB) — skip")
        return out_path

    converter = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not converter.exists():
        alt = llama_cpp_dir / "convert-hf-to-gguf.py"
        if alt.exists():
            converter = alt
        else:
            raise FileNotFoundError(
                f"no convert_hf_to_gguf.py in {llama_cpp_dir}"
            )

    _log(f"converting {merged_dir} → {out_path} (FP16)")
    _run([
        sys.executable, str(converter), str(merged_dir),
        "--outfile", str(out_path),
        "--outtype", "f16",
    ])
    return out_path


def quantize_one(
    f16_path: str | Path,
    out_path: str | Path,
    quant: str,
    quantize_bin: str | Path,
) -> Path:
    """Run ``llama-quantize`` for a single target. Idempotent."""
    f16_path = Path(f16_path)
    out_path = Path(out_path)
    if out_path.exists() and out_path.stat().st_size > 100 * 1024 * 1024:
        _log(f"{quant} exists ({out_path.stat().st_size/1e9:.2f} GB) — skip")
        return out_path
    _log(f"quantizing → {quant}")
    _run([str(quantize_bin), str(f16_path), str(out_path), quant])
    return out_path


DEFAULT_QUANTS: tuple[str, ...] = ("Q4_K_M", "Q5_K_M", "Q8_0")


def build_quants(
    merged_dir: str | Path,
    output_dir: str | Path,
    *,
    model_shortname: str,
    llama_cpp_dir: str | Path = "/workspace/llama.cpp",
    quants: Sequence[str] = DEFAULT_QUANTS,
    smoke: bool | None = None,
    smoke_prompt: str = "Explain photosynthesis in one sentence.",
    cmake_extra_args: Sequence[str] = (),
) -> dict[str, Path]:
    """
    Full GGUF build pipeline:
      1. Ensure llama.cpp is cloned + built (CPU-only by default).
      2. Install convert script's Python deps.
      3. Convert merged HF model → ``{shortname}-f16.gguf``.
      4. For each quant in ``quants``, produce ``{shortname}-{quant}.gguf``.
      5. Optional CPU smoke test on the smallest produced quant.

    Returns a ``{quant: Path}`` map.
    """
    merged_dir = Path(merged_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bin_dir = ensure_llama_cpp(llama_cpp_dir, cmake_extra_args=cmake_extra_args)
    ensure_python_deps(llama_cpp_dir)
    quantize_bin = bin_dir / "llama-quantize"
    cli_bin = bin_dir / "llama-cli"

    f16_path = output_dir / f"{model_shortname}-f16.gguf"
    convert_to_fp16(merged_dir, f16_path, llama_cpp_dir)
    _log(f"f16 size: {f16_path.stat().st_size/1e9:.2f} GB")

    outputs: dict[str, Path] = {}
    for q in quants:
        out = output_dir / f"{model_shortname}-{q}.gguf"
        quantize_one(f16_path, out, q, quantize_bin)
        outputs[q] = out
        _log(f"{q} size: {out.stat().st_size/1e9:.2f} GB")

    # Smoke test on the smallest-by-name quant
    if smoke is None:
        smoke = os.environ.get("GGUF_SKIP_SMOKE", "0") != "1"
    if smoke and outputs:
        target = min(outputs.values(), key=lambda p: p.stat().st_size)
        ok = smoke_test(cli_bin, target, prompt=smoke_prompt)
        if not ok:
            _log(f"!! smoke test on {target.name} failed or timed out — "
                 "review before upload")

    return outputs


def smoke_test(
    cli_bin: str | Path,
    model_path: str | Path,
    *,
    prompt: str = "Explain photosynthesis in one sentence.",
    n_predict: int = 64,
    temp: float = 0.2,
    timeout: float = 300.0,
) -> bool:
    """One short CPU generation. Returns ``True`` if output looks healthy."""
    cli_bin = Path(cli_bin)
    model_path = Path(model_path)
    _log(f"smoke test {model_path.name}")
    try:
        r = subprocess.run(
            [str(cli_bin), "-m", str(model_path),
             "-p", prompt, "-n", str(n_predict),
             "--temp", str(temp), "-ngl", "0"],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        _log("smoke test timed out")
        return False
    if r.returncode != 0:
        _log(f"smoke test nonzero exit {r.returncode}: {r.stderr[-500:]}")
        return False

    lines = [l for l in r.stdout.split("\n") if l.strip() and not l.startswith("llama_")]
    sample = "\n".join(lines[-10:])[:400]
    _log(f"output (last ~400 chars): {sample!r}")
    return len(sample) >= 20
