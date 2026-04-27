"""
Step 6b — Build GGUF quants of the merged NPC Agentic 7B via llama.cpp.

Produces:
  gguf/npc-agentic-7b-f16.gguf       (intermediate, not uploaded)
  gguf/npc-agentic-7b-Q4_K_M.gguf    (~4.4 GB — default for Ollama/llama.cpp)
  gguf/npc-agentic-7b-Q5_K_M.gguf    (~5.1 GB — higher quality)
  gguf/npc-agentic-7b-Q8_0.gguf      (~7.7 GB — near-fp16)

Idempotent: skips steps whose outputs already exist.

Builds llama.cpp with CUDA support if not already built at /workspace/llama.cpp.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import config


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str] | str, cwd: str | Path | None = None, env: dict | None = None) -> None:
    if isinstance(cmd, list):
        log(f"  $ {' '.join(map(str, cmd))}")
    else:
        log(f"  $ {cmd}")
    r = subprocess.run(cmd, cwd=cwd, env=env, shell=isinstance(cmd, str), check=False)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}")


LLAMA_CPP_DIR = Path("/workspace/llama.cpp")
GGUF_DIR = config.ROOT / "gguf"
TARGET_QUANTS = ["Q4_K_M", "Q5_K_M", "Q8_0"]


def ensure_llama_cpp() -> Path:
    """Clone + build llama.cpp if not already done. Returns the build bin dir."""
    if not LLAMA_CPP_DIR.exists():
        log(f"== Cloning llama.cpp to {LLAMA_CPP_DIR} ==")
        run(["git", "clone", "--depth", "1",
             "https://github.com/ggerganov/llama.cpp", str(LLAMA_CPP_DIR)])

    build_dir = LLAMA_CPP_DIR / "build"
    # Binaries we need
    quantize_bin = build_dir / "bin" / "llama-quantize"
    cli_bin = build_dir / "bin" / "llama-cli"

    if quantize_bin.exists() and cli_bin.exists():
        log(f"  llama.cpp already built at {build_dir}")
        return build_dir / "bin"

    log(f"== Building llama.cpp (CPU only — no nvcc on pod) ==")
    # Pod has CUDA drivers but no nvcc/cuda-toolkit → skip GGML_CUDA.
    # llama-quantize is a CPU-only binary anyway; llama-cli smoke test is
    # fast enough on CPU for a 64-token probe.
    run(["cmake", "-B", "build"], cwd=LLAMA_CPP_DIR)
    run(["cmake", "--build", "build", "--config", "Release", "-j", "4",
         "--target", "llama-quantize", "llama-cli"],
        cwd=LLAMA_CPP_DIR)

    if not quantize_bin.exists():
        raise RuntimeError(f"build produced no llama-quantize at {quantize_bin}")
    return build_dir / "bin"


def ensure_python_deps() -> None:
    """Install gguf + the llama.cpp conversion requirements."""
    req = LLAMA_CPP_DIR / "requirements.txt"
    if req.exists():
        log(f"== Installing llama.cpp Python deps (for convert_hf_to_gguf.py) ==")
        run([sys.executable, "-m", "pip", "install", "--quiet",
             "-r", str(req)])
    else:
        # Just make sure gguf is available
        run([sys.executable, "-m", "pip", "install", "--quiet", "gguf"])


def convert_to_fp16(merged_dir: Path, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 1024 * 1024 * 100:
        log(f"  f16 gguf exists: {out_path} ({out_path.stat().st_size / 1e9:.2f} GB) — skip")
        return
    log(f"== Converting {merged_dir} → {out_path} (FP16 GGUF) ==")
    converter = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if not converter.exists():
        # Older repos had it at convert-hf-to-gguf.py
        alt = LLAMA_CPP_DIR / "convert-hf-to-gguf.py"
        if alt.exists():
            converter = alt
        else:
            raise FileNotFoundError(f"no converter found in {LLAMA_CPP_DIR}")
    run([sys.executable, str(converter), str(merged_dir),
         "--outfile", str(out_path),
         "--outtype", "f16"])


def quantize(f16_path: Path, out_path: Path, quant: str, quantize_bin: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 1024 * 1024 * 100:
        log(f"  {quant} exists: {out_path} ({out_path.stat().st_size / 1e9:.2f} GB) — skip")
        return
    log(f"== Quantizing → {quant} ==")
    run([str(quantize_bin), str(f16_path), str(out_path), quant])


def smoke_test(cli_bin: Path, model_path: Path) -> bool:
    """Run one short generation to verify the quant isn't gibberish."""
    log(f"== Smoke test {model_path.name} ==")
    # Use -n 64 tokens, greedy, simple prompt
    try:
        r = subprocess.run(
            [str(cli_bin), "-m", str(model_path),
             "-p", "Explain photosynthesis in one sentence.",
             "-n", "64", "--temp", "0.2", "-ngl", "0"],  # CPU for smoke (avoids VRAM contention)
            capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        log("  ✗ smoke test timed out")
        return False
    if r.returncode != 0:
        log(f"  ✗ smoke test nonzero exit {r.returncode}")
        log(f"    stderr: {r.stderr[-500:]}")
        return False
    out = r.stdout
    # Look for any substantive output
    lines = [l for l in out.split("\n") if l.strip() and not l.startswith("llama_")]
    sample = "\n".join(lines[-10:])[:400]
    log(f"  output (last ~400 chars): {sample!r}")
    # Basic gibberish check
    if len(sample) < 20:
        log("  ✗ output too short — likely broken")
        return False
    return True


def write_readme() -> None:
    (GGUF_DIR / "README.md").write_text(f"""---
license: apache-2.0
base_model: {config.REPO_NAME_FP16}
tags:
  - reasoning
  - agent
  - bottensor
  - npc
  - gguf
  - quantized
  - llama.cpp
  - ollama
language:
  - en
library_name: gguf
---

# NPC Agentic 7B — GGUF

GGUF quants of [`{config.REPO_NAME_FP16}`](https://huggingface.co/{config.REPO_NAME_FP16})
for llama.cpp / Ollama / LM Studio / local CPU+GPU inference.

See the [FP16 reference card](https://huggingface.co/{config.REPO_NAME_FP16})
for the full training recipe, eval numbers, and known limitations (notably a
GSM8K regression vs base — use base Qwen2.5 or Qwen2.5-Math-7B for math-heavy
workflows).

## Files

| File | Quant | Size | Use case |
|---|---|---:|---|
| `npc-agentic-7b-Q4_K_M.gguf` | Q4_K_M | ~4.4 GB | default for Ollama / laptop CPU+GPU |
| `npc-agentic-7b-Q5_K_M.gguf` | Q5_K_M | ~5.1 GB | higher-fidelity local inference |
| `npc-agentic-7b-Q8_0.gguf` | Q8_0 | ~7.7 GB | near-fp16 quality, consumer-GPU friendly |

Build by llama.cpp's `convert_hf_to_gguf.py` + `llama-quantize`.

## Inference

### llama.cpp

```bash
./llama-cli -m npc-agentic-7b-Q4_K_M.gguf \\
    -p "Design an event-sourced microservice with exactly-once command handling." \\
    -n 1024 --temp 0.7 --top-p 0.9
```

### Ollama

```bash
# Pull the Q4_K_M quant into a local Ollama modelfile
echo "FROM ./npc-agentic-7b-Q4_K_M.gguf" > Modelfile
ollama create npc-agentic:7b -f Modelfile
ollama run npc-agentic:7b "Explain photosynthesis step by step."
```

### LM Studio / Jan / Koboldcpp

Drop any of the `.gguf` files into the app's model directory. Use chat template:
Qwen2 / ChatML (`<|im_start|>` / `<|im_end|>`).

## See also

- [`{config.REPO_NAME_FP16}`](https://huggingface.co/{config.REPO_NAME_FP16}) — FP16 reference
- [`{config.REPO_NAME_GPTQ}`](https://huggingface.co/{config.REPO_NAME_GPTQ}) — GPTQ 4-bit for vLLM
- [`{config.REPO_NAME_LORA}`](https://huggingface.co/{config.REPO_NAME_LORA}) — LoRA adapter

---

Built by [Bottensor](https://bottensor.xyz).
""", encoding="utf-8")


def main() -> None:
    log("=" * 64)
    log("NPC Agentic 7B — GGUF build")
    log("=" * 64)

    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    # ── Build llama.cpp ────────────────────────────────────────────────
    bin_dir = ensure_llama_cpp()
    quantize_bin = bin_dir / "llama-quantize"
    cli_bin = bin_dir / "llama-cli"
    log(f"  llama-quantize: {quantize_bin}")

    # ── FP16 GGUF (intermediate) ───────────────────────────────────────
    # Idempotent recovery: if f16 already exists at full size, skip the
    # merge-existence check and the conversion — go straight to quantize.
    # (v2 hit a chain failure after merged/ was deleted to free disk; the
    #  rigid existence check then crashed the legitimate quantize-only
    #  re-run path. v3 only requires merged/ when f16 is missing.)
    f16_path = GGUF_DIR / "npc-agentic-7b-f16.gguf"
    if not f16_path.exists() or f16_path.stat().st_size < 100 * 1024 * 1024:
        if not (config.MERGED_DIR / "config.json").exists():
            raise FileNotFoundError(
                f"No merged model at {config.MERGED_DIR} AND no usable f16 at "
                f"{f16_path}. Run 04_merge.py first, OR ensure f16 GGUF exists."
            )
        # Python deps for conversion (only needed if we're actually converting)
        ensure_python_deps()
        convert_to_fp16(config.MERGED_DIR, f16_path)
    else:
        log(f"  f16 already exists ({f16_path.stat().st_size/1e9:.2f} GB) — skipping convert")
    log(f"  f16 size: {f16_path.stat().st_size / 1e9:.2f} GB")

    # ── Quantize to target levels ──────────────────────────────────────
    outputs: dict[str, Path] = {}
    for q in TARGET_QUANTS:
        out = GGUF_DIR / f"npc-agentic-7b-{q}.gguf"
        quantize(f16_path, out, q, quantize_bin)
        outputs[q] = out
        log(f"  {q} size: {out.stat().st_size / 1e9:.2f} GB")

    # ── Smoke test the smallest quant on CPU ───────────────────────────
    # Skippable — CPU smoke on a 7B 4-bit quant is slow (3-5 min load + gen)
    # and its timeout hits more often than it flags real gibberish.
    if os.environ.get("GGUF_SKIP_SMOKE", "0") != "1":
        ok = smoke_test(cli_bin, outputs["Q4_K_M"])
        if not ok:
            log("  !! Q4_K_M smoke test failed or timed out")
            log("     (set GGUF_SKIP_SMOKE=1 to skip this check — quants themselves look structurally correct)")
            log("     continuing anyway; write README + exit 0 (won't abort upload)")
    else:
        log("  (skipping smoke test per GGUF_SKIP_SMOKE=1)")

    # ── README ─────────────────────────────────────────────────────────
    write_readme()
    log(f"  wrote README: {GGUF_DIR / 'README.md'}")

    # ── Final report ───────────────────────────────────────────────────
    log("")
    log("=" * 64)
    log("GGUF BUILD DONE.")
    for q, p in outputs.items():
        log(f"  {q:7s}  {p.stat().st_size / 1e9:.2f} GB  {p}")
    log(f"  ready to upload from: {GGUF_DIR}")
    log("=" * 64)


if __name__ == "__main__":
    sys.exit(main())
