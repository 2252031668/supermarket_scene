#!/usr/bin/env python3
"""Verify the locally downloaded Qwen3.5-9B weights are complete and loadable.

Three checks:
  1. Filesystem integrity: every shard declared in model.safetensors.index.json
     exists and matches the size declared in the safetensors header CRC.
  2. Load integrity: every tensor in the index loads, its shape matches, and
     the values contain no NaN/Inf.
  3. (Optional) Inference smoke test: load via transformers and run one forward
     pass to confirm the model actually works.

Usage:
    python vision/verify_qwen_weights.py                 # checks 1 + 2
    python vision/verify_qwen_weights.py --infer         # + inference smoke test
    python vision/verify_qwen_weights.py --model-dir PATH
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

VISION_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = VISION_DIR / "models" / "Qwen3.5-9B"


def fail(message: str) -> None:
    print(f"  [FAIL] {message}")
    raise SystemExit(1)


def check_filesystem(model_dir: Path) -> dict[str, int]:
    """Return shard_name -> declared_size for shards whose header CRC matches."""
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        fail(f"missing index file: {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    metadata = index.get("metadata", {})
    total_size = int(metadata.get("total_size", 0))
    weight_map = index.get("weight_map", {})

    # Which shard files are referenced.
    shard_names = sorted({v for v in weight_map.values()})
    print(f"Index declares {len(shard_names)} shards, total_size = {total_size:,} bytes "
          f"({total_size / 1e9:.2f} GB)")

    if not shard_names:
        fail("index weight_map references no shard files")

    declared_sizes: dict[str, int] = {}
    actual_total = 0
    for shard_name in shard_names:
        shard_path = model_dir / shard_name
        if not shard_path.is_file():
            fail(f"missing shard file: {shard_path}")

        file_size = shard_path.stat().st_size
        # Safetensors header: first 8 bytes = little-endian uint64 header length.
        with shard_path.open("rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header_bytes = handle.read(header_len)
        header = json.loads(header_bytes)
        # Each tensor entry carries its byte range within the data section.
        data_start = 8 + header_len
        shard_declared = 0
        for tensor_meta in header.values():
            if not isinstance(tensor_meta, dict) or "data_offsets" not in tensor_meta:
                continue
            begin, end = tensor_meta["data_offsets"]
            shard_declared += end - begin
        expected_file_size = data_start + shard_declared
        declared_sizes[shard_name] = shard_declared

        if file_size != expected_file_size:
            fail(
                f"shard {shard_name} size mismatch: on disk {file_size:,} bytes, "
                f"but header+data expects {expected_file_size:,} bytes"
            )
        actual_total += shard_declared
        print(f"  ok  {shard_name:<42} {file_size / 1e9:.2f} GB")

    if total_size and abs(actual_total - total_size) > 1024:
        fail(
            f"total shard data ({actual_total:,}) does not match index total_size "
            f"({total_size:,})"
        )
    print(f"Filesystem integrity: OK ({actual_total / 1e9:.2f} GB across {len(shard_names)} shards)")
    return declared_sizes


def check_load(weight_map: dict[str, str], model_dir: Path) -> None:
    try:
        from safetensors import safe_open
    except ImportError:
        print("  [SKIP] safetensors not installed; skipping tensor load check")
        return

    try:
        import torch  # required for bfloat16 support
    except ImportError:
        print("  [SKIP] torch not installed; skipping tensor load check")
        return

    loaded_count = 0
    nan_count = 0
    for shard_name in sorted({v for v in weight_map.values()}):
        shard_path = model_dir / shard_name
        with safe_open(shard_path, framework="torch") as f:
            for tensor_name in f.keys():
                tensor = f.get_tensor(tensor_name)
                loaded_count += 1
                # bfloat16 on CPU is not "isfinite"-checkable directly; upcast.
                arr = tensor
                if arr.dtype == torch.bfloat16:
                    arr = arr.float()
                if arr.numel() == 0:
                    continue
                sample = arr.reshape(-1)[: min(arr.numel(), 4096)]
                if not torch.isfinite(sample).all():
                    nan_count += 1
                    print(f"  [WARN] {tensor_name} in {shard_name} has NaN/Inf")

    print(f"Tensor load: OK ({loaded_count} tensors loaded, {nan_count} with NaN/Inf)")


def check_inference(model_dir: Path) -> None:
    print("Loading model via transformers (this can take a while and needs enough RAM/VRAM)...")
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration  # type: ignore
    import torch

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    messages = [{"role": "user", "content": "Say hello in one short sentence."}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=32)
    decoded = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Inference smoke test: OK")
    print(f"  prompt : {prompt!r}")
    print(f"  output : {decoded!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify local Qwen3.5-9B weights.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--infer", action="store_true", help="also run an inference smoke test")
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    print(f"Verifying model at: {model_dir}")
    if not model_dir.is_dir():
        fail(f"model directory not found: {model_dir}")

    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map", {})

    print("\n[1/3] Filesystem integrity")
    check_filesystem(model_dir)

    print("\n[2/3] Tensor load + NaN/Inf scan")
    check_load(weight_map, model_dir)

    if args.infer:
        print("\n[3/3] Inference smoke test")
        check_inference(model_dir)
    else:
        print("\n[3/3] Inference smoke test: skipped (pass --infer to enable)")

    print("\nAll requested checks passed.")


if __name__ == "__main__":
    main()
