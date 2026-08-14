"""Local Qwen3.5 candidate chooser for RGB-D stockout classification."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from vision.dino import resolve_device


_RUNTIME: tuple[Any, Any] | None = None


def parse_candidate_choice(response: str, candidate_count: int) -> int | None:
    value = response.strip()
    if value.lower() == "unknown":
        return None
    if value.startswith("```") and value.endswith("```"):
        value = value.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(value)
        choice = payload["choice"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if isinstance(choice, str) and choice.isdecimal():
        choice = int(choice)
    return choice if isinstance(choice, int) and not isinstance(choice, bool) and 1 <= choice <= candidate_count else None


def render_chat_prompt(processor: Any, messages: list[dict[str, Any]]) -> str:
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def _runtime(model_dir: str | Path) -> tuple[Any, Any]:
    global _RUNTIME
    if _RUNTIME is None:
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        directory = Path(model_dir).expanduser().resolve()
        if not directory.is_dir():
            raise RuntimeError(f"Local Qwen model directory is missing: {directory}")
        processor = AutoProcessor.from_pretrained(directory, trust_remote_code=True)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            directory, torch_dtype="auto", trust_remote_code=True
        ).to(resolve_device()).eval()
        _RUNTIME = processor, model
    return _RUNTIME


def choose_candidate_response(full_image: bytes, board_image: bytes, candidate_count: int, model_dir: str | Path,
                              max_new_tokens: int = 48) -> tuple[int | None, str]:
    processor, model = _runtime(model_dir)
    prompt = (
        "图一是完整货架图，红框标出了需要判断的后排商品。图二是候选SKU参考图，"
        "编号和名称是唯一可选项。红框内商品对应图二哪一项？"
        "只输出 {\"choice\": 1} 到 {\"choice\": " + str(candidate_count)
        + "}，无法判断时只输出 unknown。"
    )
    images = [Image.open(BytesIO(full_image)).convert("RGB"), Image.open(BytesIO(board_image)).convert("RGB")]
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": prompt}]}]
    text = render_chat_prompt(processor, messages)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)
    import torch

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    answer = processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return parse_candidate_choice(answer, candidate_count), answer


def choose_candidate(full_image: bytes, board_image: bytes, candidate_count: int, model_dir: str | Path,
                     max_new_tokens: int = 48) -> int | None:
    return choose_candidate_response(full_image, board_image, candidate_count, model_dir, max_new_tokens)[0]


def detect_misplacement_response(image_bytes: bytes, model_dir: str | Path, max_new_tokens: int = 384) -> str:
    """Ask Qwen to find only clearly misplaced products in one complete image."""
    processor, model = _runtime(model_dir)
    prompt = (
        "这是一张局部货架 RGB 图片。多数图片没有异常。仅当某件商品明显不同于其同一层左右相邻的"
        "连续同类商品序列时，才把它判为异常摆放；不确定时不要报告。最多报告两个异常商品。"
        "只输出 JSON：{\"boxes\":[{\"x\":整数,\"y\":整数,\"width\":整数,\"height\":整数,"
        "\"confidence\":0到1的小数,\"reason\":\"简短中文原因\"}]}。没有异常时输出 {\"boxes\":[]}。"
        "坐标以图片左上角为原点，必须是完整图片中的像素坐标。"
    )
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = render_chat_prompt(processor, messages)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    import torch

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
