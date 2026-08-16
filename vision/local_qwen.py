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


def parse_grounding_box(response: str) -> list[int] | None:
    """Parse exactly one Qwen normalized ``bbox_2d`` box."""
    value = response.strip()
    if value.startswith("```") and value.endswith("```"):
        value = value.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(value)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if isinstance(payload, list):
        if len(payload) != 1 or not isinstance(payload[0], dict):
            return None
        payload = payload[0]
    if not isinstance(payload, dict):
        return None
    box = payload.get("bbox_2d")
    if not isinstance(box, list) or len(box) != 4:
        return None
    if any(not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 999 for value in box):
        return None
    x1, y1, x2, y2 = box
    return box if x2 > x1 and y2 > y1 else None


def parse_grounding_prompt(response: str) -> str | None:
    """Parse a short SKU-only visual description from Qwen."""
    value = response.strip()
    if value.startswith("```") and value.endswith("```"):
        value = value.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(value)
        prompt = payload["prompt"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if not isinstance(prompt, str):
        return None
    prompt = prompt.strip()
    for phrase in ("左侧的", "右侧的", "旁边的", "左侧", "右侧", "旁边"):
        prompt = prompt.replace(phrase, "")
    if not 6 <= len(prompt) <= 48 or any(word in prompt for word in ("红框", "货架", "背景")):
        return None
    return prompt


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


def ground_sku_response(robot_image: bytes, sku: str, owlv2_prompt: str, model_dir: str | Path,
                        reference_image: bytes | None = None, reference_kind: str | None = None,
                        max_new_tokens: int = 96, visual_prompt_label: str = "英文视觉描述") -> tuple[list[int] | None, str]:
    """Locate one SKU in a robot image, optionally using a fixed reference image."""
    processor, model = _runtime(model_dir)
    fused_text = f"中文 SKU 名称：{sku}"
    if owlv2_prompt.strip():
        fused_text += f"\n{visual_prompt_label}：{owlv2_prompt.strip()}"
    if reference_image is None:
        image_note = "图 1 是机器人局部货架图。"
        images = [Image.open(BytesIO(robot_image)).convert("RGB")]
    else:
        reference_note = "SKU 主图" if reference_kind == "sku_main" else "标准示例图，红框标出一个目标商品实例"
        image_note = f"图 1 是目标商品的{reference_note}。图 2 是机器人局部货架图。只输出图 2 的坐标。"
        images = [Image.open(BytesIO(reference_image)).convert("RGB"), Image.open(BytesIO(robot_image)).convert("RGB")]
    prompt = (
        f"{image_note}\n{fused_text}\n"
        "在机器人图中定位这个 SKU 的一个最确定、完整可见的商品本体，不要框货架、价签或相邻商品。"
        "只输出一个 JSON 对象，格式严格为 {\"bbox_2d\":[x_min,y_min,x_max,y_max]}。"
        "坐标只对应机器人图，范围为 0 到 999。不要输出解释或 Markdown。"
    )
    messages = [{"role": "user", "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]}]
    text = render_chat_prompt(processor, messages)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)
    import torch

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    answer = processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return parse_grounding_box(answer), answer


def few_shot_grounding_messages(sku: str, visual_prompt: str, example_box: list[int]) -> list[dict[str, Any]]:
    """Build one completed grounding example followed by the current task."""
    example_answer = json.dumps({"bbox_2d": example_box}, separators=(",", ":"))
    history_prompt = (
        "这是一个已完成的历史示例。图 1 是货架局部图。"
        f"中文 SKU 名称：{sku}\nQwen 专用中文视觉描述：{visual_prompt.strip()}\n"
        "请在图 1 中定位一个最确定、完整可见的商品本体。坐标范围为 0 到 999，只输出 JSON。"
    )
    current_prompt = (
        "现在处理新任务。图 1 是新的机器人局部货架图。"
        f"中文 SKU 名称：{sku}\nQwen 专用中文视觉描述：{visual_prompt.strip()}\n"
        "历史 assistant 的 box 只属于历史示例图，不属于当前图。"
        "请在当前图中定位一个最确定、完整可见的商品本体，不要框货架、价签或相邻商品。"
        "只输出一个 JSON 对象，格式严格为 {\"bbox_2d\":[x_min,y_min,x_max,y_max]}。"
        "坐标只对应当前图，范围为 0 到 999。不要输出解释或 Markdown。"
    )
    return [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": history_prompt}]},
        {"role": "assistant", "content": example_answer},
        {"role": "user", "content": "示例结束。上述坐标只属于历史示例图；下一张图必须重新计算坐标。请回复“明白”。"},
        {"role": "assistant", "content": "明白"},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": current_prompt}]},
    ]


def ground_sku_few_shot_response(robot_image: bytes, example_image: bytes, sku: str, visual_prompt: str,
                                 example_box: list[int], model_dir: str | Path,
                                 max_new_tokens: int = 96) -> tuple[list[int] | None, str]:
    """Ground one SKU after one completed, unannotated image example."""
    if parse_grounding_box(json.dumps({"bbox_2d": example_box})) is None:
        raise ValueError("example_box must be one valid normalized box")
    processor, model = _runtime(model_dir)
    messages = few_shot_grounding_messages(sku, visual_prompt, example_box)
    images = [Image.open(BytesIO(example_image)).convert("RGB"), Image.open(BytesIO(robot_image)).convert("RGB")]
    text = render_chat_prompt(processor, messages)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)
    import torch

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    answer = processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return parse_grounding_box(answer), answer


def describe_sku_reference_response(reference_image: bytes, sku: str, model_dir: str | Path,
                                    max_new_tokens: int = 160) -> tuple[str | None, str]:
    """Create a Qwen-specific visual description from a fixed red-box example."""
    processor, model = _runtime(model_dir)
    prompt = (
        "图 1 是手机拍摄的货架局部图。红框内的商品是唯一目标 SKU，名称为“" + sku + "”。\n"
        "输出 3 到 4 个最有区分度的可见特征：第一项必须是具体包装配色（如白橙渐变，不能只写白）；"
        "第二项是品牌或关键文字；至少一项必须是 SKU 名称以外的视觉特征，如图案、色块、瓶盖、泵头或袋装。"
        "不要复述 SKU 名称中的完整口味词。"
        "不要描述红框、货架、位置、左右关系、相邻商品、数量或背景。\n"
        "用分号连接短特征，目标总长度为 20 到 45 个中文字符，后续特征不得重复。"
        "只输出 JSON：{\"prompt\":\"主色；品牌文字；口味或图案\"}，不要 Markdown 或解释。"
    )
    image = Image.open(BytesIO(reference_image)).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = render_chat_prompt(processor, messages)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    import torch

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    answer = processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return parse_grounding_prompt(answer), answer


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
