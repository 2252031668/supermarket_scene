#!/usr/bin/env python3
"""Locate a reference SKU in a shelf photo with Ark or Qwen VLM providers."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image
from volcenginesdkarkruntime import Ark

from vision.ark_grounding import (
    DEFAULT_MODEL,
    REQUEST_TIMEOUT_SECONDS,
    BoundingBox,
    count_overlapping_box_pairs,
    draw_boxes,
    extract_boxes,
    normalized_to_pixels,
)
from vision.config import get_api_key, require_api_key


VISION_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = VISION_DIR / "output" / "vlm_sku_query"
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1/chat/completions"
DEFAULT_SILICONFLOW_MODEL = "Qwen/Qwen3.6-35B-A3B"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_DASHSCOPE_MODEL = "qwen3-vl-plus"


@dataclass
class ProviderResponse:
    provider: str
    model: str
    content: str
    request_id: str | None
    usage: dict[str, Any] | None
    request_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find a reference SKU in a shelf photo with Ark, SiliconFlow, or DashScope Qwen."
    )
    parser.add_argument("--sku", required=True, help="Database SKU name supplied together with the reference image")
    parser.add_argument("--target-image", required=True, type=Path, help="Tightly cropped reference image of one target SKU")
    parser.add_argument("--shelf-image", required=True, type=Path, help="Whole shelf image to search")
    parser.add_argument("--provider", choices=("ark", "siliconflow", "dashscope", "both", "all"), default="ark")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ark model or endpoint ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--siliconflow-model", default=DEFAULT_SILICONFLOW_MODEL, help=f"SiliconFlow model (default: {DEFAULT_SILICONFLOW_MODEL})")
    parser.add_argument("--dashscope-model", default=DEFAULT_DASHSCOPE_MODEL, help=f"DashScope Qwen model (default: {DEFAULT_DASHSCOPE_MODEL})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum response tokens for each provider")
    parser.add_argument("--max-image-side", type=int, default=1280, help="Resize each request image so its longest side is at most this many pixels")
    return parser.parse_args()


def image_data_url(source: Path | bytes, max_image_side: int) -> tuple[str, dict[str, int]]:
    """Encode a bounded JPEG for lower visual-token and upload cost."""
    image_input = BytesIO(source) if isinstance(source, bytes) else source
    with Image.open(image_input) as image_source:
        source_width, source_height = image_source.size
        rgba = image_source.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, "white")
    canvas.alpha_composite(rgba)
    image = canvas.convert("RGB")
    image.thumbnail((max_image_side, max_image_side), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}", {
        "source_width": source_width,
        "source_height": source_height,
        "request_width": image.width,
        "request_height": image.height,
        "request_bytes": buffer.tell(),
    }


def build_ark_prompt(sku: str, max_boxes: int = 1) -> str:
    return f"""你将收到两张图片：
图 1 是目标商品的参考裁剪图。它与数据库 SKU 名称“{sku}”共同定义唯一目标。
图 2 是待检索的整面货架照片。

只在图 2 中找出每一个与图 1 相同包装/规格、且名称对应“{sku}”的可见商品实例。包装图像是主要依据；不要因为颜色、类别或品牌相似而包含不同 SKU。
不要标注图 1 中的商品，也不要标注货架、层板、背板、空位、阴影、价格标签或被前排遮挡的商品。
每个商品必须单独给出紧贴其可见边缘的框；相邻商品不能合成一个框，框之间不得重叠。
最多只输出 {max_boxes} 个最确定的匹配框。
坐标必须只对应图 2，使用 0 到 999 的归一化坐标。
每行只能输出一个框，格式严格为：<bbox>x1 y1 x2 y2</bbox>
找不到时严格输出：NONE
不要输出任何解释、商品名称或其他文字。"""


def build_qwen_prompt(sku: str, max_boxes: int = 1) -> str:
    return f"""你将收到两张图片：
图 1 是目标商品的参考裁剪图。它与数据库 SKU 名称“{sku}”共同定义唯一目标。
图 2 是待检索的整面货架照片。

定位图 2 中每一个与图 1 相同包装/规格、且名称对应“{sku}”的可见商品实例。包装图像是主要依据；不要因为颜色、类别或品牌相似而包含不同 SKU。
只标注图 2 中商品本体的可见前景，不标注图 1、货架、层板、背板、价格标签、空位或阴影。每个商品独立一个紧贴可见边缘的框，不要将相邻商品合成一个框。
最多只输出 {max_boxes} 个最确定的匹配框。
坐标必须只对应图 2。使用边长为 1000 的相对坐标网格，bbox_2d 的顺序为 [x_min, y_min, x_max, y_max]。
以 JSON 格式输出，且只输出 JSON 数组。每项严格形如：{{"bbox_2d":[x_min,y_min,x_max,y_max],"label":"{sku}"}}。
没有匹配项时输出 []。"""


def message_content(sku: str, target_url: str, shelf_url: str, prompt: str) -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": f"图 1：SKU“{sku}”的参考商品图。"},
        {"type": "image_url", "image_url": {"url": target_url}},
        {"type": "text", "text": "图 2：待检索货架图。"},
        {"type": "image_url", "image_url": {"url": shelf_url}},
        {"type": "text", "text": prompt},
    ]


def request_ark(content: list[dict[str, Any]], model: str, max_tokens: int) -> ProviderResponse:
    http_client = httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, trust_env=False)
    client = Ark(
        api_key=require_api_key("ark", "ARK_API_KEY"),
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
        http_client=http_client,
    )
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            thinking={"type": "disabled"},
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": content}],
        )
    finally:
        http_client.close()
    response_content = response.choices[0].message.content
    if not isinstance(response_content, str):
        raise RuntimeError("Ark returned a non-text VLM response.")
    usage = response.usage.model_dump() if getattr(response, "usage", None) is not None else None
    return ProviderResponse("ark", model, response_content, getattr(response, "id", None), usage, time.perf_counter() - started)


def generate_owlv2_prompt(sku: str, image_sources: list[Path | bytes], model: str = DEFAULT_MODEL) -> str:
    """Draft one short, reviewable English OWLv2 text query from normal SKU crops."""
    if not 1 <= len(image_sources) <= 3:
        raise ValueError("Provide one to three normal SKU crops")
    content: list[dict[str, Any]] = [
        {"type": "text", "text": (
            f"中文 SKU 名称是“{sku}”。以下 1-3 张图片是同一 SKU 的正常商品裁剪图。"
            "OWLv2 的原始定义是用自由文本对象描述作为检测查询，论文训练时使用图像关联文字的 N-gram。"
            "请以图片中可见的商品外观为主要依据，中文 SKU 名称只用于辅助消歧，不要把它当作检索词。"
            "禁止把中文名称直译、拼音化或音译成英文；例如不要输出 Xingmu。"
            "只输出一条适合 OWLv2 开放目标检测的简短英文对象描述，总长度为 2-10 个英文词；"
            "描述最有区分度的品类、包装形态、主色和图片中确实可见的品牌或口味特征。"
            "如果品牌文字不可读或没有稳定视觉特征，就使用通用英文商品描述，不要猜品牌。"
            "不要输出句子、提示模板、a photo of、数量、位置、背景、价格、促销语或推测信息。"
            "不要解释、不要 Markdown、不要中文、不要换行、不要句号。"
        )},
    ]
    for source in image_sources:
        image_url, _metadata = image_data_url(source, 560)
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    prompt = request_ark(content, model, 96).content.strip().strip("`\"'")
    if not prompt or "\n" in prompt or len(prompt) > 120 or prompt.lower().startswith("a photo of "):
        raise RuntimeError("Ark did not return one concise OWLv2 object description")
    return prompt


def request_siliconflow(content: list[dict[str, Any]], model: str, max_tokens: int) -> ProviderResponse:
    api_key = require_api_key("siliconflow", "SILICONFLOW_API_KEY")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if not model.lower().endswith("-thinking"):
        payload["enable_thinking"] = False
    started = time.perf_counter()
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, trust_env=False) as client:
        response = client.post(
            SILICONFLOW_BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
    elapsed = time.perf_counter() - started
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        detail = response.text[:600].replace("\n", " ")
        raise RuntimeError(f"SiliconFlow request failed ({response.status_code}): {detail}") from error
    body = response.json()
    try:
        response_content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"SiliconFlow response has no message content: {body}") from error
    if not isinstance(response_content, str):
        raise RuntimeError("SiliconFlow returned a non-text VLM response.")
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else None
    request_id = response.headers.get("x-siliconcloud-trace-id") or body.get("id")
    return ProviderResponse("siliconflow", model, response_content, request_id, usage, elapsed)


def request_dashscope(content: list[dict[str, Any]], model: str, max_tokens: int) -> ProviderResponse:
    api_key = require_api_key("dashscope", "DASHSCOPE_API_KEY")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "temperature": 0,
        "max_tokens": max_tokens,
        "enable_thinking": False,
    }
    started = time.perf_counter()
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, trust_env=False) as client:
        response = client.post(
            DASHSCOPE_BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
    elapsed = time.perf_counter() - started
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        detail = response.text[:600].replace("\n", " ")
        raise RuntimeError(f"DashScope request failed ({response.status_code}): {detail}") from error
    body = response.json()
    try:
        response_content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"DashScope response has no message content: {body}") from error
    if not isinstance(response_content, str):
        raise RuntimeError("DashScope returned a non-text VLM response.")
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else None
    return ProviderResponse("dashscope", model, response_content, body.get("id"), usage, elapsed)


def selected_providers(provider: str) -> tuple[str, ...]:
    if provider == "both":
        return ("ark", "siliconflow")
    if provider == "all":
        return ("ark", "siliconflow", "dashscope")
    return (provider,)


def extract_qwen_boxes(response_text: str, image_width: int, image_height: int) -> list[BoundingBox]:
    """Parse Qwen3-VL's JSON bbox_2d response on a 1000-unit relative grid."""
    candidate = response_text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        candidate = candidate.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        # Retain compatibility with a model response using the older Ark-style tags.
        return extract_boxes(response_text, image_width, image_height)

    raw_boxes: Any
    if isinstance(payload, list):
        raw_boxes = payload
    elif isinstance(payload, dict):
        raw_boxes = payload.get("boxes")
        if raw_boxes is None:
            raw_boxes = [payload] if "bbox_2d" in payload else []
    else:
        raw_boxes = []
    if isinstance(raw_boxes, dict):
        raw_boxes = [raw_boxes]
    if not isinstance(raw_boxes, list):
        return []

    boxes: list[BoundingBox] = []
    seen: set[tuple[int, int, int, int]] = set()
    for raw_box in raw_boxes:
        coordinates = raw_box.get("bbox_2d") if isinstance(raw_box, dict) else raw_box
        if not isinstance(coordinates, list) or len(coordinates) != 4:
            continue
        try:
            normalized = tuple(round(float(value)) for value in coordinates)
        except (TypeError, ValueError):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        x1, y1, x2, y2 = normalized
        pixels = (
            normalized_to_pixels(min(x1, x2), image_width),
            normalized_to_pixels(min(y1, y2), image_height),
            normalized_to_pixels(max(x1, x2), image_width),
            normalized_to_pixels(max(y1, y2), image_height),
        )
        if pixels[2] <= pixels[0] or pixels[3] <= pixels[1]:
            continue
        boxes.append(BoundingBox(index=len(boxes) + 1, label="", normalized=normalized, pixels=pixels))
    return boxes


def provider_artifacts(run_dir: Path, shelf_copy: Path, response: ProviderResponse, width: int, height: int,
                       sku: str, boxes: list[BoundingBox] | None = None) -> tuple[list[BoundingBox], dict[str, str]]:
    provider_dir = run_dir / response.provider
    provider_dir.mkdir()
    if boxes is None:
        boxes = response_boxes(response, width, height, sku)
    response_path = provider_dir / "raw_response.txt"
    boxes_path = provider_dir / "boxes.json"
    annotated_path = provider_dir / "matches.png"
    response_path.write_text(response.content, encoding="utf-8")
    boxes_path.write_text(json.dumps([asdict(box) for box in boxes], ensure_ascii=False, indent=2), encoding="utf-8")
    draw_boxes(shelf_copy, boxes, annotated_path)
    return boxes, {
        "raw_response": str(response_path.relative_to(run_dir)),
        "boxes": str(boxes_path.relative_to(run_dir)),
        "annotated_matches": str(annotated_path.relative_to(run_dir)),
    }


def response_boxes(response: ProviderResponse, width: int, height: int, sku: str) -> list[BoundingBox]:
    parsed_boxes = (
        extract_qwen_boxes(response.content, width, height)
        if response.provider in {"siliconflow", "dashscope"}
        else extract_boxes(response.content, width, height)
    )
    return [replace(box, index=index, label=sku) for index, box in enumerate(parsed_boxes, start=1)]


def select_dino_boxes(
    boxes: list[BoundingBox], scores: list[float], *, confidence_threshold: float, max_results: int
) -> list[BoundingBox]:
    """Keep only the strongest VLM candidates that DINO confirms against the reference crop."""
    ranked = sorted(
        ((box, score) for box, score in zip(boxes, scores) if score >= confidence_threshold),
        key=lambda item: item[1],
        reverse=True,
    )
    return [box for box, _score in ranked[:max_results]]


def dino_box_scores(target_path: Path, shelf_source: Path | bytes, boxes: list[BoundingBox]) -> list[float]:
    """Score only the VLM-proposed product crops; this deliberately does not run a full-image detector."""
    if not boxes:
        return []
    from vision.dino import reference_similarity_scores

    shelf_input = BytesIO(shelf_source) if isinstance(shelf_source, bytes) else shelf_source
    with Image.open(target_path) as target, Image.open(shelf_input) as shelf:
        reference = target.convert("RGB")
        shelf = shelf.convert("RGB")
        crops = [shelf.crop(box.pixels) for box in boxes]
    return reference_similarity_scores(reference, crops)


def run_vlm_sku_query(
    sku: str,
    target_path: Path,
    shelf_source: Path | bytes,
    provider: str,
    model: str,
    output_dir: Path | None = None,
    max_tokens: int = 1024,
    max_image_side: int = 1280,
    max_boxes: int = 1,
    dino_fallback: bool = False,
    dino_confidence_threshold: float = 0.72,
    debug: bool = True,
) -> dict[str, Any]:
    """Locate one SKU, producing review artifacts only when debug is enabled."""
    if provider not in {"ark", "siliconflow", "dashscope"}:
        raise ValueError("provider must be ark, siliconflow, or dashscope")
    if max_tokens < 128 or max_image_side < 256 or not 1 <= max_boxes <= 20:
        raise ValueError("Invalid VLM query limits")
    if not 0 <= dino_confidence_threshold <= 1:
        raise ValueError("dino_confidence_threshold must be between 0 and 1")
    target_path = target_path.expanduser().resolve()
    if not target_path.is_file():
        raise ValueError(f"Target image not found: {target_path}")
    if isinstance(shelf_source, bytes):
        if not shelf_source:
            raise ValueError("Shelf image is empty")
    else:
        shelf_source = shelf_source.expanduser().resolve()
        if not shelf_source.is_file():
            raise ValueError(f"Shelf image not found: {shelf_source}")
    if debug:
        if output_dir is None:
            raise ValueError("output_dir is required when debug is enabled")
        output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    target_url, target_metadata = image_data_url(target_path, max_image_side)
    shelf_url, shelf_metadata = image_data_url(shelf_source, max_image_side)
    prompt = build_ark_prompt(sku, max_boxes) if provider == "ark" else build_qwen_prompt(sku, max_boxes)
    content = message_content(sku, target_url, shelf_url, prompt)
    shelf_input = BytesIO(shelf_source) if isinstance(shelf_source, bytes) else shelf_source
    with Image.open(shelf_input) as shelf_image:
        width, height = shelf_image.size

    if provider == "ark":
        response = request_ark(content, model, max_tokens)
    elif provider == "siliconflow":
        response = request_siliconflow(content, model, max_tokens)
    else:
        response = request_dashscope(content, model, max_tokens)
    vlm_boxes = response_boxes(response, width, height, sku)
    dino_scores: list[float] = []
    boxes = vlm_boxes[:max_boxes]
    if dino_fallback:
        dino_scores = dino_box_scores(target_path, shelf_source, vlm_boxes)
        boxes = select_dino_boxes(
            vlm_boxes, dino_scores,
            confidence_threshold=dino_confidence_threshold,
            max_results=max_boxes,
        )
    if not debug:
        return {
            "sku": sku,
            "provider": response.provider,
            "model": response.model,
            "request_id": response.request_id,
            "request_seconds": response.request_seconds,
            "total_seconds": time.perf_counter() - started,
            "detected_boxes": [asdict(box) for box in boxes],
        }

    target_copy = output_dir / f"00_target_reference{target_path.suffix.lower()}"
    shelf_suffix = shelf_source.suffix.lower() if isinstance(shelf_source, Path) else ".png"
    shelf_copy = output_dir / f"01_shelf_source{shelf_suffix}"
    shutil.copy2(target_path, target_copy)
    if isinstance(shelf_source, bytes):
        shelf_copy.write_bytes(shelf_source)
    else:
        shutil.copy2(shelf_source, shelf_copy)
    boxes, artifacts = provider_artifacts(output_dir, shelf_copy, response, width, height, sku, boxes)
    report = {
        "run_id": output_dir.name,
        "sku": sku,
        "provider": response.provider,
        "model": response.model,
        "request_id": response.request_id,
        "usage": response.usage,
        "request_seconds": response.request_seconds,
        "total_seconds": time.perf_counter() - started,
        "raw_response": response.content,
        "vlm_detected_boxes": [asdict(box) for box in vlm_boxes],
        "detected_boxes": [asdict(box) for box in boxes],
        "dino_scores": [
            {"index": box.index, "confidence": round(score, 4)}
            for box, score in zip(vlm_boxes, dino_scores)
        ],
        "config": {
            "max_boxes": max_boxes,
            "dino_fallback": dino_fallback,
            "dino_confidence_threshold": dino_confidence_threshold,
        },
        "overlapping_box_pairs": count_overlapping_box_pairs(boxes),
        "request_image_sizes": {"target": target_metadata, "shelf": shelf_metadata},
        "artifacts": {
            **artifacts,
            "target_reference": target_copy.name,
            "shelf_source": shelf_copy.name,
        },
    }
    (output_dir / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    target_path = args.target_image.expanduser().resolve()
    shelf_path = args.shelf_image.expanduser().resolve()
    for label, path in (("Target image", target_path), ("Shelf image", shelf_path)):
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")
    if args.max_tokens < 128:
        raise SystemExit("--max-tokens must be at least 128")
    if args.max_image_side < 256:
        raise SystemExit("--max-image-side must be at least 256")
    if args.provider in {"siliconflow", "both", "all"} and not get_api_key("siliconflow", "SILICONFLOW_API_KEY"):
        raise SystemExit("A SiliconFlow key is required in vision/config.local.yaml or SILICONFLOW_API_KEY")
    if args.provider in {"dashscope", "all"} and not get_api_key("dashscope", "DASHSCOPE_API_KEY"):
        raise SystemExit("A DashScope key is required in vision/config.local.yaml or DASHSCOPE_API_KEY")

    created_at = datetime.now().astimezone()
    run_dir = args.output_dir.expanduser().resolve() / created_at.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    target_url, target_metadata = image_data_url(target_path, args.max_image_side)
    shelf_url, shelf_metadata = image_data_url(shelf_path, args.max_image_side)
    ark_content = message_content(args.sku, target_url, shelf_url, build_ark_prompt(args.sku))
    qwen_content = message_content(args.sku, target_url, shelf_url, build_qwen_prompt(args.sku))
    with Image.open(shelf_path) as shelf_image:
        width, height = shelf_image.size
    target_copy = run_dir / f"00_target_reference{target_path.suffix.lower()}"
    shelf_copy = run_dir / f"01_shelf_source{shelf_path.suffix.lower()}"
    shutil.copy2(target_path, target_copy)
    shutil.copy2(shelf_path, shelf_copy)

    print(f"Providers: {args.provider}")
    print(f"Target SKU: {args.sku}")
    print(f"Reference image: {target_path}")
    print(f"Shelf image: {shelf_path}")
    print(f"Request image limit: {args.max_image_side}px; response token limit: {args.max_tokens}")
    print(f"Started at: {created_at.strftime('%Y-%m-%d %H:%M:%S %z')}")

    reports: list[dict[str, Any]] = []
    for provider in selected_providers(args.provider):
        model = {
            "ark": args.model,
            "siliconflow": args.siliconflow_model,
            "dashscope": args.dashscope_model,
        }[provider]
        print(f"Requesting {provider}: {model}")
        try:
            if provider == "ark":
                response = request_ark(ark_content, model, args.max_tokens)
            elif provider == "siliconflow":
                response = request_siliconflow(qwen_content, model, args.max_tokens)
            else:
                response = request_dashscope(qwen_content, model, args.max_tokens)
        except Exception as error:
            elapsed = time.perf_counter() - started
            raise SystemExit(f"{provider} VLM query failed after {elapsed:.2f} seconds: {error}") from error
        boxes, artifacts = provider_artifacts(run_dir, shelf_copy, response, width, height, args.sku)
        reports.append({
            "provider": response.provider,
            "model": response.model,
            "request_id": response.request_id,
            "usage": response.usage,
            "request_seconds": response.request_seconds,
            "detected_boxes": [asdict(box) for box in boxes],
            "overlapping_box_pairs": count_overlapping_box_pairs(boxes),
            "artifacts": artifacts,
        })
        print(f"  {len(boxes)} boxes, {response.request_seconds:.2f}s")

    report = {
        "sku": args.sku,
        "created_at": created_at.isoformat(),
        "target_image": str(target_path),
        "shelf_image": str(shelf_path),
        "request_image_sizes": {"target": target_metadata, "shelf": shelf_metadata},
        "providers": reports,
        "total_seconds": time.perf_counter() - started,
        "input_artifacts": {"target_reference": target_copy.name, "shelf_source": shelf_copy.name},
    }
    report_path = run_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Finished at: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z')}")
    print(f"Total: {report['total_seconds']:.2f}s")
    for item in reports:
        print(f"{item['provider']}: {len(item['detected_boxes'])} boxes, {item['request_seconds']:.2f}s, {item['artifacts']['annotated_matches']}")
    print(f"Run directory: {run_dir}")
    print(f"Comparison report: {report_path}")
if __name__ == "__main__":
    main()
