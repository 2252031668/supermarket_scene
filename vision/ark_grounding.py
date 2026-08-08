#!/usr/bin/env python3
"""Run Volcengine Ark visual grounding on one shelf photograph.

The model returns <bbox>x1 y1 x2 y2</bbox> coordinates normalized to 0..999.
This script converts them back to image pixels and writes an annotated copy.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont
from volcenginesdkarkruntime import Ark

from vision.config import require_api_key

DEFAULT_MODEL = "doubao-seed-2-1-turbo-260628"
VISION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VISION_DIR.parent
DEFAULT_IMAGE = PROJECT_ROOT / "mujoco" / "screenshots" / "v5_front.png"
DEFAULT_OUTPUT_DIR = VISION_DIR / "output" / "grounding"
REQUEST_TIMEOUT_SECONDS = 300.0
BBOX_RE = re.compile(r"<bbox>\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*</bbox>")


@dataclass(frozen=True)
class BoundingBox:
    index: int
    label: str
    normalized: tuple[int, int, int, int]
    pixels: tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect separately boxed products in one shelf photo with Ark grounding."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"Input photograph (default: {DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ark model or endpoint ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--target",
        default="每一层货架最前排",
        help="The shelf region whose individual products should be detected.",
    )
    parser.add_argument(
        "--prompt",
        help="Override the grounding prompt entirely.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for response and annotated image (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def build_prompt(target: str, known_skus: list[str] | None = None) -> str:
    sku_catalog = "、".join(known_skus or [])
    sku_instruction = (
        f"已有 SKU 仅有以下这些：{sku_catalog}。包装可明确匹配时，必须原样使用其中一个 SKU。"
        if sku_catalog else "没有可用 SKU 目录，请根据包装文字给出商品名称。"
    )
    return f"""请检测图片中{target}的所有独立商品。
每个商品都必须单独返回一个边界框，不能把相邻商品合并为一个框。识别包装上可见的商品名称；看不清时写“未识别商品”。
{sku_instruction}
每一层只标注最靠近相机、可直接看见的第一排商品；不要标注排在它们后方或被它们遮挡的商品。
不要框选货架、层板、背板、地面、价格标签、阴影或空位。
每个框必须紧贴对应商品的可见边缘。任意两个框的内部不得重叠或交叉，框边缘允许刚好相接；无法避免重叠时，只保留前方商品。
每行只输出一个商品，格式为：商品名称 <bbox>x1 y1 x2 y2</bbox>
不要输出任何其他文字。"""


def encode_image(image_path: Path) -> tuple[str, str]:
    image_type, _ = mimetypes.guess_type(image_path.name)
    if image_type not in {"image/jpeg", "image/png", "image/webp"}:
        image_type = "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return image_type, encoded


def request_grounding_bytes(image_bytes: bytes, image_type: str, model: str, prompt: str) -> str:
    if image_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError("Unsupported image type for grounding")
    encoded = base64.b64encode(image_bytes).decode("ascii")
    http_client = httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, trust_env=False)
    client = Ark(
        api_key=require_api_key("ark", "ARK_API_KEY"),
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
        http_client=http_client,
    )
    try:
        response = client.chat.completions.create(
            model=model,
            # Grounding only needs coordinates. Avoid spending the request budget on
            # hidden reasoning before returning the bounding-box list.
            thinking={"type": "disabled"},
            max_tokens=2048,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{image_type};base64,{encoded}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
    finally:
        http_client.close()
    content = response.choices[0].message.content
    if not isinstance(content, str):
        raise RuntimeError("Ark returned a non-text grounding response.")
    return content


def request_grounding(image_path: Path, model: str, prompt: str) -> str:
    image_type, encoded = encode_image(image_path)
    return request_grounding_bytes(base64.b64decode(encoded), image_type, model, prompt)


def normalized_to_pixels(value: int, edge: int) -> int:
    return max(0, min(edge - 1, round(value / 1000 * edge)))


def extract_boxes(response_text: str, image_width: int, image_height: int) -> list[BoundingBox]:
    boxes: list[BoundingBox] = []
    seen: set[tuple[int, int, int, int]] = set()
    for match in BBOX_RE.finditer(response_text):
        normalized = tuple(int(value) for value in match.groups())
        if normalized in seen:
            continue
        seen.add(normalized)
        x1, y1, x2, y2 = normalized
        px1 = normalized_to_pixels(min(x1, x2), image_width)
        py1 = normalized_to_pixels(min(y1, y2), image_height)
        px2 = normalized_to_pixels(max(x1, x2), image_width)
        py2 = normalized_to_pixels(max(y1, y2), image_height)
        if px2 <= px1 or py2 <= py1:
            continue
        line_start = response_text.rfind("\n", 0, match.start()) + 1
        label = response_text[line_start : match.start()].strip(" -:：\t")
        label = re.sub(r"^(?:[-*•]\s*|\d+[.、)]\s*)", "", label).strip()
        boxes.append(
            BoundingBox(
                index=len(boxes) + 1,
                label=label or "未识别商品",
                normalized=normalized,
                pixels=(px1, py1, px2, py2),
            )
        )
    return boxes


def count_overlapping_box_pairs(boxes: list[BoundingBox]) -> int:
    overlaps = 0
    for left_index, left in enumerate(boxes):
        lx1, ly1, lx2, ly2 = left.pixels
        for right in boxes[left_index + 1 :]:
            rx1, ry1, rx2, ry2 = right.pixels
            if min(lx2, rx2) > max(lx1, rx1) and min(ly2, ry2) > max(ly1, ry1):
                overlaps += 1
    return overlaps


def draw_boxes(image_path: Path, boxes: list[BoundingBox], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 17)
    colors = ["#ef4444", "#22c55e", "#3b82f6", "#f59e0b", "#d946ef", "#06b6d4"]

    for box in boxes:
        color = colors[(box.index - 1) % len(colors)]
        x1, y1, x2, y2 = box.pixels
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        label = box.label[:18]
        label_box = draw.textbbox((x1, y1), label, font=font)
        label_width = label_box[2] - label_box[0] + 8
        label_height = label_box[3] - label_box[1] + 6
        label_y = max(0, y1 - label_height)
        draw.rectangle((x1, label_y, x1 + label_width, label_y + label_height), fill=color)
        draw.text((x1 + 4, label_y + 3), label, fill="white", font=font)

    image.save(output_path)


def main() -> None:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")

    prompt = args.prompt or build_prompt(args.target)
    print(f"Requesting grounding from model: {args.model}")
    print(f"Input image: {image_path}")
    started_at = datetime.now().astimezone()
    started_monotonic = time.perf_counter()
    print(f"Started at: {started_at.strftime('%Y-%m-%d %H:%M:%S %z')}")
    try:
        response_text = request_grounding(image_path, args.model, prompt)
    except Exception as exc:
        finished_at = datetime.now().astimezone()
        elapsed_seconds = time.perf_counter() - started_monotonic
        print(f"Finished at: {finished_at.strftime('%Y-%m-%d %H:%M:%S %z')}")
        print(f"Elapsed: {elapsed_seconds:.1f} seconds")
        request_id_match = re.search(r"request_id:\s*([^,\s]+)", str(exc))
        if request_id_match:
            print(f"Ark request ID: {request_id_match.group(1)}")
        raise SystemExit(f"Ark grounding request failed: {exc}") from exc

    finished_at = datetime.now().astimezone()
    elapsed_seconds = time.perf_counter() - started_monotonic
    print(f"Finished at: {finished_at.strftime('%Y-%m-%d %H:%M:%S %z')}")
    print(f"Elapsed: {elapsed_seconds:.1f} seconds")

    with Image.open(image_path) as image:
        width, height = image.size
    boxes = extract_boxes(response_text, width, height)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = image_path.stem
    response_path = output_dir / f"{output_stem}_grounding_response.txt"
    boxes_path = output_dir / f"{output_stem}_grounding_boxes.json"
    annotated_path = output_dir / f"{output_stem}_grounding.png"
    response_path.write_text(response_text, encoding="utf-8")
    boxes_path.write_text(
        json.dumps([asdict(box) for box in boxes], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    draw_boxes(image_path, boxes, annotated_path)

    print(f"Detected boxes: {len(boxes)}")
    print(f"Overlapping box pairs: {count_overlapping_box_pairs(boxes)}")
    print(f"Raw response: {response_path}")
    print(f"Box data: {boxes_path}")
    print(f"Annotated image: {annotated_path}")
    if not boxes:
        print("No <bbox> tags were found. Inspect the raw response and refine --target or --prompt.")


if __name__ == "__main__":
    main()
