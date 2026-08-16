# 本地 Qwen3.5-4B G4 SKU 框选

本文固化当前可用的 G4 链路：输入一张机器人局部 RGB 图和 SKU 名称，读取数据库中的 `qwen_grounding_prompt`，让本地 `Qwen3.5-4B` 返回一个商品框。G4 不发送 SKU 主图、不发送红框参考图，也不使用 G5 历史示例。

## 输入与输出

输入：

```python
robot_image: bytes  # 一张局部货架 RGB 图，PNG/JPEG 均可
sku: str            # sku_catalog.sku 中存在的 SKU 名称
```

输出：

```python
{
    "sku": sku,
    "normalized_box": [x_min, y_min, x_max, y_max],  # 0..999
    "box": {"x": x, "y": y, "width": width, "height": height},  # 当前 RGB 图像素坐标
    "raw_response": response,
}
```

`normalized_box` 仅对应本次输入的机器人图。模型不能定位时为 `None`；调用方必须把它作为失败处理，不能自行猜测框。

## 数据库字段

字段位于 `sku_catalog`：

```sql
qwen_grounding_prompt TEXT NOT NULL DEFAULT ''
```

当前字段保存紧凑分号提示词，规则为：

```text
具体包装配色；品牌或关键文字；SKU 名称外的图案、色块、瓶盖、泵头或袋装特征
```

例如：

```text
紫红色包装；Lay's乐事；番茄图案
白橙渐变；Dove；白桃图案
```

G4 运行前必须读取并校验字段，不得回退为 OWLv2 英文提示词：

```python
from shelf_database import ShelfDatabase

with ShelfDatabase("shelf_inventory.db") as db:
    sku_info = db.get_sku_info(sku)

if sku_info is None:
    raise ValueError(f"Unknown SKU: {sku}")
if not sku_info.qwen_grounding_prompt.strip():
    raise ValueError(f"SKU has no qwen_grounding_prompt: {sku}")
```

## 提示词生成

提示词在手机货架局部图的红框标准示例上一次性生成，生成结果经审核后写入数据库。以下是当前实际使用的生成提示词：

```python
prompt = (
    "图 1 是手机拍摄的货架局部图。红框内的商品是唯一目标 SKU，名称为“" + sku + "”。\n"
    "输出 3 到 4 个最有区分度的可见特征：第一项必须是具体包装配色（如白橙渐变，不能只写白）；"
    "第二项是品牌或关键文字；至少一项必须是 SKU 名称以外的视觉特征，如图案、色块、瓶盖、泵头或袋装。"
    "不要复述 SKU 名称中的完整口味词。"
    "不要描述红框、货架、位置、左右关系、相邻商品、数量或背景。\n"
    "用分号连接短特征，目标总长度为 20 到 45 个中文字符，后续特征不得重复。"
    "只输出 JSON：{\"prompt\":\"主色；品牌文字；口味或图案\"}，不要 Markdown 或解释。"
)
```

生成响应必须经过以下解析，才允许写入 `qwen_grounding_prompt`：

```python
def parse_grounding_prompt(response: str) -> str | None:
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
```

实现位置：[vision/local_qwen.py](../vision/local_qwen.py)。

## G4 模型请求

`ground_sku_response()` 是实际模型调用。G4 将 `qwen_grounding_prompt` 作为 `visual_prompt` 传入，且 `reference_image` 必须为 `None`：

```python
def ground_sku_response(robot_image: bytes, sku: str, owlv2_prompt: str, model_dir: str | Path,
                        reference_image: bytes | None = None, reference_kind: str | None = None,
                        max_new_tokens: int = 96,
                        visual_prompt_label: str = "Qwen 专用中文视觉描述") -> tuple[list[int] | None, str]:
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
    response = processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return parse_grounding_box(response), response
```

本地模型目录默认是：

```text
vision/models/Qwen3.5-4B
```

## 框解析与像素换算

模型返回唯一 JSON 框。以下解析只接受一组有效的 `bbox_2d`：

```python
def parse_grounding_box(response: str) -> list[int] | None:
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
```

将归一化坐标转换为当前图像像素框：

```python
def normalized_to_pixels(box: list[int] | None, width: int, height: int) -> dict[str, int] | None:
    if box is None:
        return None
    x1, y1, x2, y2 = (
        round(value * size / 999)
        for value, size in zip(box, (width, height, width, height))
    )
    x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
    y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}
```

## 直接调用

以下调用只发送机器人局部 RGB 图、SKU 名称和数据库中的 Qwen 专用提示词：

```python
from io import BytesIO

from PIL import Image

from shelf_database import ShelfDatabase
from vision.local_qwen import ground_sku_response


def run_g4(robot_image: bytes, sku: str) -> dict:
    with ShelfDatabase("shelf_inventory.db") as db:
        sku_info = db.get_sku_info(sku)
    if sku_info is None:
        raise ValueError(f"Unknown SKU: {sku}")
    if not sku_info.qwen_grounding_prompt.strip():
        raise ValueError(f"SKU has no qwen_grounding_prompt: {sku}")
    with Image.open(BytesIO(robot_image)) as image:
        width, height = image.size
    normalized_box, raw_response = ground_sku_response(
        robot_image=robot_image,
        sku=sku_info.sku,
        owlv2_prompt=sku_info.qwen_grounding_prompt,
        model_dir="vision/models/Qwen3.5-4B",
        reference_image=None,
        max_new_tokens=96,
        visual_prompt_label="Qwen 专用中文视觉描述",
    )
    return {
        "sku": sku_info.sku,
        "normalized_box": normalized_box,
        "box": normalized_to_pixels(normalized_box, width, height),
        "raw_response": raw_response,
    }
```

## 已知结果与边界

- 构造的局部图全量验证：G4 为 `85/176`，平均最佳 IoU `0.422`；纯文本 G1 为 `81/176`，平均最佳 IoU `0.415`。
- G5 的历史会话少样本方式在本地 4B 上会复制历史坐标，11 条试点中 G5 为 `3/11`，低于 G4 的 `5/11`，不作为当前链路。
- 上述局部图由手机货架图构造，只用于提示词开发。真实机器人手部相机图必须作为独立冻结测试集，不应用于调参。
