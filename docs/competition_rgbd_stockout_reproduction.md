# 比赛 RGB-D 缺货识别：复现与迁移说明

本文描述一个独立的“比赛巡检 RGB-D 缺货识别服务”。它用于识别第一排缺货后显露的后排商品，并输出待补货 SKU、RGB 框和相对后退距离。该服务只读，不修改库存。

```text
D435i 对齐 RGB + 深度 + 内参
        -> 红色货架前梁和商品组检测
        -> 每层货架前沿三维平面与相对后退距离
        -> 缺货候选 box
        -> DINO 手机录入 SKU 图检索
        -> 不确定时 Qwen3.5-4B Top-3 裁决
        -> {sku, box, setback_mm, source}
```

## 1. 结论和边界

- 适用场景：红色前梁货架；前排缺货后，后排同 SKU 商品在 RGB 图中可见。
- 不适用场景：货架前梁不可见、层板/商品带被画面裁断、没有有效深度、后排商品完全不可见。
- SKU 推断规则：可见后排商品的 SKU 被视为前排需要补的 SKU。
- 数据库：只读取手机录入阶段已经保存的 SKU 和商品裁剪图；不读取或修改 `actual_sku`。
- `test_pic/rgbd_stockout` 只用于离线回归；在线服务直接接收相机同步帧。

## 2. 目录和依赖

项目根目录内的相关资产：

```text
supermarket_scene/
├── test_pic/rgbd_stockout/<sample_name>/
│   ├── <sample_name>_rgb.png
│   ├── <sample_name>_depth_raw.png
│   └── <sample_name>_metadata.yaml
├── shelf_inventory.db
├── data/item_images/<slot_id>/0.png
├── vision/rgbd_stockout.py
├── vision/rgbd_stockout_sku.py
├── vision/local_qwen.py
├── vision/dino.py
└── vision/models/
    ├── dinov2-small/
    └── Qwen3.5-4B/
```

Python 环境：

```bash
uv sync
```

核心 Python 依赖已在 `pyproject.toml` 中声明：`opencv-python`、`numpy`、`pyyaml`、`pillow`、`torch`、`transformers`。DINO 默认模型为 `facebook/dinov2-small`，本地 Qwen 目录为 `vision/models/Qwen3.5-4B`。

如本地没有 Qwen 权重：

```bash
uv run python vision/download_qwen35_modelscope.py
```

## 3. D435i 采集

### 3.1 相机要求

必须取得时间近似同步、且深度已经对齐到 RGB 的三路数据：

| 数据 | 作用 | 要求 |
| --- | --- | --- |
| RGB | 红色前梁、商品组、DINO/Qwen SKU | `bgr8`，建议 1280x720 |
| Depth | 毫米级前后距离 | 对齐 RGB；`16UC1` 毫米或 `32FC1` 米 |
| CameraInfo | 深度反投影 | `camera_info.k` 为有效 3x3 内参 |

项目内采集实现位于：

```text
ros2/rgbd_capture/capture_rgb_depth.py
```

该采集节点已随项目迁入。迁移时保留这个文件和其 ROS2 依赖，或实现同等输出格式的采集节点。它订阅的话题默认是：

```text
/head_camera/color/image_raw
/head_camera/depth/image_raw
/head_camera/color/camera_info
```

原始包已通过 `message_filters.ApproximateTimeSynchronizer` 使用 `queue_size=20`、`slop=0.1` 秒同步 RGB 和深度；`CameraInfo` 使用最新可用的一帧。

### 3.2 推荐 ROS2 采集命令

在已启动 D435i、且相机驱动发布上述话题后，运行一次性采集节点。输出目录必须直接落到项目样本根目录下的新子目录：

```bash
source /opt/ros/humble/setup.bash
source <camera_workspace>/install/setup.bash

python3 ros2/rgbd_capture/capture_rgb_depth.py --ros-args \
  -p output_dir:=<project_root>/test_pic/rgbd_stockout/round_001 \
  -p sample_name:=round_001 \
  -p annotation:='比赛缺货测试，正面拍摄' \
  -p max_depth_m:=2.0
```

采集节点收到第一组同步帧即退出，并执行以下深度转换：

```python
if depth_msg.encoding == '16UC1':
    depth_mm = raw_depth.astype(np.uint16)
elif depth_msg.encoding == '32FC1':
    depth_mm = np.nan_to_num(raw_depth * 1000.0, nan=0.0)
    depth_mm = np.clip(depth_mm, 0, np.iinfo(np.uint16).max).astype(np.uint16)
else:
    raise ValueError('Unsupported depth encoding')
```

不要把 `depth_raw.png` 转成伪彩色图或 8 位 PNG。检测器需要原始 16 位毫米值。

### 3.3 输出文件格式

对 `sample_name=round_001`，采集器会写入：

```text
round_001_rgb.png            # BGR PNG
round_001_depth_raw.png      # 16UC1，单位毫米，必须与 RGB 同宽高
round_001_metadata.yaml      # RGB/深度时间戳、编码和相机内参
round_001_depth_color.png    # 仅供人工查看，不参与检测
round_001.png                # RGB 与伪彩深度拼图，仅供人工查看
```

离线回归工具仅要求前三项同名前缀文件存在。metadata 至少应包含：

```yaml
camera_info:
  k: [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
depth:
  encoding: 16UC1
  unit: millimeter
```

采集前应让相机正对或轻微斜对货架，完整拍到待分析层的红色前梁和其上方商品带。底部/顶部层被裁断会被明确跳过，而不是给出毫米判断。

## 4. RGB-D 缺货算法

代码入口：`vision/rgbd_stockout.detect_stockout_candidates()`。

### 4.1 输入校验

```python
if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
    raise ValueError('RGB image must be BGR')
if depth_mm is None or depth_mm.dtype != np.uint16:
    raise ValueError('Depth image must be uint16 millimetres')
if rgb.shape[:2] != depth_mm.shape:
    raise ValueError('RGB and depth image dimensions do not match')
camera_k = np.asarray(metadata['camera_info']['k'], dtype=np.float64)
```

`camera_k` 必须为 9 个有限数，且 `fx`、`fy` 大于零。

### 4.2 货架和商品组定位

检测器源码已随项目迁入：

```text
vision/front_stockout_detector.py
```

`vision/rgbd_stockout.py` 会从同目录动态加载该文件：

```python
SOURCE = Path(__file__).with_name("front_stockout_detector.py")
```

因此迁移整个项目时不需要保留任何外部下载目录或修改绝对路径。

核心算法：

1. 在 HSV 中提取红色前梁：`H < 12 或 H > 170`、`S > 100`、`V > 70`。
2. 仅保留宽度至少为画面 `40%`、面积至少为 `width * 8` 的红色连通区域。
3. 每条前梁用 500 次 RANSAC 拟合上沿和下沿直线；斜拍、画面滚转仍可使用。
4. 商品分析带是前梁上方 `round(image_height * 0.27)` 像素。顶端/底端被裁断的层写入 `skipped_shelves`。
5. 商品像素掩码使用颜色、亮度、纹理边缘和有效深度；闭运算核为 `max(5, image_width // 150)` 的奇数方核。横向占用率形成连续商品组。
6. 将红色前梁上的有效深度点按相机内参反投影到三维空间，并 RANSAC 拟合每层独立的前沿平面。
7. 每个商品组取相对平面的第 10 百分位距离作为“最靠外商品”距离。
8. 对同一层所有组的距离排序，取最靠前一半商品的中位数作为动态基准：

```python
front_count = max(1, (len(raw_setbacks) + 1) // 2)
baseline = float(np.median(raw_setbacks[:front_count]))
relative_setback = closest_item_setback_mm - baseline
status = 'stockout_candidate' if relative_setback >= threshold_mm else 'normal'
```

所以 `setback_mm` 是相对同层前排基准的后退值，不是相机到商品的绝对距离。

## 5. SKU 识别和 Qwen 复核

实现位于 `vision/rgbd_stockout_sku.py`。

### 5.1 DINO 排名

先用 RGB-D 算法输出的框裁取 RGB 商品区域，再与每个 SKU 的手机录入参考图比较。DINO 模型使用 `vision.dino.reference_similarity_scores()`，会对商品图补白成方形后计算余弦相似度。

```python
crop = rgb[y:y + height, x:x + width]
candidate = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
scores = reference_similarity_scores(candidate, reference_images)
ranked = sorted(zip(skus, scores), key=lambda item: (-item[1], item[0]))
```

直接接受 DINO 第一名的条件：

```python
top_score >= 0.80
and (只有一个候选 or top_score - second_score >= 0.05)
```

### 5.2 Qwen Top-3 裁决

DINO 未达到上述条件时，不把商品 crop 单独发给 Qwen。每个红框单独构造一次请求：

```text
图一：完整原始 RGB 图，只画当前候选框的红框和 ? 标记
图二：DINO Top-3 的手机录入参考图拼板，标记 1. SKU 名称、2. SKU 名称、3. SKU 名称
```

候选板不显示 DINO 分数。Qwen 提示要求只返回：

```json
{"choice": 1}
```

或 `unknown`。服务只接受范围 `1..Top-K` 的数字（也兼容模型输出的数字字符串），不能由模型编造 SKU。

## 6. 数据库与手机录入参考图

SKU 识别依赖两类本地数据，二者缺一不可：

```text
shelf_inventory.db
  shelf_inventory.expected_sku
      -> SKU 名称

data/item_images/<slot_id>/0.png
  slot_id 对应的一张手机录入商品裁剪图
      -> DINO 参考图
```

数据库相关表：

```sql
CREATE TABLE sku_catalog (
  sku TEXT PRIMARY KEY,
  ...
);

CREATE TABLE shelf_inventory (
  slot_id TEXT PRIMARY KEY,
  expected_sku TEXT NOT NULL REFERENCES sku_catalog(sku),
  actual_sku TEXT NULL REFERENCES sku_catalog(sku),
  image_dir TEXT NOT NULL,
  ...
);
```

当前代码按 `slot_id` 排序，针对每个 `expected_sku` 选择第一张存在的 `0.png`，作为该 SKU 的 DINO 参考图：

```python
with ShelfDatabase(db_path) as db:
    slots = sorted(db.get_all_slots(), key=lambda slot: slot.slot_id)

references = {}
for slot in slots:
    path = Path(item_images_dir) / slot.slot_id / '0.png'
    if slot.expected_sku not in references and path.is_file():
        references[slot.expected_sku] = path
```

因此迁移/复现时必须完成手机录入或等价数据导入：

1. 在 `sku_catalog` 中创建 SKU。
2. 在 `shelf_inventory` 中创建每个商品位置，并填 `expected_sku`。
3. 在 `data/item_images/<slot_id>/0.png` 放入该位置的清晰正常商品裁剪图。

快速检查参考图是否可用：

```bash
uv run python - <<'PY'
from pathlib import Path
from shelf_database import ShelfDatabase

with ShelfDatabase('shelf_inventory.db') as db:
    for slot in sorted(db.get_all_slots(), key=lambda item: item.slot_id):
        image = Path('data/item_images') / slot.slot_id / '0.png'
        print(slot.slot_id, slot.expected_sku, image.is_file())
PY
```

`actual_sku`、货位状态、缺货数量均不参与这条比赛缺货识别链路；调用不会更新 SQLite。

## 7. 参数

默认值在 `vision/config.py`，本地覆盖写在不提交的 `vision/config.local.yaml`：

```yaml
rgbd_stockout:
  setback_threshold_mm: 60.0
  dino_accept_score: 0.80
  dino_accept_margin: 0.05
  dino_top_k: 3
  qwen_model_dir: vision/models/Qwen3.5-4B
  qwen_max_new_tokens: 48
```

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `setback_threshold_mm` | 60.0 | 商品相对本层动态前沿基准后退多少毫米才是缺货候选 |
| `dino_accept_score` | 0.80 | DINO 直接确认 SKU 的最低相似度 |
| `dino_accept_margin` | 0.05 | 第一名相对第二名的最低领先值 |
| `dino_top_k` | 3 | Qwen 可选择的 SKU 数量 |
| `qwen_model_dir` | `vision/models/Qwen3.5-4B` | 本地 Qwen 权重目录 |
| `qwen_max_new_tokens` | 48 | Qwen SKU 选择输出的 token 上限 |

调参顺序：先确认红框是否正确，再调整 `setback_threshold_mm`；SKU 错误先检查手机录入参考图，再调整 DINO 阈值。不要用降低阈值修复错误红框。

## 8. 本地 Python 复现

下面调用使用项目内 `正面图` 样本，并保存调试产物：

```bash
uv run python - <<'PY'
from pathlib import Path
from vision.config import get_rgbd_stockout_config
from vision.rgbd_stockout_sku import run_rgbd_stockout

root = Path.cwd()
report = run_rgbd_stockout(
    root / 'test_pic/rgbd_stockout/正面图',
    root / 'shelf_inventory.db',
    root / 'data/item_images',
    get_rgbd_stockout_config(),
    output_dir=root / 'vision/output/rgbd_stockout/manual_check',
    debug=True,
)
print(report)
PY
```

典型单候选结果：

```json
{
  "sku": "上好佳虾片",
  "box": {"x": 108, "y": 256, "width": 191, "height": 139},
  "setback_mm": 100.4,
  "source": "qwen"
}
```

`debug=True` 会保存：

```text
vision/output/rgbd_stockout/<run_id>/
├── result.json
├── result_overlay.png
└── candidate_<shelf>_<group>/       # 仅 DINO 不确定时存在
    ├── full_image.png               # Qwen 图一
    ├── candidate_board.png          # Qwen 图二
    └── raw_response.txt
```

## 9. 独立服务封装

独立服务的输入不是文件路径，而是一组刚采集的同步数据：

```text
rgb:      OpenCV BGR ndarray
depth_mm: OpenCV uint16 ndarray，单位毫米，已对齐 RGB
metadata: {"camera_info": {"k": [fx, 0, cx, 0, fy, cy, 0, 0, 1]}}
```

相机适配层收到服务请求后，应等待一组请求之后到达的 RGB、深度、`CameraInfo`，再调用识别。不要把上一次缓存的旧帧作为比赛结果。

服务核心调用顺序如下；这段代码不保存文件，也不修改数据库：

```python
from vision.config import get_rgbd_stockout_config
from vision.rgbd_stockout import detect_stockout_candidates
from vision.rgbd_stockout_sku import resolve_candidate_sku, sku_references

def inspect_competition_stockout(rgb, depth_mm, metadata, db_path, item_images_dir):
    config = get_rgbd_stockout_config()
    detection = detect_stockout_candidates(
        rgb, depth_mm, metadata,
        threshold_mm=float(config['setback_threshold_mm']),
    )
    references = sku_references(db_path, item_images_dir)
    candidates = []
    for candidate in detection.candidates:
        decision = resolve_candidate_sku(rgb, candidate.box, references, config)
        candidates.append({
            'sku': decision['sku'],
            'box': candidate.box,
            'setback_mm': candidate.setback_mm,
            'source': decision['source'],
        })
    return {'candidates': candidates, 'skipped_shelves': detection.skipped_shelves}
```

建议的服务响应结构：

```json
{
  "candidates": [
    {
      "sku": "上好佳虾片",
      "box": {"x": 108, "y": 256, "width": 191, "height": 139},
      "setback_mm": 100.4,
      "source": "qwen"
    }
  ],
  "skipped_shelves": []
}
```

`sku: null` 表示后排商品框已被深度算法确认，但 SKU 无法可靠识别。服务应返回这个事实，不应猜测或省略该框。

## 10. 迁移清单

复制到新机器前逐项确认：

```text
[ ] 项目根目录和 Python 依赖可运行：uv sync
[ ] test_pic/rgbd_stockout/<name>/ 有 RGB、16UC1 深度、metadata 三文件（离线回归）
[ ] shelf_inventory.db 中有 sku_catalog 和 shelf_inventory 数据
[ ] data/item_images/<slot_id>/0.png 与 shelf_inventory.slot_id 对应
[ ] vision/models/dinov2-small/ 已存在，或允许首次下载 DINO
[ ] vision/models/Qwen3.5-4B/ 已存在，或执行下载脚本
[ ] vision/front_stockout_detector.py 与 vision/rgbd_stockout.py 均已随项目复制
[ ] 如需新采样，ROS2 D435i 采集器能发布对齐 RGB、深度、CameraInfo
[ ] uv run python -m unittest tests.test_rgbd_stockout
```

采集器迁移所需 ROS2 依赖：`rclpy`、`sensor_msgs`、`cv_bridge`、`message_filters`、`numpy`、`opencv-python`、`pyyaml`。RealSense 驱动侧通常还需要 `realsense2_camera`。

## 11. 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| 离线回归找不到样本 | 检查目录是否为 `test_pic/rgbd_stockout/<name>/`，且三文件同前缀 |
| `Depth image must be uint16 millimetres` | 传入了伪彩深度或 8 位图片；重新保存原始 `16UC1` 深度 PNG |
| `Camera metadata must contain...` | YAML 缺少 `camera_info.k` 或 `fx/fy` 无效 |
| 没有候选但肉眼有缺货 | 红梁/商品带不完整、后排商品不可见、或后退距离未达到阈值；先查看 `result_overlay.png` 和 `skipped_shelves` |
| box 正确但 SKU 为 `null` | 手机录入参考图缺失、DINO 无候选，或 Qwen 选择 `unknown`；检查第 6 节 |
| 迁移后报 detector source missing | 检查 `vision/front_stockout_detector.py` 是否和 `vision/rgbd_stockout.py` 一起复制；无需配置外部绝对路径 |
| 显存占用高 | DINO 在第一次检索时加载；Qwen 仅在 DINO 不确定时加载。停止独立识别服务进程会释放模型显存。 |
