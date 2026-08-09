# HTTP Web API

`api_server.py` 提供本地 HTTP JSON API，默认监听 `127.0.0.1:8000`。ROS2 节点需运行在同一主机，或由部署层提供受控的网络转发；所有世界坐标使用右手 `map` 坐标系，单位为米，`yaw` 单位为弧度。

请求头：

```text
Content-Type: application/json
```

图片字段统一为 Data URL：

```json
{"image_data":"data:image/jpeg;base64,/9j/..."}
```

支持 JPEG、PNG、WebP；解码后单张图片最大 12 MB。

## 接口速查

| 功能 | 请求 | 主要输入 | 成功返回 |
| --- | --- | --- | --- |
| 货架巡检 | `POST /api/vision/inspect` | `image_data`、可选 `config`、`debug` | `{"report":{"shelf_id":3,"face":0,"slots":[...]}}` |
| 货物查询 | `POST /api/sku-query` | `image_data`、`query`、`provider`、`model`、可选 `config`、`debug` | `{"report":{"sku":"...","reference_slot_id":"...","detected_boxes":[...]}}` |
| 查询 slot 坐标 | `GET /api/slots/{slot_id}/world-position` | URL 中的 `slot_id` | `{"slot":{...}}` |
| 查询 SKU 位置 | `GET /api/skus/{sku}/world-positions` | URL 中的 SKU | `{"sku":"...","positions":[...]}` |
| 查询缺货 / 错放 | `GET /api/shortages` / `GET /api/misplacements` | 无 | `{"slots":[{...}]}` |
| 取货 / 补货 | `POST /api/slots/{slot_id}/take` / `restock` | URL 中的 `slot_id` | `{"slot":{...},"state":{...}}` |
| 修正实际 SKU | `PUT /api/slots/{slot_id}` | `actual_sku` 或 `expected_sku` | `{"slot":{...},"state":{...}}` |
| 新建 / 修改货架 | `POST /api/shelves` / `PUT /api/shelves/{id}` | 名称、`world_x`、`world_y`、`yaw`、类型 | `{"id":3,"state":{...}}` / `{"state":{...}}` |
| 新建 / 修改交付桌 | `POST /api/delivery-tables` / `PUT /api/delivery-tables/{id}` | 名称、`world_x`、`world_y`、`yaw` | `{"id":1,"state":{...}}` / `{"state":{...}}` |

下文给出每个视觉请求的完整 JSON、所有返回对象字段和库存状态语义。

## 通用对象

### `Slot`

```json
{
  "slot_id": "3-0-2-43",
  "shelf_id": 3,
  "face": 0,
  "level": 2,
  "y_cm": 43.0,
  "expected_sku": "可乐",
  "actual_sku": "可乐",
  "width_cm": 8.0,
  "height_cm": 25.0,
  "image_dir": "",
  "status": "正常"
}
```

`face` 为 `0`（货架 -X 面）或 `1`（+X 面）。`status` 由 SKU 自动推导：`actual_sku` 为 `null` 是“缺货”，等于 `expected_sku` 是“正常”，其余是“摆放错误”。

### `SlotWithWorld`

`GET /api/slots/{slot_id}/world-position` 中的 `slot` 在 `Slot` 基础上增加：

```json
{
  "frame": "map",
  "world_x": 1.32,
  "world_y": 2.40,
  "world_z": 0.85
}
```

### `State`

现有写接口会附带完整 Web 快照：

```json
{
  "stats": {},
  "shelf_types": [],
  "shelves": [],
  "shelf_images": [],
  "delivery_tables": [],
  "delivery_table_spec": {},
  "skus": [],
  "slots": []
}
```

机器人可忽略其中的 `state`，以操作返回的 `slot`、`id` 或查询结果为准。

## 视觉识别

### 货架巡检

```text
POST /api/vision/inspect
```

```json
{
  "image_data": "data:image/jpeg;base64,/9j/...",
  "debug": false,
  "config": {
    "min_current_coverage": 0.05,
    "analysis_center_ratio": 0.8,
    "lab_distance_threshold": 12.0,
    "slot_change_ratio_threshold": 0.15,
    "dino_confidence_threshold": 0.72,
    "ambiguity_margin": 0.05,
    "vlm_fallback": false,
    "vlm_top_k": 4
  }
}
```

`config` 可省略，省略时使用 `vision/config.local.yaml` 或内置默认值。请求中的 `config` 只覆盖本次调用，不会写入配置文件。`debug` 默认为 `false`。

成功返回 `201 Created`：

```json
{
  "report": {
    "shelf_id": 3,
    "face": 0,
    "slots": [
      {
        "slot_id": "3-0-2-43",
        "expected_sku": "可乐",
        "actual_sku": null,
        "status": "缺货",
        "source": "shortage",
        "confidence": 0.31,
        "reason": "low_confidence",
        "bbox": {"x": 120, "y": 80, "width": 90, "height": 210}
      }
    ]
  }
}
```

`slots` 只包含异常位置。巡检不修改数据库；机器人完成物理操作后，必须调用取货、补货或修改货位接口。`bbox` 是配准后的有效重叠裁剪图的像素坐标，不是原始上传图坐标。

`debug=false` 不创建运行目录、不保存图片、不写结果 JSON 或 VLM 原文。`debug=true` 仅用于 Web 审核流程，会生成 `run_id` 与调试产物。

### 按 SKU 查询货物

```text
POST /api/sku-query
```

```json
{
  "image_data": "data:image/jpeg;base64,/9j/...",
  "query": "康师傅白桃",
  "provider": "ark",
  "model": "doubao-seed-2-1-pro-260628",
  "debug": false,
  "config": {
    "max_boxes": 1,
    "dino_fallback": false,
    "dino_confidence_threshold": 0.72
  }
}
```

`query` 必须是已有 SKU 或已有 `slot_id`。服务端从该 slot 的 `data/item_images/{slot_id}/0.png` 读取参考商品图。`provider` 只能是 `ark`、`siliconflow` 或 `dashscope`。`model` 必填。

成功返回 `201 Created`：

```json
{
  "report": {
    "query": "康师傅白桃",
    "reference_slot_id": "3-0-2-43",
    "sku": "康师傅白桃",
    "provider": "ark",
    "model": "doubao-seed-2-1-pro-260628",
    "request_id": "request-id",
    "request_seconds": 0.82,
    "total_seconds": 0.88,
    "detected_boxes": [
      {
        "index": 1,
        "label": "康师傅白桃",
        "normalized": [120, 220, 380, 800],
        "pixels": [154, 246, 486, 896]
      }
    ]
  }
}
```

`pixels` 的顺序是 `[x1, y1, x2, y2]`。`debug=false` 不创建运行目录或图片文件。

## 坐标与库存查询

| 功能 | 请求 | 成功返回 |
| --- | --- | --- |
| 健康检查 | `GET /api/health` | `{"ok":true}` |
| 查询固定位置世界坐标 | `GET /api/slots/{slot_id}/world-position` | `{"slot": SlotWithWorld}` |
| 查询当前有货 SKU 的位置 | `GET /api/skus/{sku}/world-positions` | `{"sku":"可乐","positions":[SkuWorldPosition]}` |
| 查询缺货位置 | `GET /api/shortages` | `{"slots":[Slot]}` |
| 查询错放位置 | `GET /api/misplacements` | `{"slots":[Slot]}` |
| 查询货架层位尺寸 | `GET /api/shelves/{shelf_id}/calibration` | `{"shelf_id":3,"shelf_length_cm":186.0,"levels":[{"level":0,"surface_z_cm":5.0,"opening_height_cm":38.0}]}` |

`SkuWorldPosition` 的字段为：

```json
{
  "slot_id": "3-0-2-43",
  "shelf_id": 3,
  "shelf_name": "3号货架",
  "face": 0,
  "level": 2,
  "y_cm": 43.0,
  "width_cm": 8.0,
  "height_cm": 25.0,
  "image_dir": "",
  "expected_sku": "可乐",
  "actual_sku": "可乐",
  "status": "正常",
  "world_x": 1.32,
  "world_y": 2.40,
  "world_z": 0.85
}
```

## 库存状态写入

| 功能 | 请求 | 成功返回 | 状态效果 |
| --- | --- | --- | --- |
| 取走位置商品 | `POST /api/slots/{slot_id}/take`，请求体 `{}` | `{"slot":Slot,"state":State}` | `actual_sku=null`，进入缺货查询；若原来错放，也从错放查询移除。 |
| 按应放 SKU 补货 | `POST /api/slots/{slot_id}/restock`，请求体 `{}` | `{"slot":Slot,"state":State}` | `actual_sku=expected_sku`，从缺货和错放查询移除。 |
| 标记实际 SKU 或修正 SKU | `PUT /api/slots/{slot_id}` | `{"slot":Slot,"state":State}` | 按 `expected_sku` 与 `actual_sku` 自动派生状态。 |

`PUT /api/slots/{slot_id}` 的可写字段：

```json
{
  "expected_sku": "可乐",
  "actual_sku": "农夫山泉",
  "width_cm": 8.0,
  "height_cm": 25.0,
  "image_dir": "",
  "bbox": {"x": 120, "y": 80, "width": 90, "height": 210}
}
```

`slot_id`、`shelf_id`、`face`、`level`、`y_cm` 不可修改。`bbox` 不是 SQLite 的 `Slot` 字段，只写入 `data/shelf_calibration/{shelf_id}.json` 的 CV 投影；其余字段写入数据库。每次货位写入后，服务会同步重建该 JSON。

## 场景配置写入

| 功能 | 请求体 | 成功返回 |
| --- | --- | --- |
| 新建货架：`POST /api/shelves` | `{"name":"3号货架","world_x":1.2,"world_y":2.4,"yaw":0.0,"shelf_type_id":1}` | `{"id":3,"state":State}` |
| 修改货架：`PUT /api/shelves/{shelf_id}` | 可传 `name`、`world_x`、`world_y`、`yaw`、`shelf_type_id` | `{"state":State}` |
| 新建固定位置：`POST /api/slots` | `{"shelf_id":3,"face":0,"level":2,"y_cm":43,"expected_sku":"可乐","actual_sku":"可乐","width_cm":8,"height_cm":25}` | `{"slot":Slot,"state":State}` |
| 新建交付桌：`POST /api/delivery-tables` | `{"name":"1号交付桌","world_x":2.0,"world_y":1.0,"yaw":0.0}` | `{"id":1,"state":State}` |
| 修改交付桌：`PUT /api/delivery-tables/{table_id}` | 可传 `name`、`world_x`、`world_y`、`yaw` | `{"state":State}` |

## 推荐调用顺序

补货：

```text
GET /api/shortages
GET /api/slots/{slot_id}/world-position
机器人完成补货
POST /api/slots/{slot_id}/restock
```

取货：

```text
GET /api/skus/{sku}/world-positions
机器人完成取货
POST /api/slots/{slot_id}/take
```

巡检后更新库存：

```text
POST /api/vision/inspect (debug=false)
机器人核实或完成物理处理
POST /api/slots/{slot_id}/take
POST /api/slots/{slot_id}/restock
或 PUT /api/slots/{slot_id} {"actual_sku":"..."}
```

## 错误返回

参数错误通常返回 `400`：

```json
{"error":"具体错误原因"}
```

不存在的货架、SKU、slot 或运行记录通常返回 `404`。巡检找不到匹配货架面时返回 `422`：

```json
{"error":"No matching calibrated shelf face found"}
```
