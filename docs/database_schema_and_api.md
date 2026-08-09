# 货架数据库：表结构与 API 清单

---

## 一、数据库概述

- **数据库类型**: SQLite
- **数据库文件**: `shelf_inventory.db`
- **编码**: `shelf_database.py` — `ShelfDatabase` 类

### 坐标契约 (`map`)

数据库中的世界坐标遵循 ROS `map` 帧的右手系约定，单位为米：`+X`、`+Y` 位于地面平面，`+Z` 向上，且 `+X × +Y = +Z`。`yaw` 是绕 `+Z` 轴的逆时针旋转（弧度）。

`shelf_groups.world_x` / `world_y` 与 `delivery_tables.world_x` / `world_y` 都存储各自**局部原点**在 `map` 中的坐标，不使用屏幕方位描述。在 `yaw = 0` 时，该原点是实体占地范围的最小局部 X、最小局部 Y 参考角，实体沿本地 `+X` 和 `+Y` 延展。现有数值可直接用于 ROS2、MuJoCo 与 Three.js 的 Z-up 场景，无需转换。

### 货架物理参数 (单位: 米, 存储在 `shelf_types` 表中)

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `shelf_length` | 1.86 | 货架组长边 (沿Y) |
| `shelf_width` | 0.80 | 货架组宽边 (沿X) |
| `shelf_height` | 1.65 | 货架组高度 (沿Z) |
| `num_levels` | 5 | 层数 (0-4) |
| `bottom_clearance` | 0.05 | 底层离地高度 |
| `level_spacing` | 0.40 | 层间距 |
| `panel_thick` | 0.02 | 层板厚度 |
| `back_thick` | 0.005 | 背板厚度 |
| `shelf_depth_normal` | 0.30 | 上层层板深度 (每面) |
| `shelf_depth_bottom` | 0.40 | 底层深度 (每面) |

> **设计说明**: 以上 10 个参数全部存储在 `shelf_types` 表中。代码中的 `DEFAULT_*` 常量仅作为 fallback 值。不同货架类型可拥有不同的尺寸配置。

### 货架局部坐标系

```
原点: 货架局部原点 (reference anchor)
+X: 货架宽度方向
+Y: 货架长度方向
+Z: 货架高度方向

face = 0: -X 侧
face = 1: +X 侧

level: 0 到 (num_levels-1), 从下到上
```

### 世界坐标转换

```
wx = world_x + lx·cos(yaw) - ly·sin(yaw)
wy = world_y + lx·sin(yaw) + ly·cos(yaw)
wz = lz

yaw = 0 时, 货架局部 +X/+Y 与世界 +X/+Y 对齐
```

### Slot ID 字符串格式

```
{shelf_id}-{face}-{level}-{y_cm}

一个 slot = 货架上的固定位置，`slot_id` 在首次录入时生成并持久化，后续状态变化不会改变它。

例如: 1-1-2-9
  1  — 1号货架
  1  — +X 侧
  2  — 第2层 (从下到上)
  9  — 商品中心距货架原点Y轴 9cm (沿货架长度方向)
```

---

## 二、五张表结构

### 2.1 `shelf_types` — 货架类型表

记录不同货架的物理尺寸参数。每个 `shelf_group` 通过 `shelf_type_id` 关联一种类型，实现同一场景中不同尺寸货架的共存。

| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | 自增主键 |
| `name` | TEXT | NOT NULL UNIQUE | 类型名 (如 `"standard"`) |
| `shelf_length` | REAL | NOT NULL | 货架组长边长度 (沿Y, 米) |
| `shelf_width` | REAL | NOT NULL | 货架组宽边长度 (沿X, 米) |
| `shelf_height` | REAL | NOT NULL | 货架组高度 (沿Z, 米) |
| `num_levels` | INTEGER | NOT NULL | 层数 |
| `bottom_clearance` | REAL | NOT NULL | 底层离地高度 (米) |
| `level_spacing` | REAL | NOT NULL | 层间距 (米) |
| `panel_thick` | REAL | NOT NULL | 层板厚度 (米) |
| `back_thick` | REAL | NOT NULL | 背板厚度 (米) |
| `shelf_depth_normal` | REAL | NOT NULL | 上层层板深度, 每面 (米) |
| `shelf_depth_bottom` | REAL | NOT NULL | 底层深度, 每面 (米) |

```sql
CREATE TABLE IF NOT EXISTS shelf_types (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT NOT NULL UNIQUE,
    shelf_length        REAL NOT NULL,
    shelf_width         REAL NOT NULL,
    shelf_height        REAL NOT NULL,
    num_levels          INTEGER NOT NULL,
    bottom_clearance    REAL NOT NULL,
    level_spacing       REAL NOT NULL,
    panel_thick         REAL NOT NULL,
    back_thick          REAL NOT NULL,
    shelf_depth_normal  REAL NOT NULL,
    shelf_depth_bottom  REAL NOT NULL
);
```

---

### 2.2 `shelf_groups` — 货架组表

记录每个货架组在世界中的位置信息。通过 `shelf_type_id` 关联物理参数。

| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `id` | INTEGER | PRIMARY KEY | 自增主键, 货架组唯一标识 |
| `name` | TEXT | NOT NULL DEFAULT '' | 货架名称 |
| `world_x` | REAL | NOT NULL DEFAULT 0.0 | 货架局部原点的 map X 坐标 |
| `world_y` | REAL | NOT NULL DEFAULT 0.0 | 货架局部原点的 map Y 坐标 |
| `yaw` | REAL | NOT NULL DEFAULT 0.0 | 绕 map +Z 的逆时针旋转 (弧度) |
| `shelf_type_id` | INTEGER | FK→shelf_types.id, DEFAULT NULL | 货架类型 |
| `created_at` | TEXT | NOT NULL DEFAULT (datetime('now')) | 创建时间 |

**约束**:
- `FOREIGN KEY (shelf_type_id) REFERENCES shelf_types(id) ON DELETE SET NULL` — 删除类型时不级联删除货架组

```sql
CREATE TABLE shelf_groups (
    id              INTEGER PRIMARY KEY,
    name            TEXT NOT NULL DEFAULT '',
    world_x         REAL NOT NULL DEFAULT 0.0,
    world_y         REAL NOT NULL DEFAULT 0.0,
    yaw             REAL NOT NULL DEFAULT 0.0,
    shelf_type_id   INTEGER DEFAULT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (shelf_type_id) REFERENCES shelf_types(id) ON DELETE SET NULL
);
```

---

### 2.3 `delivery_tables` — 交付桌表

交付桌是用于承载交付清单和后续交付流程的独立场景实体，**不属于货架，也不拥有库存货位**。当前物理规格固定为局部 `+X` 长 `1.20m`、局部 `+Y` 宽 `0.80m`、桌面中心高 `0.75m`，由 `scene_geometry.py` 统一定义。

| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | 交付桌唯一标识 |
| `name` | TEXT | NOT NULL DEFAULT '' | 交付桌名称 |
| `world_x` | REAL | NOT NULL DEFAULT 0.0 | 局部原点的 map X 坐标 |
| `world_y` | REAL | NOT NULL DEFAULT 0.0 | 局部原点的 map Y 坐标 |
| `yaw` | REAL | NOT NULL DEFAULT 0.0 | 绕 map +Z 逆时针旋转，单位弧度 |
| `created_at` | TEXT | NOT NULL | 创建时间 |

局部原点位于桌体在 `yaw=0` 时的最小局部 X、最小局部 Y 地面参考角；`+X` 沿桌长，`+Y` 沿桌宽，`+Z` 向上。

```sql
CREATE TABLE delivery_tables (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL DEFAULT '',
    world_x         REAL NOT NULL DEFAULT 0.0,
    world_y         REAL NOT NULL DEFAULT 0.0,
    yaw             REAL NOT NULL DEFAULT 0.0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
```

### 2.4 `sku_catalog` — SKU 目录表

记录所有可放置的商品类型及其 3D 模型资源路径。

| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `sku` | TEXT | PRIMARY KEY | 商品 SKU 唯一标识 (如 `cracker_box`) |
| `category` | TEXT | NOT NULL DEFAULT '' | 品类 (food/drink/cleaning/kitchen) |
| `mesh_file` | TEXT | NOT NULL DEFAULT '' | OBJ 网格文件路径 |
| `tex_file` | TEXT | NOT NULL DEFAULT '' | 纹理贴图路径 |

```sql
CREATE TABLE sku_catalog (
    sku         TEXT PRIMARY KEY,
    category    TEXT NOT NULL DEFAULT '',
    mesh_file   TEXT NOT NULL DEFAULT '',
    tex_file    TEXT NOT NULL DEFAULT ''
);
```

---

### 2.5 `shelf_inventory` — 货架库存表 (核心)

**一个 slot = 货架上的固定位置**。`expected_sku` 表示该位置应摆商品，`actual_sku` 表示当前实际商品；状态不入库，由两者实时派生。位置字段创建后不可修改，移动货位必须删除旧位置并新建。

| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `slot_id` | TEXT | PRIMARY KEY | 稳定位置 ID，格式为 `{shelf_id}-{face}-{level}-{y_cm}` |
| `shelf_id` | INTEGER | NOT NULL, FK→shelf_groups.id | 所属货架组 |
| `face` | INTEGER | NOT NULL, CHECK(0,1) | 面: 0=-X 侧, 1=+X 侧 |
| `level` | INTEGER | NOT NULL, CHECK(>=0) | 层号, 0=最下层 (上限由 shelf_types.num_levels 决定) |
| `y_cm` | REAL | NOT NULL, CHECK(>=0) | 商品中心距货架原点Y轴距离 (厘米, 沿货架长度方向) |
| `expected_sku` | TEXT | NOT NULL, FK→sku_catalog.sku | 该位置应摆 SKU |
| `actual_sku` | TEXT | NULL, FK→sku_catalog.sku | 当前实际 SKU；NULL 表示缺货 |
| `width_cm` | REAL | NULL | 商品沿货架局部 Y 的宽度 (厘米) |
| `height_cm` | REAL | NULL | 商品沿货架局部 Z 的高度 (厘米) |
| `image_dir` | TEXT | NOT NULL DEFAULT '' | 实例图片相对目录，如 `data/item_images/3-0-4-34` |

**约束**:
- `FOREIGN KEY (shelf_id) REFERENCES shelf_groups(id) ON DELETE CASCADE` — 删除货架时级联删除库存
- `expected_sku`、`actual_sku` 均使用 `ON DELETE RESTRICT`，被货位引用的 SKU 不允许删除
- `UNIQUE(shelf_id, face, level, y_cm)` — 同一位置只能有一个商品

**索引**:
- `idx_inventory_shelf` ON `(shelf_id, face, level, y_cm)` — 按位置查询
- `idx_inventory_expected_sku`、`idx_inventory_actual_sku` — 分别查询计划位置和当前商品

```sql
CREATE TABLE shelf_inventory (
    slot_id         TEXT PRIMARY KEY,
    shelf_id        INTEGER NOT NULL,
    face            INTEGER NOT NULL CHECK(face IN (0, 1)),
    level           INTEGER NOT NULL CHECK(level >= 0),
    y_cm            REAL NOT NULL CHECK(y_cm >= 0),
    expected_sku    TEXT NOT NULL,
    actual_sku      TEXT DEFAULT NULL,
    width_cm        REAL DEFAULT NULL,
    height_cm       REAL DEFAULT NULL,
    image_dir       TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (shelf_id) REFERENCES shelf_groups(id) ON DELETE CASCADE,
    FOREIGN KEY (expected_sku) REFERENCES sku_catalog(sku) ON DELETE RESTRICT,
    FOREIGN KEY (actual_sku) REFERENCES sku_catalog(sku) ON DELETE RESTRICT,
    UNIQUE(shelf_id, face, level, y_cm)
);

CREATE INDEX idx_inventory_shelf ON shelf_inventory(shelf_id, face, level, y_cm);
CREATE INDEX idx_inventory_expected_sku ON shelf_inventory(expected_sku);
CREATE INDEX idx_inventory_actual_sku ON shelf_inventory(actual_sku);
```

派生状态规则：`actual_sku IS NULL` 为“缺货”；`actual_sku = expected_sku` 为“正常”；其余为“摆放错误”。

### 2.6 校准 JSON v2

SQLite 是唯一可写业务真源。`data/shelf_calibration/{shelf_id}.json` 是供 CV 快速读取的派生缓存，使用 `schema_version: 2` 和 `faces[*].slots`，每个 slot 保存 `slot_id`、`expected_sku`、`actual_sku` 与可选 `bbox`。API 在数据库提交后按货架原子重建该投影，同时保留 `image_file`、`image_hash`、`layers` 和已有 `bbox`。

---

## 三、HTTP JSON API

`api_server.py` 默认监听 `127.0.0.1:8000`，请求和响应均为 UTF-8 JSON。URL 中的 `slot_id`、SKU 和 `run_id` 应进行 URL 编码。

### 3.1 通用行为

- `POST /api/vision/inspect` 和 `POST /api/sku-query` 的顶层 `debug` 默认为 `false`。
- `debug=false` 直接返回生产结果，不创建运行目录、不绘制图片、不写结果 JSON 或 VLM 原文。
- Web 显式传递 `debug=true`，才会生成 `run_id` 和 `vision/output/` 下的调试产物。
- 视觉请求的 `image_data` 是 `data:image/jpeg;base64,...`、PNG 或 WebP Data URL，单张图片最大 12 MB。
- 数据库写入成功后同步更新 `data/shelf_calibration/{shelf_id}.json`；该 JSON 是 CV 投影，不是独立业务真源。
- HTTP 坐标接口返回 `world_x`、`world_y`、`world_z`，单位为米，坐标系为 `map`。HTTP 单 slot 坐标接口当前不返回 `shelf_yaw`；ROS2 直连服务会额外返回 `shelf_yaw` 和 `face`。

### 3.2 路由清单

#### GET

| 路径 | 返回 | 说明 |
|------|------|------|
| `/api/health` | `{"ok": true}` | 服务健康检查 |
| `/api/state` | 完整快照 | Web 使用，包含统计、货架、SKU、交付桌和所有 slot 世界坐标 |
| `/api/vision/config` | 巡检配置对象 | 返回阈值和 Ark 保底配置，不返回密钥 |
| `/api/sku-query/config` | SKU 查询配置对象 | 返回 `max_boxes`、DINO 保底和阈值 |
| `/api/shortages` | `{"slots": [...]}` | `actual_sku IS NULL` 的位置，不含世界坐标 |
| `/api/misplacements` | `{"slots": [...]}` | 实际 SKU 与预期 SKU 不同的位置，不含世界坐标 |
| `/api/skus/{sku}/world-positions` | `{"sku", "positions"}` | 只返回 `actual_sku` 等于该 SKU 的有货位置 |
| `/api/slots/{slot_id}/world-position` | `{"slot": {...}}` | 返回一个 slot 的库存字段和世界坐标 |
| `/api/shelves/{shelf_id}/calibration` | 货架层位标定 | 返回货架长度、层板表面高度和开口高度 |
| `/api/item-images/{slot_id}/0.png` | PNG 文件 | 读取固定位置参考商品裁剪图 |
| `/api/shelf-images/{shelf_id}/{face}/0.png` | PNG 文件 | 读取货架面原图 |
| `/api/vision/runs/{run_id}/result` | 巡检 `result.json` | 只用于 debug 巡检运行 |
| `/api/vision/runs/{run_id}/artifact/{name}` | JSON 或 PNG | 读取巡检报告登记的调试产物 |
| `/api/sku-query/runs/{run_id}/artifact/{name}` | JSON、文本或图片 | 读取 debug SKU 查询产物 |
| `/api/image-stitch/runs/{run_id}/artifact/{name}` | PNG | 读取图片拼接产物 |

#### POST

| 路径 | 请求参数 | 返回 |
|------|------|------|
| `/api/shelves` | `name`、`world_x`、`world_y`、`yaw`、`shelf_type_id` 可选 | `{"id": shelf_id, "state": State}` |
| `/api/delivery-tables` | `name`、`world_x`、`world_y`、`yaw` | `{"id": table_id, "state": State}` |
| `/api/skus` | `sku`，可选 `category`、`mesh_file`、`tex_file` | `{"state": State}` |
| `/api/slots` | `shelf_id`、`face`、`level`、`y_cm`、`expected_sku`；可选 `actual_sku`、尺寸、`bbox`、`image_dir` | `{"slot": Slot, "state": State}` |
| `/api/slots/{slot_id}/take` | 空对象 `{}` | `{"slot": Slot, "state": State}`；实际 SKU 置空 |
| `/api/slots/{slot_id}/restock` | 空对象 `{}` | `{"slot": Slot, "state": State}`；实际 SKU 恢复为预期 SKU |
| `/api/imports/manual` | `items`、`new_skus`、`layers`、`shelf_image` | `{"slot_ids": [...], "state": State}` |
| `/api/grounding/products` | `image_data` | `{"boxes": [...], "detected": number}` |
| `/api/vision/inspect` | `image_data`、可选 `config`、`debug` | `{"report": InspectionReport}`，状态不写数据库 |
| `/api/sku-query` | `image_data`、`query`、`provider`、`model`、可选 `config`、`debug` | `{"report": SkuQueryReport}` |
| `/api/image-stitch` | `images`（2 至 8 张）、可选 `main_index` | `{"report": ImageStitchReport}` |
| `/api/image-stitch/runs/{run_id}/rectify` | `points` 四点数组 | `{"report": ImageStitchReport}` |
| `/api/vision/runs/{run_id}/apply` | `slot_ids` 非空字符串数组 | `{"slots": [...], "state": State}` |

#### PUT

| 路径 | 请求参数 | 返回 |
|------|------|------|
| `/api/vision/config` | 巡检阈值和 `vlm_fallback` 等配置 | `{"inspection": config}` |
| `/api/sku-query/config` | `max_boxes`、`dino_fallback`、`dino_confidence_threshold` | `{"sku_query": config}` |
| `/api/shelf-types/{id}` | 10 个货架物理参数 | `{"state": State}` |
| `/api/shelves/{id}` | 可选 `name`、`world_x`、`world_y`、`yaw`、`shelf_type_id` | `{"state": State}` |
| `/api/delivery-tables/{id}` | 可选 `name`、`world_x`、`world_y`、`yaw` | `{"state": State}` |
| `/api/slots/{slot_id}` | 可选 `expected_sku`、`actual_sku`、尺寸、`bbox`、`image_dir` | `{"slot": Slot, "state": State}` |

#### DELETE

| 路径 | 请求参数 | 返回 |
|------|------|------|
| `/api/shelves/{id}/inventory` | `scope=level/face/all`；按范围附带 `face`、`level` | `{"removed": number, "state": State}` |
| `/api/delivery-tables/{id}` | 空对象 `{}` | `{"state": State}` |
| `/api/shelves/{id}` | 空对象 `{}` | `{"removed": number, "state": State}` |
| `/api/slots/{slot_id}` | 空对象 `{}` | `{"deleted": Slot, "state": State}` |
| `/api/skus/{sku}` | 空对象 `{}` | `{"removed": sku, "state": State}` |

`State` 的完整字段为 `stats`、`shelf_types`、`shelves`、`shelf_images`、`delivery_tables`、`delivery_table_spec`、`skus`、`slots`。机器人生产调用不需要解析 `State`，应使用 [机器人 Web API](robot_web_api.md) 中的精简调用说明。

### 3.3 视觉配置与结果

巡检默认配置：

```yaml
inspection:
  min_current_coverage: 0.05
  analysis_center_ratio: 0.8
  lab_distance_threshold: 12.0
  slot_change_ratio_threshold: 0.15
  dino_confidence_threshold: 0.72
  ambiguity_margin: 0.05
  vlm_fallback: false
  vlm_top_k: 4
```

SKU 查询默认配置：

```yaml
sku_query:
  max_boxes: 1
  dino_fallback: false
  dino_confidence_threshold: 0.72
```

未开启 Ark 保底时，DINO 低于置信度阈值会返回 `actual_sku=null`；开启后交给 Ark 判断，Ark 无法判断时同样按缺货处理。巡检结果只有在调用 `/api/vision/runs/{run_id}/apply` 后才会更新数据库；`debug=false` 没有运行记录，不能调用 `apply`。

### 3.4 货架面原图文件

人工批量导入确认时，会将上传的整面照片转换为 PNG 并写入受控目录，不增加独立数据表：

```text
data/shelf_images/{shelf_id}/-x_0.png
data/shelf_images/{shelf_id}/+x_0.png
```

文件名中的 `-x` / `+x` 表示货架局部坐标面，分别对应 `face=0` / `face=1`；`0` 是当前唯一的面图。对同一货架面再次人工导入会替换该文件。`GET /api/state` 的 `shelf_images` 数组仅列出实际存在的面图，`GET /api/shelf-images/{shelf_id}/{face}/0.png` 可读取其 PNG。删除整个货架会一并清理该货架的图片目录。

## 四、ShelfDatabase Python API 总览

| 分类 | 方法数 | 涉及方法 |
|------|--------|----------|
| 初始化 | 1 | `ShelfDatabase(db_path)` |
| 货架类型管理 | 5 | `add_shelf_type`, `update_shelf_type`, `get_shelf_type`, `get_shelf_type_by_name`, `get_all_shelf_types` |
| 货架组管理 | 6 | `add_shelf_group`, `update_shelf_group`, `remove_shelf_group`, `get_shelf_group`, `get_all_shelf_groups`, `get_shelf_world_pos` |
| 交付桌管理 | 5 | `add_delivery_table`, `update_delivery_table`, `remove_delivery_table`, `get_delivery_table`, `get_all_delivery_tables` |
| SKU 目录管理 | 5 | `register_sku`, `register_skus_batch`, `get_sku_info`, `get_all_skus`, `remove_sku_from_catalog` |
| 货位写入 | 11 | `create_slot`, `update_slot`, `set_actual_sku`, `set_actual_sku_batch`, `take_slot`, `restock_slot`, `import_slots_batch`, `delete_slot`, `clear_shelf`, `clear_shelf_face`, `clear_shelf_face_level` |
| 货位查询 | 12 | `get_slot_by_id`, `get_slot`, `get_shelf_inventory`, `get_all_slots`, `get_shortage_slots`, `get_misplaced_slots`, `get_shelf_sku_summary`, `find_sku_locations`, `find_expected_sku_locations`, `find_sku_world_positions`, `get_sku_total_quantity`, `get_sku_total_quantity_by_shelf` |
| 标定与世界坐标查询 | 4 | `get_shelf_calibration`, `get_slot_world_pos`, `get_all_slots_world`, `get_shelf_group_all_slots_world` |
| 局部坐标计算 | 5 | `level_surface_z`, `face_center_x`, `level_opening_height`, `slot_id_to_local`, `local_to_world` (静态) |
| 工具方法 | 4 | `format_slot_id`, `get_stats`, `slot_id_str_to_tuple`, `close` |
| **合计** | **54** | 含初始化和模块级便捷函数；不含内部方法及上下文管理方法 |

---

### 4.1 初始化

| API | 说明 |
|-----|------|
| `ShelfDatabase(db_path)` | 创建/打开数据库, 自动建表。`db_path=":memory:"` 为内存模式 |

---

### 4.2 货架类型管理 (shelf_types)

| API | 签名 | 返回值 | 说明 |
|-----|------|--------|------|
| `add_shelf_type` | `(name, shelf_length, shelf_width, shelf_height, num_levels, bottom_clearance, level_spacing, panel_thick, back_thick, shelf_depth_normal, shelf_depth_bottom)` | `int` (新类型ID) | 添加一种货架类型 (10 个物理参数) |
| `update_shelf_type` | `(type_id, shelf_length, shelf_width, shelf_height, num_levels, bottom_clearance, level_spacing, panel_thick, back_thick, shelf_depth_normal, shelf_depth_bottom)` | `None` | 更新共享物理参数；不修改类型名称或 ID，并校验关联库存范围 |
| `get_shelf_type` | `(type_id)` | `Optional[ShelfType]` | 按 ID 获取货架类型 |
| `get_shelf_type_by_name` | `(name)` | `Optional[ShelfType]` | 按名称获取货架类型 |
| `get_all_shelf_types` | `()` | `List[ShelfType]` | 获取所有货架类型 |

---

### 4.3 货架组管理 (shelf_groups)

| API | 签名 | 返回值 | 说明 |
|-----|------|--------|------|
| `add_shelf_group` | `(name="", world_x=0.0, world_y=0.0, yaw=0.0, shelf_type_id=None)` | `int` (新货架ID) | 添加货架组 |
| `update_shelf_group` | `(shelf_id, name=None, world_x=None, world_y=None, yaw=None, shelf_type_id=None)` | `None` | 更新货架组, None 参数不修改 |
| `remove_shelf_group` | `(shelf_id)` | `int` | 删除货架组并返回级联删除的库存实例数 |
| `get_shelf_group` | `(shelf_id)` | `Optional[ShelfGroup]` | 获取货架组信息 |
| `get_all_shelf_groups` | `()` | `List[ShelfGroup]` | 获取所有货架组 |
| `get_shelf_world_pos` | `(shelf_id)` | `Optional[WorldPos]` | 获取货架局部原点的 map 坐标 |

---

### 4.4 SKU 目录管理 (sku_catalog)

| API | 签名 | 返回值 | 说明 |
|-----|------|--------|------|
| `register_sku` | `(sku, category="", mesh_file="", tex_file="")` | `None` | 注册/更新一个 SKU；使用安全 UPSERT，不影响已关联库存 |
| `register_skus_batch` | `(skus: List[Dict])` | `None` | 批量注册 SKU；不替换已有目录行 |
| `get_sku_info` | `(sku)` | `Optional[SkuInfo]` | 获取 SKU 信息 |
| `get_all_skus` | `()` | `List[SkuInfo]` | 获取所有 SKU |
| `remove_sku_from_catalog` | `(sku)` | `None` | 删除未被任何货位引用的 SKU；被引用时拒绝 |

---

### 4.5 库存管理 — 写入 (shelf_inventory)

一个 slot = 一个固定货架位置。`slot_id` 和位置字段创建后保持不变。

| API | 签名 | 说明 |
|-----|------|------|
| `create_slot` | `(shelf_id, face, level, y_cm, expected_sku, actual_sku=expected_sku, ...)` | 创建固定位置并返回稳定 `slot_id` |
| `update_slot` | `(slot_id, **changes)` | 仅更新 expected/actual SKU、尺寸和图片目录 |
| `set_actual_sku` | `(slot_id, actual_sku)` | 更新一个位置的实际 SKU，并自动派生状态 |
| `set_actual_sku_batch` | `(changes)` | 在一个事务中批量更新实际 SKU |
| `take_slot` | `(slot_id)` | 将 `actual_sku` 置空，固定位置不删除 |
| `restock_slot` | `(slot_id)` | 将 `actual_sku` 恢复为 `expected_sku` |
| `delete_slot` | `(slot_id)` | 删除固定位置 |
| `import_slots_batch` | `(new_skus, slots)` | 原子创建新 SKU 和多条固定货位；冲突时全部回滚 |
| `clear_shelf_face_level` | `(shelf_id, face, level)` | 清空指定局部 X 侧与层号的库存，返回删除数量 |
| `clear_shelf_face` | `(shelf_id, face)` | 清空指定局部 X 侧全部库存，返回删除数量 |
| `clear_shelf` | `(shelf_id)` | 清空货架组所有库存，返回删除数量；不删除 SKU 目录 |

---

### 4.6 库存查询 (shelf_inventory)

| API | 签名 | 返回值 | 说明 |
|-----|------|--------|------|
| `get_slot_by_id` | `(slot_id)` | `Optional[ShelfSlot]` | 按稳定 ID 获取固定位置及状态 |
| `get_slot` | `(shelf_id, face, level, y_cm)` | `Optional[ShelfSlot]` | 按坐标获取固定位置 |
| `get_shelf_inventory` | `(shelf_id)` | `List[ShelfSlot]` | 获取指定货架所有固定位置 |
| `get_shortage_slots` | `()` | `List[ShelfSlot]` | 获取全部缺货位置 |
| `get_misplaced_slots` | `()` | `List[ShelfSlot]` | 获取全部摆放错误位置 |
| `get_shelf_sku_summary` | `(shelf_id)` | `List[Dict]` | 聚合当前实际 SKU 数量 |
| `find_sku_locations` | `(sku)` | `List[Dict]` | 查询当前实际 SKU 的位置 |
| `find_expected_sku_locations` | `(sku)` | `List[Dict]` | 查询应摆该 SKU 的位置 |
| `find_sku_world_positions` | `(sku)` | `List[Dict]` | 指定 SKU 所在位置的世界坐标 (含坐标计算) |
| `get_sku_total_quantity` | `(sku)` | `int` | 指定 SKU 在所有货架的总数量 |
| `get_sku_total_quantity_by_shelf` | `(sku)` | `List[Dict]` | 指定 SKU 在每个货架的总数量 |

---

### 4.7 标定与坐标查询

| API | 签名 | 返回值 | 说明 |
|-----|------|--------|------|
| `get_slot_world_pos` | `(shelf_id, face, level, y_cm, height_cm=None)` | `Optional[WorldPos]` | 指定商品几何中心的世界坐标 |
| `get_all_slots_world` | `()` | `List[Dict]` | 所有固定位置的世界坐标、双 SKU 和状态 |
| `get_shelf_group_all_slots_world` | `(shelf_id)` | `List[Dict]` | 指定货架组所有固定位置的世界坐标 |
| `get_shelf_calibration` | `(shelf_id)` | `Optional[Dict]` | 获取货架面照片标注所需的层高和开口尺寸 |

---

### 4.8 坐标计算方法

| API | 签名 | 返回值 | 说明 |
|-----|------|--------|------|
| `level_surface_z(level, shelf_type=None)` | `(int, ShelfType?)` | `float` | 指定层的层板上表面 Z 坐标 (局部, 单位: 米) |
| `face_center_x(face, level, shelf_type=None)` | `(int, int, ShelfType?)` | `float` | 指定面/层的层板中心 X 坐标 (局部, 单位: 米) |
| `level_opening_height(level, shelf_type=None)` | `(int, ShelfType?)` | `float` | 指定层可用于照片标定的开口高度 (局部, 单位: 米) |
| `slot_id_to_local(shelf_id, face, level, y_cm, height_cm=None)` | `(int, int, int, float, float?)` | `LocalPos` | 槽位 → 局部坐标；Z 为层板表面加 `height_cm / 2` |
| `local_to_world(local, world_x, world_y, yaw)` | `(LocalPos, float, float, float)` | `WorldPos` | 局部坐标 → 世界坐标 (静态方法) |

> **内部方法**: `_resolve_shelf_params(shelf_id)` → `ShelfType` — 解析货架物理参数的统一入口。优先级: `shelf_type_id` → `shelf_types` 表 → `DEFAULT_*` 常量。

---

### 4.9 其他工具方法

| API | 签名 | 返回值 | 说明 |
|-----|------|--------|------|
| `get_stats()` | `()` | `Dict` | 包含 `total_positions`、`actual_items`、`shortages`、`misplacements` |
| `slot_id_str_to_tuple(slot_id_str)` | `(str)` | `(int, int, int, float)` | 将 `"0-1-2-9"` 解析为 `(shelf_id, face, level, y_cm)` |
| `close()` | `()` | `None` | 关闭数据库连接 |
| `__enter__` / `__exit__` | — | — | 支持 `with` 语句 |

---

### 4.10 模块级便捷函数

| 函数 | 说明 |
|------|------|
| `init_database_from_scene_params(db, shelf_positions, skus)` | 根据场景参数列表初始化数据库 |

---

## 五、数据类 (Dataclass)

| 类名 | 字段 | 说明 |
|------|------|------|
| `ShelfType` | `id, name, shelf_length, shelf_width, shelf_height, num_levels, bottom_clearance, level_spacing, panel_thick, back_thick, shelf_depth_normal, shelf_depth_bottom` | 货架类型标识及 10 个物理参数 |
| `ShelfGroup` | `id, name, world_x, world_y, yaw, shelf_type_id, created_at` | 货架组信息 |
| `SkuInfo` | `sku, category, mesh_file, tex_file` | SKU 信息 |
| `ShelfSlot` | `slot_id, shelf_id, face, level, y_cm, expected_sku, actual_sku, width_cm, height_cm, image_dir, status` | 固定位置、当前内容和派生状态 |
| `LocalPos` | `x, y, z` | 货架局部坐标 |
| `WorldPos` | `x, y, z` | 世界坐标 |

---

## 六、当前数据规模

以下是仓库当前 `shelf_inventory.db` 的快照；数据库继续变化时，以 `GET /api/state` 或 `ShelfDatabase.get_stats()` 为准。

| 项目 | 数量 |
|------|------|
| 货架类型 | 1 ("standard") |
| 货架组 | 4 |
| 交付桌 | 4 |
| SKU 种类 | 22 |
| 当前固定位置 | 35 |
| 当前有货位置 | 35 |
| 当前缺货位置 | 0 |
| 当前错放位置 | 0 |

---

## 七、使用示例

```python
from shelf_database import (ShelfDatabase, ShelfType,
    DEFAULT_SHELF_LENGTH, DEFAULT_SHELF_WIDTH, DEFAULT_SHELF_HEIGHT,
    DEFAULT_NUM_LEVELS, DEFAULT_BOTTOM_CLEARANCE, DEFAULT_LEVEL_SPACING,
    DEFAULT_PANEL_THICK, DEFAULT_BACK_THICK,
    DEFAULT_SHELF_DEPTH_NORMAL, DEFAULT_SHELF_DEPTH_BOTTOM)

db = ShelfDatabase("shelf_inventory.db")

# --- 创建货架类型 ---
type_id = db.add_shelf_type(
    name="standard",
    shelf_length=DEFAULT_SHELF_LENGTH,
    shelf_width=DEFAULT_SHELF_WIDTH,
    shelf_height=DEFAULT_SHELF_HEIGHT,
    num_levels=DEFAULT_NUM_LEVELS,
    bottom_clearance=DEFAULT_BOTTOM_CLEARANCE,
    level_spacing=DEFAULT_LEVEL_SPACING,
    panel_thick=DEFAULT_PANEL_THICK,
    back_thick=DEFAULT_BACK_THICK,
    shelf_depth_normal=DEFAULT_SHELF_DEPTH_NORMAL,
    shelf_depth_bottom=DEFAULT_SHELF_DEPTH_BOTTOM,
)

# --- 添加货架组 (关联类型) ---
sid = db.add_shelf_group(
    name="shelf_0",
    world_x=-1.5, world_y=0.25,
    yaw=0.0,
    shelf_type_id=type_id,
)

# --- 查询货架类型 ---
st = db.get_shelf_type(type_id)
print(f"类型: {st.name}, 层数: {st.num_levels}")

# --- 创建正常、缺货和错放位置 ---
normal_id = db.create_slot(sid, 0, 2, 50, "cracker_box", "cracker_box", width_cm=12, height_cm=8)
missing_id = db.create_slot(sid, 0, 2, 65, "tomato_soup_can", None, width_cm=7, height_cm=8)
wrong_id = db.create_slot(sid, 1, 4, 90, "banana", "cracker_box", width_cm=15, height_cm=5)

print(db.get_slot_by_id(missing_id).status)  # 缺货
db.restock_slot(missing_id)
db.take_slot(normal_id)

# 查询 cracker_box 在哪些位置
for loc in db.find_sku_locations("cracker_box"):
    print(f"  货架{loc['shelf_id']} 面{loc['face']} 层{loc['level']} "
          f"y={loc['y_cm']:.0f}cm height={loc['height_cm'] or 0:.0f}cm")

# 查询所有 cracker_box 的世界位置
positions = db.find_sku_world_positions("cracker_box")
for p in positions:
    print(f"  {p['slot_id']}: ({p['world_x']:.3f}, {p['world_y']:.3f}, {p['world_z']:.3f})")

# 删除固定位置；拿走商品应使用 take_slot
db.delete_slot(wrong_id)

# 生成 MuJoCo 场景
all_slots = db.get_all_slots_world()
for s in all_slots:
    # s 包含 slot_id、expected_sku、actual_sku、status 和世界坐标
    ...

db.close()
```
