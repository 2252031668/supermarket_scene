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

## 三、API 完整清单

巡检相关接口：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/vision/config | 返回 DINO 阈值、并列差值、Ark 保底和调试图开关，不返回 API 密钥 |
| PUT | /api/vision/config | 校验并保存 vision/config.local.yaml 中的巡检配置 |
| POST | /api/vision/inspect | 上传局部照片并生成只读巡检报告，不修改 SQLite |
| GET | /api/vision/runs/{run_id}/result | 读取一次巡检的固定位置结果 |
| GET | /api/vision/runs/{run_id}/artifact/{name} | 读取报告中登记的调试产物 |
| POST | /api/vision/runs/{run_id}/apply | 仅应用请求中勾选且属于该报告的 slot ID，并同步 JSON |

巡检配置默认值：

    inspection:
      dino_confidence_threshold: 0.72
      ambiguity_margin: 0.05
      vlm_fallback: false
      vlm_top_k: 4
      save_debug: true

巡检结果先写入 vision/output/slot_inspection/{run_id}/result.json。低置信度且未开启 Ark 保底时，actual_sku 为 null；开启保底但 Ark 返回空位、无法解析或请求失败时，同样为 null。API 只允许应用报告中标记为可变更、且用户提交的 slot ID，更新成功后由数据库派生“正常、缺货、摆放错误”状态，再同步 data/shelf_calibration/{shelf_id}.json。

### HTTP JSON API (`api_server.py`)

| 方法 | 路径 | 返回值 | 说明 |
|------|------|--------|------|
| `GET` | `/api/state` | 完整数据库快照 | 网页手动刷新时读取；包含全部货位的世界坐标 |
| `GET` | `/api/shortages` | `{slots}` | 缺货位置列表 |
| `GET` | `/api/misplacements` | `{slots}` | 摆放错误位置列表 |
| `GET` | `/api/skus/{sku}/world-positions` | `{sku, positions}` | 查询一个当前实际 SKU 的全部 `map` 世界坐标 |
| `GET` | `/api/slots/{slot_id}/world-position` | `{slot}` | 按稳定位置 ID 查询 `map` 世界坐标 |
| `POST` | `/api/slots` | `{slot, state}` | 创建固定货位；服务端生成 `slot_id` |
| `PUT` | `/api/slots/{slot_id}` | `{slot, state}` | 更新 expected/actual SKU、尺寸或 bbox；拒绝位置字段 |
| `POST` | `/api/slots/{slot_id}/take` | `{slot, state}` | 拿走商品，将 `actual_sku` 置空 |
| `POST` | `/api/slots/{slot_id}/restock` | `{slot, state}` | 补货，将 `actual_sku` 设为 `expected_sku` |
| `DELETE` | `/api/slots/{slot_id}` | `{deleted, state}` | 删除固定位置 |
| `GET` | `/api/shelves/{id}/calibration` | 货架类型层位标定尺寸 | 返回货架长度、各层板表面高度与可用高度 |
| `PUT` | `/api/shelf-types/{id}` | `{state}` | 更新货架类型的 10 个物理参数；拒绝使已关联商品落到层号或长度范围外的修改 |
| `POST` | `/api/delivery-tables` | `{id, state}` | 新建交付桌，字段为 `name`、`world_x`、`world_y`、`yaw` |
| `PUT` | `/api/delivery-tables/{id}` | `{state}` | 更新交付桌名称或位姿 |
| `DELETE` | `/api/delivery-tables/{id}` | `{state}` | 删除交付桌；不影响库存 |
| `POST` | `/api/imports/manual` | `{slot_ids, state}` | 人工照片标注审核后的批量事务导入，并保存每个实例的 `0.png` 裁剪图 |

所有世界坐标字段为 `world_x`、`world_y`、`world_z`，单位为米，坐标系为本文件定义的右手 `map` 坐标系。

### 3.1 货架面原图文件

人工批量导入确认时，会将上传的整面照片转换为 PNG 并写入受控目录，不增加独立数据表：

```text
data/shelf_images/{shelf_id}/-x_0.png
data/shelf_images/{shelf_id}/+x_0.png
```

文件名中的 `-x` / `+x` 表示货架局部坐标面，分别对应 `face=0` / `face=1`；`0` 是当前唯一的面图。对同一货架面再次人工导入会替换该文件。`GET /api/state` 的 `shelf_images` 数组仅列出实际存在的面图，`GET /api/shelf-images/{shelf_id}/{face}/0.png` 可读取其 PNG。删除整个货架会一并清理该货架的图片目录。

### 3.0 总览

| 分类 | 方法数 | 涉及方法 |
|------|--------|----------|
| 初始化 | 1 | `ShelfDatabase(db_path)` |
| 货架类型管理 | 5 | `add_shelf_type`, `update_shelf_type`, `get_shelf_type`, `get_shelf_type_by_name`, `get_all_shelf_types` |
| 货架组管理 | 6 | `add_shelf_group`, `update_shelf_group`, `remove_shelf_group`, `get_shelf_group`, `get_all_shelf_groups`, `get_shelf_world_pos` |
| 交付桌管理 | 5 | `add_delivery_table`, `update_delivery_table`, `remove_delivery_table`, `get_delivery_table`, `get_all_delivery_tables` |
| SKU 目录管理 | 5 | `register_sku`, `register_skus_batch`, `get_sku_info`, `get_all_skus`, `remove_sku_from_catalog` |
| 货位写入 | 7 | `create_slot`, `update_slot`, `take_slot`, `restock_slot`, `delete_slot`, `import_slots_batch`, `clear_shelf` |
| 货位查询 | 8 | `get_slot_by_id`, `get_slot`, `get_all_slots`, `get_shelf_inventory`, `get_shortage_slots`, `get_misplaced_slots`, `find_sku_locations`, `find_expected_sku_locations` |
| 坐标计算 | 4 | `get_slot_world_pos`, `get_all_slots_world`, `get_shelf_group_all_slots_world`, `slot_id_to_local`, `local_to_world` (静态) |
| 工具方法 | 3 | `get_stats`, `slot_id_str_to_tuple`, `close` |
| **合计** | **33** | |

---

### 3.1 初始化

| API | 说明 |
|-----|------|
| `ShelfDatabase(db_path)` | 创建/打开数据库, 自动建表。`db_path=":memory:"` 为内存模式 |

---

### 3.2 货架类型管理 (shelf_types)

| API | 签名 | 返回值 | 说明 |
|-----|------|--------|------|
| `add_shelf_type` | `(name, shelf_length, shelf_width, shelf_height, num_levels, bottom_clearance, level_spacing, panel_thick, back_thick, shelf_depth_normal, shelf_depth_bottom)` | `int` (新类型ID) | 添加一种货架类型 (10 个物理参数) |
| `update_shelf_type` | `(type_id, shelf_length, shelf_width, shelf_height, num_levels, bottom_clearance, level_spacing, panel_thick, back_thick, shelf_depth_normal, shelf_depth_bottom)` | `None` | 更新共享物理参数；不修改类型名称或 ID，并校验关联库存范围 |
| `get_shelf_type` | `(type_id)` | `Optional[ShelfType]` | 按 ID 获取货架类型 |
| `get_shelf_type_by_name` | `(name)` | `Optional[ShelfType]` | 按名称获取货架类型 |
| `get_all_shelf_types` | `()` | `List[ShelfType]` | 获取所有货架类型 |

---

### 3.3 货架组管理 (shelf_groups)

| API | 签名 | 返回值 | 说明 |
|-----|------|--------|------|
| `add_shelf_group` | `(name="", world_x=0.0, world_y=0.0, yaw=0.0, shelf_type_id=None)` | `int` (新货架ID) | 添加货架组 |
| `update_shelf_group` | `(shelf_id, name=None, world_x=None, world_y=None, yaw=None, shelf_type_id=None)` | `None` | 更新货架组, None 参数不修改 |
| `remove_shelf_group` | `(shelf_id)` | `int` | 删除货架组并返回级联删除的库存实例数 |
| `get_shelf_group` | `(shelf_id)` | `Optional[ShelfGroup]` | 获取货架组信息 |
| `get_all_shelf_groups` | `()` | `List[ShelfGroup]` | 获取所有货架组 |
| `get_shelf_world_pos` | `(shelf_id)` | `Optional[WorldPos]` | 获取货架局部原点的 map 坐标 |

---

### 3.4 SKU 目录管理 (sku_catalog)

| API | 签名 | 返回值 | 说明 |
|-----|------|--------|------|
| `register_sku` | `(sku, category="", mesh_file="", tex_file="")` | `None` | 注册/更新一个 SKU；使用安全 UPSERT，不影响已关联库存 |
| `register_skus_batch` | `(skus: List[Dict])` | `None` | 批量注册 SKU；不替换已有目录行 |
| `get_sku_info` | `(sku)` | `Optional[SkuInfo]` | 获取 SKU 信息 |
| `get_all_skus` | `()` | `List[SkuInfo]` | 获取所有 SKU |
| `remove_sku_from_catalog` | `(sku)` | `None` | 删除未被任何货位引用的 SKU；被引用时拒绝 |

---

### 3.5 库存管理 — 写入 (shelf_inventory)

一个 slot = 一个固定货架位置。`slot_id` 和位置字段创建后保持不变。

| API | 签名 | 说明 |
|-----|------|------|
| `create_slot` | `(shelf_id, face, level, y_cm, expected_sku, actual_sku=expected_sku, ...)` | 创建固定位置并返回稳定 `slot_id` |
| `update_slot` | `(slot_id, **changes)` | 仅更新 expected/actual SKU、尺寸和图片目录 |
| `take_slot` | `(slot_id)` | 将 `actual_sku` 置空，固定位置不删除 |
| `restock_slot` | `(slot_id)` | 将 `actual_sku` 恢复为 `expected_sku` |
| `delete_slot` | `(slot_id)` | 删除固定位置 |
| `import_slots_batch` | `(new_skus, slots)` | 原子创建新 SKU 和多条固定货位；冲突时全部回滚 |
| `clear_shelf_face_level` | `(shelf_id, face, level)` | 清空指定局部 X 侧与层号的库存，返回删除数量 |
| `clear_shelf_face` | `(shelf_id, face)` | 清空指定局部 X 侧全部库存，返回删除数量 |
| `clear_shelf` | `(shelf_id)` | 清空货架组所有库存，返回删除数量；不删除 SKU 目录 |

---

### 3.6 库存查询 (shelf_inventory)

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

### 3.7 坐标查询 (用于场景生成)

| API | 签名 | 返回值 | 说明 |
|-----|------|--------|------|
| `get_slot_world_pos` | `(shelf_id, face, level, y_cm, height_cm=None)` | `Optional[WorldPos]` | 指定商品几何中心的世界坐标 |
| `get_all_slots_world` | `()` | `List[Dict]` | 所有固定位置的世界坐标、双 SKU 和状态 |
| `get_shelf_group_all_slots_world` | `(shelf_id)` | `List[Dict]` | 指定货架组所有固定位置的世界坐标 |

---

### 3.8 坐标计算方法

| API | 签名 | 返回值 | 说明 |
|-----|------|--------|------|
| `level_surface_z(level, shelf_type=None)` | `(int, ShelfType?)` | `float` | 指定层的层板上表面 Z 坐标 (局部, 单位: 米) |
| `face_center_x(face, level, shelf_type=None)` | `(int, int, ShelfType?)` | `float` | 指定面/层的层板中心 X 坐标 (局部, 单位: 米) |
| `slot_id_to_local(shelf_id, face, level, y_cm, height_cm=None)` | `(int, int, int, float, float?)` | `LocalPos` | 槽位 → 局部坐标；Z 为层板表面加 `height_cm / 2` |
| `local_to_world(local, world_x, world_y, yaw)` | `(LocalPos, float, float, float)` | `WorldPos` | 局部坐标 → 世界坐标 (静态方法) |

> **内部方法**: `_resolve_shelf_params(shelf_id)` → `ShelfType` — 解析货架物理参数的统一入口。优先级: `shelf_type_id` → `shelf_types` 表 → `DEFAULT_*` 常量。

---

### 3.9 其他工具方法

| API | 签名 | 返回值 | 说明 |
|-----|------|--------|------|
| `get_stats()` | `()` | `Dict` | 包含 `total_positions`、`actual_items`、`shortages`、`misplacements` |
| `slot_id_str_to_tuple(slot_id_str)` | `(str)` | `(int, int, int, float)` | 将 `"0-1-2-9"` 解析为 `(shelf_id, face, level, y_cm)` |
| `close()` | `()` | `None` | 关闭数据库连接 |
| `__enter__` / `__exit__` | — | — | 支持 `with` 语句 |

---

### 3.10 模块级便捷函数

| 函数 | 说明 |
|------|------|
| `init_database_from_scene_params(db, shelf_positions, skus)` | 根据场景参数列表初始化数据库 |

---

## 四、数据类 (Dataclass)

| 类名 | 字段 | 说明 |
|------|------|------|
| `ShelfType` | `id, name, shelf_length, shelf_width, shelf_height, num_levels, bottom_clearance, level_spacing, panel_thick, back_thick, shelf_depth_normal, shelf_depth_bottom` | 货架类型物理参数 (10字段) |
| `ShelfGroup` | `id, name, world_x, world_y, yaw, shelf_type_id, created_at` | 货架组信息 |
| `SkuInfo` | `sku, category, mesh_file, tex_file` | SKU 信息 |
| `ShelfSlot` | `slot_id, shelf_id, face, level, y_cm, expected_sku, actual_sku, width_cm, height_cm, image_dir, status` | 固定位置、当前内容和派生状态 |
| `LocalPos` | `x, y, z` | 货架局部坐标 |
| `WorldPos` | `x, y, z` | 世界坐标 |

---

## 五、当前数据规模

| 项目 | 数量 |
|------|------|
| 货架类型 | 1 ("standard") |
| 货架组 | 4 |
| SKU 种类 | 19 (11 YCB + 8 scanned) |
| 当前固定位置 | 以 `get_stats().total_positions` 为准 |

---

## 六、使用示例

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
