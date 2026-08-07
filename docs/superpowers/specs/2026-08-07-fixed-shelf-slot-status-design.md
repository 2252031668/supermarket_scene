# 固定货位与货架状态设计

日期：2026-08-07

## 目标

把 `slot` 从“当前存在的一个商品实例”改为“货架上的固定位置”。每个固定位置记录应该摆放的 SKU 和当前实际 SKU，系统据此自动判断正常、缺货或摆放错误。

本阶段同时调整 SQLite、JSON、API 和 Web 管理界面。CV 不接入 Web，只保留未来可直接读取的 JSON 数据和可复用的 slot 更新 API。

## 范围

本阶段包括：

- 稳定的文本 `slot_id` 主键；
- `expected_sku` 与 `actual_sku`；
- 自动计算的货位状态；
- 缺货与摆放错误查询；
- `take`、`restock` 和固定位置删除操作；
- 人工照片导入对正常商品、空位和错放商品的支持；
- SQLite 更新后同步每个货架的标定 JSON；
- Web 列表、编辑器、统计和 3D 状态展示；
- 现有数据库和 JSON 的一次性迁移。

本阶段不包括：

- 巡查照片上传页面；
- CV 识别接口、结果审核页面和自动写库；
- 后仓库存、补货篮数量和库存扣减；
- 状态历史、巡查历史和审计日志；
- 旧 JSON `products` 格式兼容层。

## 数据所有权

SQLite 是货位业务数据的唯一真实来源。Web 和其他调用方只能通过 API 修改 SQLite。

`data/shelf_calibration/{shelf_id}.json` 是面向 CV 的读取文件。它保存照片标定信息、像素框，以及从 SQLite 导出的当前货位数据。JSON 中重复出现的 `slot_id`、`expected_sku` 和 `actual_sku` 不是独立主数据。

CV 未来直接读取 JSON；识别结果写回时调用现有 slot API，不直接编辑 JSON。

## 固定货位模型

`shelf_inventory` 保留现有表名，但语义改为固定货位：

```sql
CREATE TABLE shelf_inventory (
    slot_id       TEXT PRIMARY KEY,
    shelf_id      INTEGER NOT NULL,
    face          INTEGER NOT NULL CHECK(face IN (0, 1)),
    level         INTEGER NOT NULL CHECK(level >= 0),
    y_cm          REAL NOT NULL CHECK(y_cm >= 0),
    expected_sku  TEXT NOT NULL,
    actual_sku    TEXT DEFAULT NULL,
    width_cm      REAL DEFAULT NULL CHECK(width_cm IS NULL OR width_cm >= 0),
    height_cm     REAL DEFAULT NULL CHECK(height_cm IS NULL OR height_cm >= 0),
    image_dir     TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (shelf_id) REFERENCES shelf_groups(id) ON DELETE CASCADE,
    FOREIGN KEY (expected_sku) REFERENCES sku_catalog(sku) ON DELETE RESTRICT,
    FOREIGN KEY (actual_sku) REFERENCES sku_catalog(sku) ON DELETE RESTRICT,
    UNIQUE(shelf_id, face, level, y_cm)
);
```

原来的自增整数 `id` 删除。`slot_id` 在首次创建时由后端使用现有规范生成：

```text
{shelf_id}-{face}-{level}-{y_cm:g}
```

示例：`1-0-2-43`、`1-0-2-43.5`。

前端不生成也不提交 `slot_id`。后端生成后持久化并返回。`slot_id` 同时用于 API 路径、JSON、图片目录、缺货/错放列表和世界坐标查询。

创建后，`slot_id`、`shelf_id`、`face`、`level` 和 `y_cm` 不允许修改。位置标错时删除旧货位并创建新货位。货架整体位姿变化只修改 `shelf_groups`，不会改变其货位 ID。

## 状态规则

状态不存入数据库，也没有状态修改 API：

```text
actual_sku IS NULL                -> 缺货
actual_sku = expected_sku         -> 正常
actual_sku != expected_sku        -> 摆放错误
```

所有查询和 API 返回通过同一段数据库逻辑计算 `status`，避免字段与状态不一致。

`expected_sku` 必填并引用 SKU 目录。`actual_sku` 可空；非空时也必须引用 SKU 目录。

SKU 删除使用 `RESTRICT`。任何固定货位仍以该 SKU 作为应摆或实际 SKU 时，删除请求返回冲突，不级联删除货位。

## 读写语义

初次录入正常商品：

```text
expected_sku = 选择的 SKU
actual_sku = expected_sku
```

初次录入空位：

```text
expected_sku = 选择的 SKU
actual_sku = NULL
```

初次录入错放商品：

```text
expected_sku = 该位置应该摆放的 SKU
actual_sku = 当前实际商品的 SKU
```

拿走商品只清空 `actual_sku`，不删除固定货位。补货完成把 `actual_sku` 设置为 `expected_sku`。删除固定货位是独立且明确的危险操作。

现有库存聚合和 MuJoCo 场景生成只统计或生成 `actual_sku IS NOT NULL` 的实际商品。按计划查询位置时使用 `expected_sku`。

世界坐标不持久化。`GET /api/slots/{slot_id}/world-position` 查询固定货位，再按货架位姿与局部位置计算。

## API

### 创建固定货位

```http
POST /api/slots
```

请求不含 `slot_id`：

```json
{
  "shelf_id": 1,
  "face": 0,
  "level": 2,
  "y_cm": 43,
  "expected_sku": "雪碧",
  "actual_sku": "雪碧",
  "width_cm": 6,
  "height_cm": 22
}
```

创建普通商品时，省略 `actual_sku` 表示默认等于 `expected_sku`。创建缺货位置时必须显式提交 `actual_sku: null`，避免省略和空值含义混淆。

### 编辑货位状态

```http
PUT /api/slots/{slot_id}
```

仅允许修改 `expected_sku`、`actual_sku`、`width_cm`、`height_cm` 和受控图片信息。位置字段和 `slot_id` 不接受修改。

### 业务动作

```http
POST /api/slots/{slot_id}/take
POST /api/slots/{slot_id}/restock
DELETE /api/slots/{slot_id}
```

- `take`：设置 `actual_sku = NULL`；
- `restock`：设置 `actual_sku = expected_sku`；
- `DELETE`：删除固定货位、对应 JSON 项和受控实例图片。

这些动作是幂等的：重复拿走或重复确认补货不会创建新记录。

### 查询

```http
GET /api/shortages
GET /api/misplacements
GET /api/slots/{slot_id}/world-position
```

缺货和错放项至少返回 `slot_id`、位置字段、`expected_sku`、`actual_sku` 和计算后的 `status`。

## JSON v2

迁移后不保留旧 `products`：

```json
{
  "schema_version": 2,
  "shelf_id": 1,
  "shelf_name": "1号货架",
  "faces": {
    "0": {
      "image_file": "data/shelf_images/1/-x_0.png",
      "image_hash": "sha256:...",
      "layers": {},
      "slots": [
        {
          "slot_id": "1-0-2-43",
          "expected_sku": "雪碧",
          "actual_sku": null,
          "bbox": {
            "x": 1424,
            "y": 909,
            "width": 32,
            "height": 219
          }
        }
      ]
    }
  }
}
```

JSON 不保存 `status`，消费者根据两个 SKU 使用同一规则计算。`image_file`、`image_hash`、`layers` 和 `bbox` 由标定流程维护；货位状态同步时必须保留这些标定字段。

初次人工导入同时创建数据库货位并写入相应 JSON `slots`。后续创建、编辑、拿走、补货和删除固定货位后，后端读取该货架数据库记录，按 `slot_id` 保留已有 `bbox`，并重建对应 JSON 的 `slots`。

JSON 写入使用同目录临时文件并通过 `os.replace` 原子替换。SQLite 先提交；JSON 同步失败时 SQLite 仍是主数据，API 返回明确错误，维护命令可从 SQLite 与已有标定字段重建 JSON。当前本地单进程范围不增加同步状态表或分布式事务。

## Web 录入和管理

现有人工导入流程保留，框选工具增加“空位”开关：

- 普通框选默认 `expected_sku = actual_sku`；
- 空位框选设置 `expected_sku`，并显式提交 `actual_sku = null`；
- 错放商品先按实际 SKU 框选，再在审核表修改应摆 SKU。

审核表显示裁剪图、`slot_id`、应摆 SKU、实际 SKU、位置、尺寸和只读状态。导入确认前可调整位置；创建后位置字段只读。

普通货位编辑器显示稳定 `slot_id`、只读位置、应摆 SKU、实际 SKU和自动状态，并按状态提供拿走或补货动作。删除固定位置放在单独的危险操作区。

主页面提供全部位置、缺货和摆放错误筛选。缺货列表显示应补 SKU 和稳定 `slot_id`；错放列表同时显示应摆与实际 SKU。点击列表项选中对应场景位置。

3D 管理场景：

- 正常位置显示 `actual_sku` 商品；
- 缺货位置显示可点击的红色空心位置框；
- 摆放错误位置显示 `actual_sku` 商品并增加黄色状态边框。

状态不能只靠颜色表达，列表和编辑面板同时显示文字。

统计改为固定位置总数、当前商品数、缺货数和摆放错误数。已有 `total_items` 的含义不能继续代表数据库行数。

## 一次性迁移

实现时编写并运行一次性迁移脚本：

1. 备份 `shelf_inventory.db` 和 `data/shelf_calibration`；
2. 重建 `shelf_inventory`，为每条旧记录生成稳定 `slot_id`；
3. 将旧 `sku` 同时写入 `expected_sku` 和 `actual_sku`；
4. 将每个 JSON `products` 转换为 `slots`，保留 bbox、层位、图片路径和哈希；
5. 校验数据库与 JSON 的 `slot_id` 唯一性和对应关系；
6. 校验通过后删除旧表、旧 `products` 字段、迁移脚本和兼容代码。

迁移遇到重复位置、非法 ID、JSON 孤立项或无法对应的数据库行时立即失败并报告，不静默丢弃数据。备份保留，迁移脚本按用户要求在成功后删除。

旧 CV 当前读取 `products`，JSON v2 后不再兼容。CV 已明确不属于本阶段 Web 功能，后续单独改为读取 `slots`。

## 错误处理

- 创建位置越界、SKU 不存在、位置重复或生成的 `slot_id` 冲突时返回 400；
- 查询、编辑或操作不存在的 `slot_id` 时返回 404；
- 删除仍被货位引用的 SKU 时返回 409；
- JSON 同步失败时返回明确错误并保留 SQLite 已提交数据，不伪装成整体成功；
- 批量人工导入在数据库写入前完成全部位置、SKU、图片和重复 ID 校验。

## 验证

最小自动检查覆盖：

- 旧数据库记录迁移后 `expected_sku = actual_sku` 且 `slot_id` 正确；
- 正常、缺货和错放三种状态计算；
- `take`、`restock` 的幂等行为；
- 固定位置字段不能通过更新 API 修改；
- 实际库存聚合和 MuJoCo 输出跳过 `actual_sku IS NULL`；
- JSON `products` 完全迁移为 `slots`，并保留 bbox 与标定字段；
- SQLite 更新后 JSON 同步，写入失败时原 JSON 不被截断；
- Web 构建通过，正常、缺货和错放位置均可选中和编辑。

实现完成后运行 Python 自检、前端构建，并在桌面和移动宽度下验证主要 Web 流程。
