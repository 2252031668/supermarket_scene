# 超市货架场景管理器

本地货架管理系统：用 React + Three.js 管理货架，用 SQLite 记录稳定货位与库存状态，并提供人工录入、巡检识别、SKU 查询和货架面图片拼接。

## 核心数据模型

一个 `slot` 是货架上的固定位置，首次录入时生成稳定的 `slot_id`：`{shelf_id}-{face}-{level}-{y_cm}`。后续缺货、错放和补货都只更新 SKU，不改变位置 ID。

| 字段 | 含义 |
| --- | --- |
| `expected_sku` | 该位置应摆放的 SKU |
| `actual_sku` | 该位置当前实际 SKU；`NULL` 表示缺货 |
| 状态 | 由两个字段派生：相同为正常，`actual_sku = NULL` 为缺货，否则为摆放错误 |

`shelf_inventory.db` 是唯一业务真源。每次通过 API 改动货位后，服务会同步生成 `data/shelf_calibration/{shelf_id}.json` 的 `slots` 投影，供 CV 快速读取。不要手工分别维护数据库和 JSON。世界坐标由稳定 `slot_id` 和货架参数统一查询。

详细字段、API 和坐标公式见 [数据库与 API 文档](docs/database_schema_and_api.md)。

## 仓库结构

```text
.
├── api_server.py             本地 HTTP JSON API，视觉模型在启动时预加载
├── shelf_database.py         SQLite 表结构、迁移、坐标和库存读写
├── calibration_manager.py    数据库到 CV 校准 JSON 的同步
├── init_database.py          重建示例数据库
├── web/                      React + Three.js 管理界面
├── vision/                   巡检、SKU 查询、图片拼接和共享视觉模块
├── mujoco/                   MuJoCo 场景生成器、MJCF 和模型资产
├── tests/                    固定货位与状态逻辑测试
├── test_pic/                 本地测试图片
└── data/                     本地运行时照片和校准 JSON，已忽略
```

## 安装与启动

Python 依赖使用 `uv`，前端依赖使用 npm：

```bash
uv sync
cd web
npm install
```

需要 Ark VLM 或其他远程模型时，创建本地配置并填入密钥：

```bash
cp vision/config.example.yaml vision/config.local.yaml
```

`vision/config.example.yaml` 是可提交的模板；`vision/config.local.yaml` 是运行时本地配置，已忽略。没有本地配置时服务采用 `vision/config.py` 的默认参数，涉及远程 VLM 的功能则需要对应 API Key。

在仓库根目录启动 API：

```bash
uv run python api_server.py
```

在另一个终端启动网页：

```bash
cd web
npm run dev
```

打开 `http://127.0.0.1:5173`。前端会将 `/api` 代理到 `http://127.0.0.1:8000`。

## Web 功能

- **货架管理**：维护货架、交付桌、SKU 和固定货位，提供 2D/3D 视图。
- **人工批量录入**：标定货架面、框选已有商品或空位，填写 `expected_sku` 与 `actual_sku` 后批量写入。
- **巡检识别**：上传局部货架照片，先用 SIFT 配准和 Lab 色彩距离筛选货位异常，再以 DINOv2 识别 SKU；可选 Ark VLM 保底。结果仅在点击“应用修改”后才更新数据库和 JSON。
- **货物查询**：按 SKU 或固定位置查询局部照片中的目标货物，使用 Ark VLM 定位，并可选 DINOv2 复核候选框。
- **图片拼接**：上传同一货架面的 2 至 8 张重叠照片，手工选择主平面。后端对所有图片两两进行 SIFT/MAGSAC 匹配，拼接主图所在的可靠连通组；再用 GraphCut 接缝选择和三层多频段融合输出拼接图。结果可四点透视校正后下载，或直接送到人工批量录入。

视觉运行结果保存在 `vision/output/`：巡检为 `slot_inspection/{run_id}/`，SKU 查询为 `vlm_sku_query/{run_id}/`，图片拼接为 `image_stitch/{run_id}/`。这些都是可删除的运行产物。

页面保存的巡检和 SKU 查询参数写入 `vision/config.local.yaml`。单次 API 请求可携带 `config` 覆盖本次运行参数，但不会回写本地配置。

## MuJoCo

从仓库根目录运行：

```bash
uv run python -m mujoco.generate_scene
uv run python -m mujoco.generate_scene_from_database
uv run python -m mujoco.take_screenshots
```

`init_database.py` 会删除并重建 `shelf_inventory.db`，仅用于重置示例/测试数据：

```bash
uv run python init_database.py
```

## 检查

```bash
uv run python -m unittest tests.test_fixed_slots
uv run python -m py_compile api_server.py shelf_database.py calibration_manager.py scene_geometry.py init_database.py mujoco/*.py vision/*.py
cd web && npm run build
```

`.gitignore` 已排除本地 `data/`、`vision/output/`、模型权重、Python 缓存与前端构建产物。
