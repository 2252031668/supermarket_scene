# ROS2 服务接口设计

> 状态：ROS2 源码包已写入仓库，尚未在安装 ROS2 的环境中构建验证。

目标是让机器人直接调用类型化 ROS2 service，不处理 HTTP、JSON、Base64 或完整 Web `state`。ROS2 节点直接调用共享业务层；Web API 和 ROS2 使用同一套数据库、视觉逻辑与 JSON 投影同步逻辑。

## 包结构

```text
ros2/
├── supermarket_scene_interfaces/
│   ├── msg/
│   └── srv/
└── supermarket_scene_ros/
    └── supermarket_scene_node.py
```

`supermarket_scene_interfaces` 只包含消息和服务定义；`supermarket_scene_ros` 包含 `rclpy` 节点。业务代码从 `api_server.py` 中抽到共享模块，两个传输层都调用该模块，不复制业务逻辑。

节点通过 `SUPERMARKET_SCENE_ROOT` 定位仓库根目录中的 `robot_service.py`。从本仓库源码目录启动时会自动找到；安装到其他 ROS2 工作区后，启动前应设置该环境变量。

## 命名空间

服务统一使用：

```text
/supermarket_scene/<service_name>
```

图像使用 `sensor_msgs/msg/CompressedImage`，格式为 JPEG、PNG 或 WebP。坐标使用 `map` frame，单位为米；`shelf_yaw` 使用弧度。

## 服务速查

| 服务名 | 请求输入 | 成功输出 |
| --- | --- | --- |
| `/supermarket_scene/inspect_shelf` | `CompressedImage image` | 货架 ID、面、`InspectionSlot[] anomalies` |
| `/supermarket_scene/find_sku` | `CompressedImage image`、`string query` | SKU、参考 slot、VLM 耗时、`DetectedBox[] boxes` |
| `/supermarket_scene/get_slot_pose` | `string slot_id` | `SlotPose slot` |
| `/supermarket_scene/get_sku_locations` | `string sku` | 数量与 `SlotPose[] positions` |
| `/supermarket_scene/list_shortages` | 无 | 数量与缺货 `SlotPose[] slots` |
| `/supermarket_scene/list_misplacements` | 无 | 数量与错放 `SlotPose[] slots` |
| `/supermarket_scene/take_slot` | `string slot_id` | 更新后的 `Slot`，状态为缺货 |
| `/supermarket_scene/restock_slot` | `string slot_id` | 更新后的 `Slot`，状态为正常 |
| `/supermarket_scene/update_slot` | slot ID、SKU 更新标志和 SKU 值 | 更新后的 `Slot` |
| `/supermarket_scene/create_shelf` | 名称、`map` 位姿、可选货架类型 | 新 `shelf_id` |
| `/supermarket_scene/update_shelf` | 货架 ID 与更新字段 | 成功或错误 |
| `/supermarket_scene/create_delivery_table` | 名称、`map` 位姿 | 新 `table_id` |
| `/supermarket_scene/update_delivery_table` | 交付桌 ID 与更新字段 | 成功或错误 |
| `/supermarket_scene/create_slot` | 固定位置、应放 / 实际 SKU、尺寸 | 生成的稳定 `Slot` |

所有服务响应都有 `bool success` 和 `string error`。以下章节定义每个输入字段与返回消息的完整结构。

## 消息定义

### `Slot.msg`

机器人只需要库存和固定位置字段，不包含 Web 图片目录、模型文件等管理字段。

```text
string slot_id
int32 shelf_id
int32 face
int32 level
float64 y_cm

string expected_sku
bool has_actual_sku
string actual_sku
string status
```

状态规则：

```text
has_actual_sku == false                 -> 缺货
actual_sku == expected_sku              -> 正常
has_actual_sku == true 且 SKU 不相等     -> 摆放错误
```

### `SlotPose.msg`

```text
Slot slot
string frame_id
float64 world_x
float64 world_y
float64 world_z
float64 shelf_yaw
```

`face` 和 `shelf_yaw` 都返回给机器人，但服务不猜测末端执行器的接近方向。运动规划器根据货架面、机器人自身姿态和安全距离计算接近姿态。

### `DetectedBox.msg`

```text
int32 index
string sku
int32 x
int32 y
int32 width
int32 height
bool has_confidence
float64 confidence
```

框坐标是输入图像像素坐标，格式为 `x`、`y`、`width`、`height`。没有 DINO 评分时 `has_confidence=false`。

### `InspectionSlot.msg`

```text
Slot slot
string source
string reason
bool has_confidence
float64 confidence
int32 x
int32 y
int32 width
int32 height
```

巡检只返回异常 slot；正常 slot 不出现在数组中。巡检框是配准后的有效重叠裁剪图坐标，不是原始照片坐标。

## 视觉服务

### `InspectShelf.srv`

```text
sensor_msgs/CompressedImage image
---
bool success
string error
int32 shelf_id
int32 face
supermarket_scene_interfaces/InspectionSlot[] anomalies
```

服务名：

```text
/supermarket_scene/inspect_shelf
```

固定使用生产模式：

```text
debug = false
```

不会保存调试图片、结果 JSON、VLM 原始文本或运行目录，也不会直接修改库存。机器人完成实际取放动作后，再调用库存写入服务。

### `FindSku.srv`

```text
sensor_msgs/CompressedImage image
string query
---
bool success
string error
string sku
string reference_slot_id  # 使用 SKU 主图时为空字符串
string provider
string model
string request_id
float64 request_seconds
float64 total_seconds
supermarket_scene_interfaces/DetectedBox[] boxes
```

服务名：

```text
/supermarket_scene/find_sku
```

`query` 可以是已有 SKU，也可以是已有 `slot_id`。如果 SKU 有 `data/sku_images/<SKU>.png`，两种查询都优先使用 SKU 主图；否则才回退到该位置或正常 slot 的 `data/item_images/{slot_id}/0.png`。返回的 `boxes` 是最终保留的框，不返回原始 VLM 文本和调试图片。节点参数 `provider=local` 时固定使用 `google/owlv2-large-patch14-ensemble` 和该 SKU 已审核的 `owlv2_prompt`；空提示词会返回 `success=false`。`sku_query.dino_fallback` 是节点级 DINO 保底开关，不是单次服务字段：关闭时直接返回 OWLv2 前 `sku_query.max_boxes` 个框，开启时由 DINO 对全部 OWLv2 候选排序后返回前 `max_boxes` 个。本地路径的 `DetectedBox.confidence` 优先返回 DINO 复核分数，否则返回 OWLv2 分数。

## 坐标与状态查询服务

### `GetSlotPose.srv`

```text
string slot_id
---
bool success
string error
supermarket_scene_interfaces/SlotPose slot
```

服务名：

```text
/supermarket_scene/get_slot_pose
```

### `GetSkuLocations.srv`

```text
string sku
---
bool success
string error
string sku
uint32 count
supermarket_scene_interfaces/SlotPose[] positions
```

服务名：

```text
/supermarket_scene/get_sku_locations
```

只返回 `actual_sku == sku` 的有货位置，不返回缺货的预期位置。

### `ListSlots.srv`

```text
---
bool success
string error
uint32 count
supermarket_scene_interfaces/SlotPose[] slots
```

使用同一个服务类型注册两个服务：

```text
/supermarket_scene/list_shortages
/supermarket_scene/list_misplacements
```

缺货和错放列表都返回 `SlotPose`，机器人不需要再次逐个查询世界坐标。

## 库存写入服务

### `TakeSlot.srv`

```text
string slot_id
---
bool success
string error
supermarket_scene_interfaces/Slot slot
```

服务名：

```text
/supermarket_scene/take_slot
```

物理取货成功后调用。服务将 `actual_sku` 置为空，固定位置不删除。

### `RestockSlot.srv`

```text
string slot_id
---
bool success
string error
supermarket_scene_interfaces/Slot slot
```

服务名：

```text
/supermarket_scene/restock_slot
```

物理补货成功后调用。服务将 `actual_sku` 设置为 `expected_sku`。

### `UpdateSlot.srv`

```text
string slot_id
bool update_expected_sku
string expected_sku
bool update_actual_sku
bool actual_sku_is_null
string actual_sku
---
bool success
string error
supermarket_scene_interfaces/Slot slot
```

服务名：

```text
/supermarket_scene/update_slot
```

当 `update_actual_sku=false` 时不修改实际 SKU；当 `update_actual_sku=true` 且 `actual_sku_is_null=true` 时设置为缺货；否则使用 `actual_sku` 字段。位置 ID 永远不能被修改。

所有库存写入完成后：

```text
SQLite 数据库提交
-> 更新状态
-> 同步 data/shelf_calibration/{shelf_id}.json
-> 返回更新后的 Slot
```

## 场景维护服务

这些服务主要用于初始化和维护，不是机器人每次取放货的运行路径。

### `CreateShelf.srv`

```text
string name
float64 world_x
float64 world_y
float64 yaw
bool has_shelf_type_id
int32 shelf_type_id
---
bool success
string error
int32 shelf_id
```

服务名：`/supermarket_scene/create_shelf`

### `UpdateShelf.srv`

```text
int32 shelf_id
bool update_name
string name
bool update_pose
float64 world_x
float64 world_y
float64 yaw
bool update_shelf_type_id
bool has_shelf_type_id
int32 shelf_type_id
---
bool success
string error
```

服务名：`/supermarket_scene/update_shelf`

### `CreateDeliveryTable.srv`

```text
string name
float64 world_x
float64 world_y
float64 yaw
---
bool success
string error
int32 table_id
```

服务名：`/supermarket_scene/create_delivery_table`

### `UpdateDeliveryTable.srv`

```text
int32 table_id
bool update_name
string name
bool update_pose
float64 world_x
float64 world_y
float64 yaw
---
bool success
string error
```

服务名：`/supermarket_scene/update_delivery_table`

### `CreateSlot.srv`

```text
int32 shelf_id
int32 face
int32 level
float64 y_cm
string expected_sku
bool has_actual_sku
string actual_sku
float64 width_cm
float64 height_cm
---
bool success
string error
supermarket_scene_interfaces/Slot slot
```

服务名：`/supermarket_scene/create_slot`。

服务端根据 `shelf_id`、`face`、`level`、`y_cm` 生成稳定 `slot_id`。首次录入已有商品时传 `has_actual_sku=true`；录入虚空缺货位置时传 `has_actual_sku=false`。

## ROS2 节点参数

服务请求不重复携带算法阈值，节点启动参数提供默认配置：

```text
api_url                         默认不需要，ROS 节点直接调用共享业务层
provider                        ark 或 local
model                           云端 VLM 模型；provider=local 时忽略
inspection.min_current_coverage 0.05
inspection.analysis_center_ratio 0.8
inspection.lab_distance_threshold 12.0
inspection.slot_change_ratio_threshold 0.15
inspection.dino_confidence_threshold 0.72
inspection.ambiguity_margin 0.05
inspection.vlm_fallback false
inspection.vlm_top_k 4
sku_query.max_boxes 1
sku_query.dino_fallback false（本地 OWLv2 候选是否使用 DINO 保底复核）
sku_query.dino_confidence_threshold 0.72
sku_query.owlv2_score_threshold 0.10
```

视觉服务如果失败，返回 `success=false` 和具体 `error`，不把异常伪装成缺货。只有视觉算法成功判断某个位置为缺货时，才在 `anomalies` 中返回“缺货”。

## 推荐机器人流程

### 补货

```text
list_shortages
-> 读取 SlotPose.slot.world_x/y/z、shelf_yaw、face
-> 机器人从供货区域取出 expected_sku
-> 机器人移动到固定位置
-> restock_slot
```

### 取货

```text
get_sku_locations
-> 选择一个 SlotPose
-> 机器人移动到 world_x/y/z
-> 机器人取走商品
-> take_slot
```

### 巡检

```text
inspect_shelf
-> 根据异常 slot_id 和 status 生成机器人任务
-> 完成实际取放
-> 调用 take_slot / restock_slot / update_slot
```

## 当前明确不包含

- ROS2 Action、任务取消和任务反馈。
- 图片拼接、人工录入和调试图显示服务。
- 机器人导航、机械臂规划和夹爪控制。
- ROS2 节点与 API 服务同时写数据库时的跨进程锁。
