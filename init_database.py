#!/usr/bin/env python3
"""
初始化货架数据库 - 填充SKU目录、货架组位置、商品库存

运行此脚本生成 shelf_inventory.db

Slot ID 格式: {shelf_id}-{face}-{level}-{y_cm}
    一个 slot = 一个商品实例, y_cm 精确到厘米
"""

import os
import numpy as np
from shelf_database import (
    ShelfDatabase, ShelfGroup, SkuInfo, ShelfType,
    DEFAULT_SHELF_LENGTH, DEFAULT_SHELF_WIDTH, DEFAULT_SHELF_HEIGHT,
    DEFAULT_NUM_LEVELS, DEFAULT_BOTTOM_CLEARANCE, DEFAULT_LEVEL_SPACING,
    DEFAULT_PANEL_THICK, DEFAULT_BACK_THICK,
    DEFAULT_SHELF_DEPTH_NORMAL, DEFAULT_SHELF_DEPTH_BOTTOM,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# SKU 目录 (19种商品)
# ============================================================

YCB_MODELS = {
    "cracker_box":       {"category": "food",
                          "mesh_file": "assets/ycb/asset/ycb_models/003_cracker_box/google_16k/textured.obj",
                          "tex_file": "assets/ycb/asset/ycb_models/003_cracker_box/google_16k/texture_map.png"},
    "sugar_box":         {"category": "food",
                          "mesh_file": "assets/ycb/asset/ycb_models/004_sugar_box/google_16k/textured.obj",
                          "tex_file": "assets/ycb/asset/ycb_models/004_sugar_box/google_16k/texture_map.png"},
    "tomato_soup_can":   {"category": "food",
                          "mesh_file": "assets/ycb/asset/ycb_models/005_tomato_soup_can/google_16k/textured.obj",
                          "tex_file": "assets/ycb/asset/ycb_models/005_tomato_soup_can/google_16k/texture_map.png"},
    "mustard_bottle":    {"category": "drink",
                          "mesh_file": "assets/ycb/asset/ycb_models/006_mustard_bottle/google_16k/textured.obj",
                          "tex_file": "assets/ycb/asset/ycb_models/006_mustard_bottle/google_16k/texture_map.png"},
    "tuna_fish_can":     {"category": "food",
                          "mesh_file": "assets/ycb/asset/ycb_models/007_tuna_fish_can/google_16k/textured.obj",
                          "tex_file": "assets/ycb/asset/ycb_models/007_tuna_fish_can/google_16k/texture_map.png"},
    "potted_meat_can":   {"category": "food",
                          "mesh_file": "assets/ycb/asset/ycb_models/010_potted_meat_can/google_16k/textured.obj",
                          "tex_file": "assets/ycb/asset/ycb_models/010_potted_meat_can/google_16k/texture_map.png"},
    "banana":            {"category": "food",
                          "mesh_file": "assets/ycb/asset/ycb_models/011_banana/google_16k/textured.obj",
                          "tex_file": "assets/ycb/asset/ycb_models/011_banana/google_16k/texture_map.png"},
    "pitcher_base":      {"category": "drink",
                          "mesh_file": "assets/ycb/asset/ycb_models/019_pitcher_base/google_16k/textured.obj",
                          "tex_file": "assets/ycb/asset/ycb_models/019_pitcher_base/google_16k/texture_map.png"},
    "bleach_cleanser":   {"category": "cleaning",
                          "mesh_file": "assets/ycb/asset/ycb_models/021_bleach_cleanser/google_16k/textured.obj",
                          "tex_file": "assets/ycb/asset/ycb_models/021_bleach_cleanser/google_16k/texture_map.png"},
    "mug":               {"category": "kitchen",
                          "mesh_file": "assets/ycb/asset/ycb_models/025_mug/google_16k/textured.obj",
                          "tex_file": "assets/ycb/asset/ycb_models/025_mug/google_16k/texture_map.png"},
    "large_marker":      {"category": "cleaning",
                          "mesh_file": "assets/ycb/asset/ycb_models/040_large_marker/google_16k/textured.obj",
                          "tex_file": "assets/ycb/asset/ycb_models/040_large_marker/google_16k/texture_map.png"},
}

SCANNED_MODELS = {
    "coffee_mug":           {"category": "kitchen"},
    "fruit_snacks_grape":   {"category": "food"},
    "brisk_tea":            {"category": "drink"},
    "milk_frother":         {"category": "kitchen"},
    "porcelain_bowl":       {"category": "kitchen"},
    "plastic_bowl":         {"category": "kitchen"},
    "elderberry_syrup":     {"category": "drink"},
    "fruit_snacks_juicy":   {"category": "food"},
}

ALL_SKUS = {}
for k, v in YCB_MODELS.items():
    ALL_SKUS[k] = v
for k, v in SCANNED_MODELS.items():
    ALL_SKUS[k] = v


# ============================================================
# 货架组位置 (与 generate_scene.py 一致)
# ============================================================
# 现有场景货架中心X = ±1.1, 中心Y = ±(0.93+0.25) = ±1.18
# 货架局部原点 = 中心 - (DEFAULT_SHELF_WIDTH/2, DEFAULT_SHELF_LENGTH/2) = 中心 - (0.40, 0.93)
#
#   世界坐标系俯视图:
#       Y=4 (上)
#   ┌──────────────────────┐
#   │     货架1     货架2   │
#   │                      │
#   │  ──── 警戒线 ────    │ Y=0
#   │                      │
#   │     货架3     货架4   │
#   └──────────────────────┘
#       Y=-4 (下)

SHELF_CENTERS = [
    ("1号货架", -1.1,  1.18),   # ID 1
    ("2号货架",  1.1,  1.18),   # ID 2
    ("3号货架", -1.1, -1.18),   # ID 3
    ("4号货架",  1.1, -1.18),   # ID 4
]

# 货架局部原点坐标 (name, world_x, world_y, yaw)
SHELF_ORIGINS = [
    (name, cx - DEFAULT_SHELF_WIDTH/2, cy - DEFAULT_SHELF_LENGTH/2, 0.0)
    for name, cx, cy in SHELF_CENTERS
]


# ============================================================
# 商品分类 (用于合理分配)
# ============================================================

FOOD_SKUS = ["cracker_box", "sugar_box", "tomato_soup_can", "tuna_fish_can",
             "potted_meat_can", "banana", "fruit_snacks_grape", "fruit_snacks_juicy"]
DRINK_SKUS = ["mustard_bottle", "pitcher_base", "mug", "coffee_mug",
              "milk_frother", "elderberry_syrup", "brisk_tea"]
CLEANING_SKUS = ["bleach_cleanser", "large_marker"]
KITCHEN_SKUS = ["porcelain_bowl", "plastic_bowl"]


# ============================================================
# 主初始化函数
# ============================================================

def init_database(db_path: str):
    """初始化数据库，填充所有数据"""
    # 删除旧数据库
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"已删除旧数据库: {db_path}")

    db = ShelfDatabase(db_path)

    # 0. 创建货架类型
    print("创建货架类型...")
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
    print(f"  已创建: standard (id={type_id})")
    st = db.get_shelf_type(type_id)
    print(f"  参数: {st.shelf_length:.2f}x{st.shelf_width:.2f}x{st.shelf_height:.2f}m, "
          f"{st.num_levels}层")

    # 1. 注册所有 SKU
    print("注册 SKU 目录...")
    sku_data = []
    for sku_name, info in ALL_SKUS.items():
        sku_data.append({
            "sku": sku_name,
            "category": info["category"],
            "mesh_file": info.get("mesh_file", ""),
            "tex_file": info.get("tex_file", ""),
        })
    db.register_skus_batch(sku_data)
    print(f"  已注册 {len(sku_data)} 个 SKU")

    # 2. 添加货架组
    print("添加货架组...")
    shelf_ids = []
    for name, wx, wy, yaw in SHELF_ORIGINS:
        sid = db.add_shelf_group(name=name, world_x=wx, world_y=wy,
                                  yaw=yaw, shelf_type_id=type_id)
        shelf_ids.append(sid)
        print(f"  ID={sid}: {name} 局部原点=({wx:.3f}, {wy:.3f}) type=standard")

    # 3. 填充商品库存
    #    策略: 在货架长度范围内随机放置商品, 间距约 10-15cm
    #    一个 slot = 一个商品实例, y_cm 精确到厘米；商品高度决定中心高度
    print("填充商品库存...")

    # 层 -> 品类映射
    level_categories = {
        0: FOOD_SKUS + DRINK_SKUS,           # 底层: 食品+饮料 (重物在下)
        1: FOOD_SKUS,                         # 一层: 食品
        2: FOOD_SKUS + DRINK_SKUS,            # 二层: 食品+饮料
        3: DRINK_SKUS + KITCHEN_SKUS,         # 三层: 饮料+厨具
        4: KITCHEN_SKUS + CLEANING_SKUS,     # 四层: 厨具+清洁 (轻物在上)
    }

    np.random.seed(42)
    total_items = 0
    shelf_length_cm = int(DEFAULT_SHELF_LENGTH * 100)  # 货架长度, 单位厘米

    for sid in shelf_ids:
        for face in [0, 1]:
            for level in range(DEFAULT_NUM_LEVELS):
                candidate_skus = level_categories[level]
                # 从 5cm 开始, 每隔 10-20cm 放置一个商品, 避免重叠
                y_pos = 5  # 起始位置 (cm)
                while y_pos < shelf_length_cm - 10:
                    if np.random.random() > 0.5:
                        sku = str(np.random.choice(candidate_skus))
                        height_cm = float(np.random.randint(4, 21))  # 4-20cm 随机商品高度
                        db.set_slot(sid, face=face, level=level,
                                    y_cm=float(y_pos), sku=sku, height_cm=height_cm)
                        total_items += 1
                    y_pos += np.random.randint(10, 21)  # 10-20cm 间距

    print(f"  已填充 {total_items} 个商品")

    # 4. 验证数据
    print("\n" + "=" * 60)
    stats = db.get_stats()
    print("数据库统计:")
    print(f"  货架类型数: {stats['shelf_types']}")
    print(f"  货架组数量: {stats['shelf_groups']}")
    print(f"  SKU种类数:  {stats['sku_catalog']}")
    print(f"  商品总数量: {stats['total_items']}")

    # 打印每个货架的摘要
    print("\n各货架商品摘要:")
    for sid in shelf_ids:
        summary = db.get_shelf_sku_summary(sid)
        shelf = db.get_shelf_group(sid)
        total = sum(s["total_quantity"] for s in summary)
        print(f"  货架{sid} ({shelf.name}): {len(summary)}种SKU, {total}个商品")

    # 打印几个 SKU 的分布
    print("\nSKU 分布示例:")
    for test_sku in ["cracker_box", "banana", "bleach_cleanser"]:
        qty = db.get_sku_total_quantity(test_sku)
        locs = db.find_sku_locations(test_sku)
        print(f"  {test_sku}: 共{qty}个, 分布在{len(locs)}个槽位")

    db.close()
    print(f"\n✅ 数据库已保存: {db_path}")
    return db_path


if __name__ == "__main__":
    db_path = os.path.join(BASE_DIR, "shelf_inventory.db")
    init_database(db_path)
