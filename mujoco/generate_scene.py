#!/usr/bin/env python3
"""
超市场景生成器 - 比赛场地版本
6m x 8m，竖排货架，真实3D Mesh (YCB + Scanned Objects)
"""

import os
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom

from scene_geometry import (
    DELIVERY_TABLE_HEIGHT as COUNTER_HEIGHT,
    DELIVERY_TABLE_LENGTH as COUNTER_LENGTH,
    DELIVERY_TABLE_TOP_THICKNESS as COUNTER_THICK,
    DELIVERY_TABLE_WIDTH as COUNTER_WIDTH,
)

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YCB_DIR = os.path.join(BASE_DIR, "assets/ycb/asset/ycb_models")
SCANNED_DIR = os.path.join(BASE_DIR, "assets/scanned/models")

# ============================================================
# 全局参数 (单位：米)
# ============================================================
MAP_WIDTH = 6.0   # X 方向
MAP_LENGTH = 8.0  # Y 方向

# 货架参数（双面货架，中间背板）
# 比赛图尺寸：货架长 1.86m (沿 Y), 宽 0.80m (沿 X), 高 165cm
# 两个拼接：长 1.86m (沿 Y)
SHELF_UNIT_Y = 0.93   # 单组货架长度（沿 Y）
SHELF_UNIT_X = 0.80   # 货架宽度（沿 X）
SHELF_HEIGHT = 1.65
SHELF_DEPTH_NORMAL = 0.30   # 层板深度 30cm（每面）
SHELF_DEPTH_BOTTOM = 0.40   # 最下层深度 40cm（每面）
SHELF_PANEL_THICK = 0.02    # 普通层板厚度
SHELF_BOTTOM_THICK = 0.02   # 最底层厚度同其他层
SHELF_BACK_THICK = 0.005    # 中间背板厚度 5mm
NUM_LEVELS = 5
BOTTOM_CLEARANCE = 0.05     # 最底层离地高度 5cm
LEVEL_SPACING = (SHELF_HEIGHT - BOTTOM_CLEARANCE) / (NUM_LEVELS - 1)  # 层间距 = 160/4 = 40cm

SHELFS_PER_GROUP = 2
# 货架组在 X 方向：双面各30cm + 背板 = 60.5cm 深（最下层40cm + 背板 + 40cm）
GROUP_DEPTH = SHELF_DEPTH_BOTTOM * 2 + SHELF_BACK_THICK  # 0.805m 最深
GROUP_Y_LEN = SHELF_UNIT_Y * SHELFS_PER_GROUP  # 1.86m 沿Y

# 购物篮
BASKET_BOTTOM_LENGTH = 0.33
BASKET_BOTTOM_WIDTH = 0.22
BASKET_HEIGHT = 0.21
BASKET_THICK = 0.008
BASKET_HANDLE_RADIUS = 0.003

# 外卖清单
RECEIPT_LENGTH = 0.15
RECEIPT_WIDTH = 0.10
RECEIPT_THICK = 0.002

FLOOR_SIZE = (MAP_WIDTH, MAP_LENGTH, 0.02)

# 围墙参数 - 在6x8场地外侧，不侵占内部空间
WALL_HEIGHT = 0.25    # 围墙高度 25cm
WALL_THICK = 0.05     # 围墙厚度 5cm
WALL_INNER_X = MAP_WIDTH / 2   # 围墙内径半宽 = 3.0m
WALL_INNER_Y = MAP_LENGTH / 2  # 围墙内径半长 = 4.0m
WALL_CENTER_X = WALL_INNER_X + WALL_THICK / 2   # 围墙中心X = 3.025m
WALL_CENTER_Y = WALL_INNER_Y + WALL_THICK / 2   # 围墙中心Y = 4.025m

# 场地布局（根据参考图）
# 场地宽6m(X方向)：左边界1.5m | 货架0.8m | 间距1.4m | 货架0.8m | 右边界1.5m = 6m
# 货架宽0.80m（沿X方向），所以货架中心X = ±(1.5 + 0.4) = ±1.9m
# 警戒线(y=0)将场地分为上下两半，货架在警戒线两侧，上下货架之间不接壤（有过道间隙）
# 货架长1.86m（沿Y方向，两拼接），上下货架中心间距 = 1.86 + 过道间隙
SHELF_LEFT_X = -(MAP_WIDTH/2 - 1.5 - SHELF_UNIT_X/2)   # 左货架中心X = -1.1m
SHELF_RIGHT_X = MAP_WIDTH/2 - 1.5 - SHELF_UNIT_X/2       # 右货架中心X = 1.1m
AISLE_GAP = 0.5        # 上下货架之间的过道间隙
SHELF_TOP_Y = GROUP_Y_LEN/2 + AISLE_GAP/2      # 上半区货架中心
SHELF_BOTTOM_Y = -(GROUP_Y_LEN/2 + AISLE_GAP/2)  # 下半区货架中心

# 交付台/补货台位置（四个角落）
COUNTER_TOP_LEFT_POS = (-2.4, 3.6, COUNTER_HEIGHT / 2)
COUNTER_TOP_RIGHT_POS = (2.4, 3.6, COUNTER_HEIGHT / 2)
COUNTER_BOTTOM_LEFT_POS = (-2.4, -3.6, COUNTER_HEIGHT / 2)
COUNTER_BOTTOM_RIGHT_POS = (2.4, -3.6, COUNTER_HEIGHT / 2)

# 起点区域（中间小圆形）
START_ZONE_TOP_POS = (0, 3.6, 0.01)
START_ZONE_BOTTOM_POS = (0, -3.6, 0.01)
START_ZONE_RADIUS = 0.3

# 中间分割线
DIVIDER_Y = 0.0

# ============================================================
# 真实模型定义
# ============================================================

YCB_MODELS = {
    "cracker_box":       {"obj": "assets/ycb/asset/ycb_models/003_cracker_box/google_16k/textured.obj",
                          "tex": "assets/ycb/asset/ycb_models/003_cracker_box/google_16k/texture_map.png",
                          "scale": 1.0},
    "sugar_box":         {"obj": "assets/ycb/asset/ycb_models/004_sugar_box/google_16k/textured.obj",
                          "tex": "assets/ycb/asset/ycb_models/004_sugar_box/google_16k/texture_map.png",
                          "scale": 1.0},
    "tomato_soup_can":   {"obj": "assets/ycb/asset/ycb_models/005_tomato_soup_can/google_16k/textured.obj",
                          "tex": "assets/ycb/asset/ycb_models/005_tomato_soup_can/google_16k/texture_map.png",
                          "scale": 1.0},
    "mustard_bottle":    {"obj": "assets/ycb/asset/ycb_models/006_mustard_bottle/google_16k/textured.obj",
                          "tex": "assets/ycb/asset/ycb_models/006_mustard_bottle/google_16k/texture_map.png",
                          "scale": 1.0},
    "tuna_fish_can":     {"obj": "assets/ycb/asset/ycb_models/007_tuna_fish_can/google_16k/textured.obj",
                          "tex": "assets/ycb/asset/ycb_models/007_tuna_fish_can/google_16k/texture_map.png",
                          "scale": 1.0},
    "potted_meat_can":   {"obj": "assets/ycb/asset/ycb_models/010_potted_meat_can/google_16k/textured.obj",
                          "tex": "assets/ycb/asset/ycb_models/010_potted_meat_can/google_16k/texture_map.png",
                          "scale": 1.0},
    "banana":            {"obj": "assets/ycb/asset/ycb_models/011_banana/google_16k/textured.obj",
                          "tex": "assets/ycb/asset/ycb_models/011_banana/google_16k/texture_map.png",
                          "scale": 1.0},
    "pitcher_base":      {"obj": "assets/ycb/asset/ycb_models/019_pitcher_base/google_16k/textured.obj",
                          "tex": "assets/ycb/asset/ycb_models/019_pitcher_base/google_16k/texture_map.png",
                          "scale": 1.0},
    "bleach_cleanser":   {"obj": "assets/ycb/asset/ycb_models/021_bleach_cleanser/google_16k/textured.obj",
                          "tex": "assets/ycb/asset/ycb_models/021_bleach_cleanser/google_16k/texture_map.png",
                          "scale": 1.0},
    "mug":               {"obj": "assets/ycb/asset/ycb_models/025_mug/google_16k/textured.obj",
                          "tex": "assets/ycb/asset/ycb_models/025_mug/google_16k/texture_map.png",
                          "scale": 1.0},
    "large_marker":      {"obj": "assets/ycb/asset/ycb_models/040_large_marker/google_16k/textured.obj",
                          "tex": "assets/ycb/asset/ycb_models/040_large_marker/google_16k/texture_map.png",
                          "scale": 1.0},
}

SCANNED_MODELS = {
    "coffee_mug":           {"dir": "ACE_Coffee_Mug_Kristen_16_oz_cup"},
    "fruit_snacks_grape":   {"dir": "Black_Forest_Fruit_Snacks_28_Pack_Grape"},
    "brisk_tea":            {"dir": "Brisk_Iced_Tea_Lemon_12_12_fl_oz_355_ml_cans_144_fl_oz_426_lt"},
    "milk_frother":         {"dir": "Aroma_Stainless_Steel_Milk_Frother_2_Cup"},
    "porcelain_bowl":       {"dir": "BIA_Porcelain_Ramekin_With_Glazed_Rim_35_45_oz_cup"},
    "plastic_bowl":         {"dir": "Bradshaw_International_11642_7_Qt_MP_Plastic_Bowl"},
    "elderberry_syrup":     {"dir": "Black_Elderberry_Syrup_54_oz_Gaia_Herbs"},
    "fruit_snacks_juicy":   {"dir": "Black_Forest_Fruit_Snacks_Juicy_Filled_Centers_10_pouches_9_oz_total"},
}

ALL_MODELS = {}
for k, v in YCB_MODELS.items():
    ALL_MODELS[k] = {"type": "ycb", **v}
for k, v in SCANNED_MODELS.items():
    d = v["dir"]
    ALL_MODELS[k] = {
        "type": "scanned",
        "obj": f"{SCANNED_DIR}/{d}/model.obj",
        "tex": f"{SCANNED_DIR}/{d}/texture.png",
        "scale": 1.0,
    }

FOOD_MODELS = ["cracker_box", "sugar_box", "tomato_soup_can", "tuna_fish_can",
               "potted_meat_can", "banana", "fruit_snacks_grape", "fruit_snacks_juicy"]
DRINK_MODELS = ["mustard_bottle", "pitcher_base", "mug", "coffee_mug", "milk_frother",
                "elderberry_syrup", "brisk_tea"]
CLEANING_MODELS = ["bleach_cleanser", "large_marker"]
KITCHEN_MODELS = ["porcelain_bowl", "plastic_bowl"]

# ============================================================
# XML 构建辅助
# ============================================================

def add_geom(parent, name, pos, size, material="", rgba="", geom_type="box", euler=""):
    attrs = {"name": name, "pos": fmt_vec(pos), "type": geom_type}
    if geom_type == "box":
        attrs["size"] = fmt_vec(size)
    elif geom_type == "cylinder":
        attrs["size"] = f"{size[0]} {size[1]}"
    elif geom_type == "sphere":
        attrs["size"] = str(size)
    if material:
        attrs["material"] = material
    if rgba:
        attrs["rgba"] = rgba
    if euler:
        attrs["euler"] = euler
    ET.SubElement(parent, "geom", attrs)

def fmt_vec(v):
    return f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f}"

def make_body(parent, name, pos="0 0 0", euler=""):
    attrs = {"name": name, "pos": pos}
    if euler:
        attrs["euler"] = euler
    return ET.SubElement(parent, "body", attrs)

def add_site(parent, name, pos, size="0.02", rgba="0 0 0 0"):
    ET.SubElement(parent, "site", {
        "name": name, "pos": fmt_vec(pos), "size": str(size),
        "rgba": rgba, "type": "sphere",
    })

# ============================================================
# Assets: 纹理 & Mesh
# ============================================================

def create_assets():
    assets = ET.Element("asset")

    # 地面棋盘格
    ET.SubElement(assets, "texture", {
        "name": "floor_tex", "type": "2d", "builtin": "checker",
        "width": "256", "height": "256",
        "rgb1": "0.85 0.82 0.78", "rgb2": "0.75 0.72 0.68",
    })

    # 材质
    materials = {
        "floor_mat":       ("floor_tex", "0.05", "12 16"),
        "shelf_mat":       ("", "0.3", ""),
        "shelf_layer_mat": ("", "0.3", ""),  # 红色层板
        "frame_mat":       ("", "0.4", ""),
        "counter_mat":     ("", "0.15", ""),
        "counter_top_mat": ("", "0.15", ""),  # 绿色台面
        "basket_mat":      ("", "0.1", ""),
        "receipt_mat":     ("", "0.05", ""),
        "boundary_mat":    ("", "0.1", ""),
        "divider_mat":     ("", "0.1", ""),  # 黄黑警示线
        "start_zone_mat":  ("", "0.05", ""),
        "wall_mat":        ("", "0.2", ""),  # 围墙
        "shelf_side_red":  ("", "0.3", ""),  # 搁板侧面红
        "shelf_side_black":("", "0.3", ""),  # 搁板侧面黑
    }
    mat_rgba = {
        "floor_mat": "0.85 0.82 0.78 1.0",
        "shelf_mat": "0.15 0.15 0.18 1.0",      # 深色货架主体
        "shelf_layer_mat": "0.75 0.15 0.15 1.0",  # 红色层板
        "frame_mat": "0.25 0.25 0.30 1.0",
        "counter_mat": "0.50 0.50 0.52 1.0",    # 银色桌腿
        "counter_top_mat": "0.05 0.50 0.25 1.0",  # 绿色台面
        "basket_mat": "0.15 0.25 0.55 1.0",
        "receipt_mat": "0.98 0.98 0.95 1.0",
        "boundary_mat": "0.9 0.9 0.9 1.0",
        "divider_mat": "1.0 0.85 0.0 1.0",       # 黄色警示线
        "start_zone_mat": "0.0 0.7 0.2 0.8",
        "wall_mat": "1.0 1.0 1.0 1.0",            # 围墙白色不透明
        "shelf_side_red": "0.75 0.15 0.15 1.0",   # 搁板过道侧面红色
        "shelf_side_black": "0.08 0.08 0.08 1.0",  # 搁板其他面黑色
    }
    for name, (tex, refl, repeat) in materials.items():
        attrs = {"name": name, "rgba": mat_rgba[name], "reflectance": refl}
        if tex:
            attrs["texture"] = tex
            attrs["texrepeat"] = repeat
        ET.SubElement(assets, "material", attrs)

    # YCB Mesh + Texture
    for model_name, info in YCB_MODELS.items():
        tex_name = f"tex_{model_name}"
        mesh_name = f"mesh_{model_name}"
        mat_name = f"mat_{model_name}"
        ET.SubElement(assets, "texture", {
            "name": tex_name, "type": "2d",
            "file": info["tex"],
        })
        ET.SubElement(assets, "material", {
            "name": mat_name, "texture": tex_name,
            "specular": "0.3", "shininess": "0.3",
        })
        ET.SubElement(assets, "mesh", {
            "name": mesh_name,
            "file": info["obj"],
        })

    # Scanned Objects Mesh + Texture
    for model_name, info in SCANNED_MODELS.items():
        d = info["dir"]
        tex_name = f"tex_{model_name}"
        mesh_name = f"mesh_{model_name}"
        mat_name = f"mat_{model_name}"
        ET.SubElement(assets, "texture", {
            "name": tex_name, "type": "2d",
            "file": f"assets/scanned/models/{d}/texture.png",
        })
        ET.SubElement(assets, "material", {
            "name": mat_name, "texture": tex_name,
            "specular": "0.5", "shininess": "0.5",
        })
        ET.SubElement(assets, "mesh", {
            "name": mesh_name,
            "file": f"assets/scanned/models/{d}/model.obj",
        })

    return assets


# ============================================================
# 场景构建函数
# ============================================================

def build_floor(worldbody):
    body = make_body(worldbody, "floor", pos=fmt_vec((0, 0, -FLOOR_SIZE[2]/2)))
    add_geom(body, "floor_geom", (0, 0, 0),
             (FLOOR_SIZE[0]/2, FLOOR_SIZE[1]/2, FLOOR_SIZE[2]/2), material="floor_mat")

def build_divider_line(worldbody):
    """中间黄黑相间警示线 - 贯穿整个场地宽度"""
    # 主黄线 - 从场地左边界到右边界
    body = make_body(worldbody, "divider_line", pos=fmt_vec((0, DIVIDER_Y, 0.001)))
    add_geom(body, "divider_geom", (0, 0, 0),
             (MAP_WIDTH/2, 0.03, 0.001), material="divider_mat")
    # 黑色虚线效果（用多个小段）
    n_dashes = 30
    dash_len = MAP_WIDTH / n_dashes
    for i in range(n_dashes):
        if i % 2 == 1:
            x = -MAP_WIDTH/2 + dash_len * (i + 0.5)
            db = make_body(worldbody, f"divider_black_{i}", pos=fmt_vec((x, DIVIDER_Y, 0.002)))
            add_geom(db, f"divider_black_{i}_geom", (0, 0, 0),
                     (dash_len/2 * 0.8, 0.03, 0.001), rgba="0.1 0.1 0.1 1.0")

def build_start_zones(worldbody):
    """起点区域 - 地面圆形标记（平躺在地面上）"""
    for prefix, pos in [("start_zone_top", START_ZONE_TOP_POS),
                        ("start_zone_bottom", START_ZONE_BOTTOM_POS)]:
        body = make_body(worldbody, prefix, pos=fmt_vec(pos))
        # 扁平圆柱体，平躺在地面（不加旋转）
        add_geom(body, f"{prefix}_geom", (0, 0, 0),
                 (START_ZONE_RADIUS, 0.002),
                 material="start_zone_mat", geom_type="cylinder")
        add_site(body, f"{prefix}_site", (0, 0, 0.05), size="0.05", rgba="0 0 0 0")

def build_boundary_lines(worldbody):
    fl = FLOOR_SIZE
    lines = [
        ("boundary_top",    (0, fl[1]/2, 0.001), (fl[0]/2, 0.01, 0.001)),
        ("boundary_bottom", (0, -fl[1]/2, 0.001), (fl[0]/2, 0.01, 0.001)),
        ("boundary_left",   (-fl[0]/2, 0, 0.001), (0.01, fl[1]/2, 0.001)),
        ("boundary_right",  (fl[0]/2, 0, 0.001), (0.01, fl[1]/2, 0.001)),
    ]
    for name, pos, size in lines:
        body = make_body(worldbody, name, pos=fmt_vec(pos))
        add_geom(body, f"{name}_geom", (0, 0, 0), size, material="boundary_mat")

def build_walls(worldbody):
    """场地四周0.25m高围挡，落在6x8场地外侧，内径=6x8"""
    hh = WALL_HEIGHT / 2  # 围墙半高 = 0.125m
    ht = WALL_THICK / 2   # 围墙半厚 = 0.025m
    
    # 围墙盒子：(半X, 半Y, 半Z)，中心在场地边缘外侧
    # 上围墙：沿X方向长条，中心Y=WALL_CENTER_Y，X半长=内径半宽+半厚，Y半长=半厚
    # 下围墙：沿X方向长条，中心Y=-WALL_CENTER_Y
    # 左围墙：沿Y方向长条，中心X=-WALL_CENTER_X，Y半长=内径半长+半厚，X半长=半厚
    # 右围墙：沿Y方向长条，中心X=WALL_CENTER_X
    walls = [
        ("wall_top",    0,  WALL_CENTER_Y,   WALL_INNER_X + ht, ht, hh),
        ("wall_bottom", 0, -WALL_CENTER_Y,   WALL_INNER_X + ht, ht, hh),
        ("wall_left",  -WALL_CENTER_X, 0,  ht, WALL_INNER_Y + ht, hh),
        ("wall_right",  WALL_CENTER_X, 0,  ht, WALL_INNER_Y + ht, hh),
    ]
    
    for name, cx, cy, sx, sy, sh in walls:
        body = make_body(worldbody, name, pos=fmt_vec((cx, cy, hh)))
        add_geom(body, f"{name}_geom", (0, 0, 0),
                 (sx, sy, sh), material="wall_mat")

def build_shelf_group(worldbody, group_name, base_x, base_y, base_z):
    """构建双面货架组（中间背板，两面层板）
    
    规格：
    - 双面，中间有背板（从地面到165cm）
    - 4层板：最底层离地5cm，深度40cm；上面3层深度30cm
    - 最上层板上表面=165cm高
    - 4层均匀分布（间隔相等）
    - 无立柱
    - 总高165cm，宽160cm（两拼接）
    
    结构（沿X方向）：
    -X侧层板(30/40cm) | 中间背板(5mm) | +X侧层板(30/40cm)
    """
    group_body = make_body(worldbody, group_name, pos=fmt_vec((base_x, base_y, base_z)))

    half_y = GROUP_Y_LEN / 2   # 0.80 (沿Y)
    panel_thick = SHELF_PANEL_THICK
    prefix = group_name + "_"

    # 中间背板：薄板，从地面到165cm，宽160cm，厚5mm
    # 背板中心在 z = SHELF_HEIGHT/2，高度为 SHELF_HEIGHT
    back_panel = make_body(group_body, f"{prefix}back_panel", pos=fmt_vec((0, 0, SHELF_HEIGHT/2)))
    add_geom(back_panel, f"{prefix}back_geom", (0, 0, 0),
             (SHELF_BACK_THICK/2, half_y, SHELF_HEIGHT/2), material="shelf_mat")

    # 4层板（均匀分布）
    # z位置：最底层离地5cm，最上层上表面=165cm
    for level in range(NUM_LEVELS):
        z = BOTTOM_CLEARANCE + level * LEVEL_SPACING
        
        # 层板深度：最底层40cm，其他30cm
        if level == 0:
            depth = SHELF_DEPTH_BOTTOM  # 40cm
        else:
            depth = SHELF_DEPTH_NORMAL  # 30cm
        
        shelf_body = make_body(group_body, f"{prefix}shelf_level_{level}", pos=fmt_vec((0, 0, z)))
        
        # -X 侧层板：主体黑色，过道侧(-X方向)贴红色面
        shelf_cx_neg = -(SHELF_BACK_THICK/2 + depth/2)
        # 黑色主体
        add_geom(shelf_body, f"{prefix}shelf_L{level}_neg_body",
                 (shelf_cx_neg, 0, 0), 
                 (depth/2, half_y, panel_thick/2), 
                 material="shelf_side_black")
        # 过道侧红色面（在-X最外侧）
        add_geom(shelf_body, f"{prefix}shelf_L{level}_neg_face",
                 (shelf_cx_neg - depth/2 + 0.001, 0, 0), 
                 (0.002, half_y, panel_thick/2), 
                 material="shelf_side_red")

        # +X 侧层板：主体黑色，过道侧(+X方向)贴红色面
        shelf_cx_pos = (SHELF_BACK_THICK/2 + depth/2)
        # 黑色主体
        add_geom(shelf_body, f"{prefix}shelf_L{level}_pos_body",
                 (shelf_cx_pos, 0, 0), 
                 (depth/2, half_y, panel_thick/2), 
                 material="shelf_side_black")
        # 过道侧红色面（在+X最外侧）
        add_geom(shelf_body, f"{prefix}shelf_L{level}_pos_face",
                 (shelf_cx_pos + depth/2 - 0.001, 0, 0), 
                 (0.002, half_y, panel_thick/2), 
                 material="shelf_side_red")

        # 导航 sites（不可见，在层板前端）
        add_site(group_body, f"{group_name}_L{level}_face_neg", 
                 (shelf_cx_neg - depth/2 - 0.05, 0, z), 
                 size="0.03", rgba="0 0 0 0")
        add_site(group_body, f"{group_name}_L{level}_face_pos", 
                 (shelf_cx_pos + depth/2 + 0.05, 0, z), 
                 size="0.03", rgba="0 0 0 0")

    return group_body


def place_mesh_product(parent, model_name, pos, euler, unique_id, scale=1.0):
    """放置真实mesh模型（作为独立body带free joint，碰撞体使用原始mesh）"""
    info = ALL_MODELS.get(model_name)
    if not info:
        return None
    body_name = f"product_{model_name}_uid{unique_id}"
    body = make_body(parent, body_name, pos=fmt_vec(pos))
    if euler != "0 0 0":
        body.set("euler", euler)

    # 添加 free joint 使商品成为独立可移动物体
    ET.SubElement(body, "freejoint", {"name": f"{body_name}_joint"})

    # 视觉几何体（无碰撞，仅渲染）
    ET.SubElement(body, "geom", {
        "name": f"{body_name}_visual",
        "type": "mesh",
        "mesh": f"mesh_{model_name}",
        "material": f"mat_{model_name}",
        "contype": "0",
        "conaffinity": "0",
        "group": "2",
    })

    # 碰撞几何体：直接使用原始 mesh 形状，不可见，参与物理碰撞
    ET.SubElement(body, "geom", {
        "name": f"{body_name}_collision",
        "type": "mesh",
        "mesh": f"mesh_{model_name}",
        "contype": "1",
        "conaffinity": "1",
        "rgba": "0 0 0 0",
        "group": "3",
    })
    return body


def place_products_on_shelf(worldbody, group_x, group_y, group_z, group_index, group_name, global_pid=0):
    """在双面货架上放置真实mesh商品（独立body+free joint，有碰撞体积）
    
    worldbody: 直接挂在worldbody下
    group_x, group_y, group_z: 货架组在世界坐标系中的位置
    """
    half_y = GROUP_Y_LEN / 2   # 0.93
    panel_thick = SHELF_PANEL_THICK
    bottom_thick = SHELF_BOTTOM_THICK

    np.random.seed(42 + group_index)
    product_id = global_pid
    all_product_models = FOOD_MODELS + DRINK_MODELS + CLEANING_MODELS + KITCHEN_MODELS

    for level in range(NUM_LEVELS):
        # 商品初始 z：层板顶部 + 偏移量（让碰撞体底在层板上方，模拟后自然落稳）
        z_local = BOTTOM_CLEARANCE + level * LEVEL_SPACING + panel_thick/2 + 0.15
        z_world = group_z + z_local
        
        if level == 0:
            depth = SHELF_DEPTH_BOTTOM  # 40cm
        else:
            depth = SHELF_DEPTH_NORMAL  # 30cm

        # 层板中心 x 坐标（相对于货架中心）
        cx_neg_local = -(SHELF_BACK_THICK/2 + depth/2)
        cx_pos_local = (SHELF_BACK_THICK/2 + depth/2)

        # -X 侧商品（减少数量，mesh碰撞）
        num_items = np.random.randint(2, 4)
        y_positions = np.linspace(-half_y + 0.2, half_y - 0.2, num_items)
        for i, y_local in enumerate(y_positions):
            model_name = all_product_models[(level + i + group_index * 3) % len(all_product_models)]
            jitter_y = np.random.uniform(-0.02, 0.02)
            x_local = cx_neg_local + np.random.uniform(-depth/2 + 0.08, depth/2 - 0.08)
            # 转为世界坐标
            wx = group_x + x_local
            wy = group_y + y_local + jitter_y
            wz = z_world
            place_mesh_product(worldbody, model_name, (wx, wy, wz), "0 0 0", product_id)
            product_id += 1

        # +X 侧商品（减少数量，mesh碰撞）
        num_items = np.random.randint(2, 4)
        y_positions = np.linspace(-half_y + 0.2, half_y - 0.2, num_items)
        for i, y_local in enumerate(y_positions):
            model_name = all_product_models[(level + i + group_index * 5 + 1) % len(all_product_models)]
            jitter_y = np.random.uniform(-0.02, 0.02)
            x_local = cx_pos_local + np.random.uniform(-depth/2 + 0.08, depth/2 - 0.08)
            # 转为世界坐标
            wx = group_x + x_local
            wy = group_y + y_local + jitter_y
            wz = z_world
            place_mesh_product(worldbody, model_name, (wx, wy, wz), "0 0 0", product_id)
            product_id += 1

    return product_id - global_pid, product_id


def build_counter(worldbody, name, pos):
    """长方形工作台，绿色台面，银色金属腿，无抽屉无隔层"""
    counter = make_body(worldbody, name, pos=fmt_vec(pos))
    cl2 = COUNTER_LENGTH / 2
    cw2 = COUNTER_WIDTH / 2
    ch2 = COUNTER_HEIGHT / 2
    ct = COUNTER_THICK

    # 绿色台面
    add_geom(counter, f"{name}_top", (0, 0, ch2), (cl2, cw2, ct/2), material="counter_top_mat")

    # 四条银色金属腿（参考图：桌腿在四角）
    for i, (lx, ly) in enumerate([
        (-cl2+0.03, -cw2+0.03), (-cl2+0.03, cw2-0.03),
        (cl2-0.03, -cw2+0.03), (cl2-0.03, cw2-0.03),
    ]):
        leg = make_body(counter, f"{name}_leg_{i}", pos=fmt_vec((lx, ly, 0)))
        add_geom(leg, f"{name}_leg_{i}_geom", (0, 0, 0), (0.02, 0.02, ch2), material="counter_mat")

    # 去掉中间层板/抽屉，参考图无隔层

    return counter


def place_receipt(worldbody, counter_pos, counter_name, ox=0, oy=0):
    rx = counter_pos[0] + ox
    ry = counter_pos[1] + oy
    rz = counter_pos[2] + COUNTER_HEIGHT/2 + COUNTER_THICK/2 + RECEIPT_THICK/2 + 0.001
    r = make_body(worldbody, f"receipt_{counter_name}", pos=fmt_vec((rx, ry, rz)))
    add_geom(r, f"receipt_{counter_name}_geom", (0, 0, 0),
             (RECEIPT_LENGTH/2, RECEIPT_WIDTH/2, RECEIPT_THICK/2), material="receipt_mat")


def place_basket_on_counter(worldbody, counter_name, counter_pos, ox=0, oy=0):
    """购物篮放在台面上"""
    bx = counter_pos[0] + ox
    by = counter_pos[1] + oy
    bz = counter_pos[2] + COUNTER_HEIGHT/2 + COUNTER_THICK/2 + BASKET_HEIGHT/2

    basket = make_body(worldbody, f"basket_{counter_name}", pos=fmt_vec((bx, by, bz)))

    bll2 = BASKET_BOTTOM_LENGTH / 2
    blw2 = BASKET_BOTTOM_WIDTH / 2
    bhh2 = BASKET_HEIGHT / 2
    bt = BASKET_THICK

    add_geom(basket, f"basket_{counter_name}_bottom", (0, 0, -bhh2 + bt/2),
             (bll2, blw2, bt/2), material="basket_mat")

    for wname, wpos, size in [
        (f"basket_{counter_name}_front",  (0, blw2 - bt/2, 0), (bll2, bt/2, bhh2)),
        (f"basket_{counter_name}_back",   (0, -blw2 + bt/2, 0), (bll2, bt/2, bhh2)),
        (f"basket_{counter_name}_left",   (-bll2 + bt/2, 0, 0), (bt/2, blw2, bhh2)),
        (f"basket_{counter_name}_right",  (bll2 - bt/2, 0, 0), (bt/2, blw2, bhh2)),
    ]:
        add_geom(basket, f"{wname}_geom", wpos, size, material="basket_mat")

    for x_sign in [-1, 1]:
        hx = x_sign * (bll2 - 0.03)
        hb = make_body(basket, f"basket_{counter_name}_handle_{x_sign}",
                       pos=fmt_vec((hx, 0, bhh2 + 0.03)))
        add_geom(hb, f"basket_{counter_name}_handle_{x_sign}_geom", (0, 0, 0),
                 (BASKET_HANDLE_RADIUS, BASKET_HANDLE_RADIUS, 0.03), material="basket_mat")

    hbar = make_body(basket, f"basket_{counter_name}_handle_bar", pos=fmt_vec((0, 0, bhh2 + 0.06)))
    add_geom(hbar, f"basket_{counter_name}_handle_bar_geom", (0, 0, 0),
             (bll2 - 0.03, BASKET_HANDLE_RADIUS, BASKET_HANDLE_RADIUS), material="basket_mat")

    return basket


# ============================================================
# 主生成函数
# ============================================================

def generate_mjcf():
    mujoco = ET.Element("mujoco", {"model": "supermarket_scene_6x8_competition"})

    ET.SubElement(mujoco, "compiler", {
        "angle": "degree", "coordinate": "local", "inertiafromgeom": "true",
        "meshdir": ".",
        "texturedir": ".",
    })

    ET.SubElement(mujoco, "option", {
        "timestep": "0.005", "gravity": "0 0 -9.81",
        "iterations": "50", "ls_iterations": "50",
    })

    default = ET.SubElement(mujoco, "default")
    ET.SubElement(default, "geom", {"condim": "3", "friction": "0.5 0.1 0.1"})

    visual = ET.SubElement(mujoco, "visual")
    ET.SubElement(visual, "global", {
        "azimuth": "120", "elevation": "-30",
        "offwidth": "1920", "offheight": "1080",
    })

    # Assets
    assets = create_assets()
    mujoco.append(assets)

    # Worldbody
    worldbody = ET.SubElement(mujoco, "worldbody")

    # 光照：室内散射光，模拟超市天花板日光灯
    # 使用 directional 光保证均匀亮度，但禁用阴影（castshadow=false）
    # 在3m高度均匀分布，覆盖6x8场地
    lights = [
        ("light_center", "0 0 3", "0.5 0.5 0.5"),
        ("light_nw", "-2 3 3", "0.4 0.4 0.4"),
        ("light_ne", "2 3 3", "0.4 0.4 0.4"),
        ("light_sw", "-2 -3 3", "0.4 0.4 0.4"),
        ("light_se", "2 -3 3", "0.4 0.4 0.4"),
        ("light_n",  "0 3 3", "0.4 0.4 0.4"),
        ("light_s",  "0 -3 3", "0.4 0.4 0.4"),
        ("light_w",  "-3 0 3", "0.4 0.4 0.4"),
        ("light_e",  "3 0 3", "0.4 0.4 0.4"),
    ]
    for name, pos, diffuse in lights:
        ET.SubElement(worldbody, "light", {
            "name": name, "pos": pos, "directional": "true",
            "castshadow": "false", "diffuse": diffuse,
        })

    build_floor(worldbody)
    build_divider_line(worldbody)
    build_start_zones(worldbody)
    build_boundary_lines(worldbody)
    build_walls(worldbody)

    # 四组货架
    shelf_groups = [
        ("shelf_group_1", (SHELF_LEFT_X,  SHELF_TOP_Y,    0)),
        ("shelf_group_2", (SHELF_RIGHT_X, SHELF_TOP_Y,    0)),
        ("shelf_group_3", (SHELF_LEFT_X,  SHELF_BOTTOM_Y, 0)),
        ("shelf_group_4", (SHELF_RIGHT_X, SHELF_BOTTOM_Y, 0)),
    ]

    total_products = 0
    global_pid = 0
    for i, (name, pos) in enumerate(shelf_groups):
        build_shelf_group(worldbody, name, pos[0], pos[1], pos[2])
        count, global_pid = place_products_on_shelf(worldbody, pos[0], pos[1], pos[2], i, name, global_pid)
        total_products += count

    # 交付台/补货台
    counters = [
        ("counter_top_left", COUNTER_TOP_LEFT_POS),      # 补货台
        ("counter_top_right", COUNTER_TOP_RIGHT_POS),    # 交付台
        ("counter_bottom_left", COUNTER_BOTTOM_LEFT_POS), # 交付台
        ("counter_bottom_right", COUNTER_BOTTOM_RIGHT_POS), # 补货台
    ]
    for name, pos in counters:
        build_counter(worldbody, name, pos)
        place_receipt(worldbody, pos, name,
                      ox=np.random.uniform(-0.3, 0.3),
                      oy=np.random.uniform(-0.2, 0.2))

    # 购物篮放在台面上
    counter_pos_map = {
        "counter_top_left": COUNTER_TOP_LEFT_POS,
        "counter_top_right": COUNTER_TOP_RIGHT_POS,
        "counter_bottom_left": COUNTER_BOTTOM_LEFT_POS,
        "counter_bottom_right": COUNTER_BOTTOM_RIGHT_POS,
    }
    basket_offsets = [
        ("counter_top_left", -0.35, 0.25),
        ("counter_top_right", 0.35, -0.25),
        ("counter_bottom_left", 0.35, 0.25),
        ("counter_bottom_right", -0.35, -0.25),
    ]
    for cname, ox, oy in basket_offsets:
        place_basket_on_counter(worldbody, cname, counter_pos_map[cname], ox=ox, oy=oy)

    # 相机
    ET.SubElement(worldbody, "camera", {
        "name": "top_view", "pos": "0 0 6",
        "quat": "0.7071 0.7071 0 0", "fovy": "60",
    })
    ET.SubElement(worldbody, "camera", {
        "name": "aisle_view", "pos": "0 0 1.5",
        "quat": "0.5 0.5 0.5 0.5", "fovy": "75",
    })
    ET.SubElement(worldbody, "camera", {
        "name": "shelf_close", "pos": "1.5 2 1.2",
        "xyaxes": "-0.8 0 0 0 0 0.8", "fovy": "60",
    })
    ET.SubElement(worldbody, "camera", {
        "name": "overview_angle", "pos": "3 3 3",
        "xyaxes": "-0.5 0.5 0 -0.25 -0.25 0.5", "fovy": "60",
    })

    return mujoco, total_products


def prettify_xml(elem):
    rough = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough)
    return reparsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")


if __name__ == "__main__":
    output_path = os.path.join(BASE_DIR, "supermarket_scene_6x8_competition.xml")

    mjcf, total_products = generate_mjcf()
    xml_str = prettify_xml(mjcf)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    print(f"✅ 比赛场景文件已生成: {output_path}")
    print(f"   文件大小: {len(xml_str):,} bytes")
    print(f"   商品数量: {total_products} (真实mesh模型)")
    print(f"\n🚀 预览:")
    print(f"   python -m mujoco.viewer --mjcf={output_path}")
