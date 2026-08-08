#!/usr/bin/env python3
"""
从数据库读取货架配置和商品信息，生成 MuJoCo 场景 XML

与 generate_scene.py 的区别:
  - 货架位置从 shelf_groups 表读取（而非硬编码）
  - 商品摆放从 shelf_inventory 表读取（而非随机生成）
  - 交付桌位置从 delivery_tables 表读取（而非硬编码）
  - 地板、围墙等场景结构保持固定
"""

import os
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom

from shelf_database import (
    ShelfDatabase, ShelfGroup, ShelfType,
    DEFAULT_SHELF_LENGTH, DEFAULT_SHELF_WIDTH, DEFAULT_SHELF_HEIGHT,
    DEFAULT_NUM_LEVELS, DEFAULT_BOTTOM_CLEARANCE, DEFAULT_LEVEL_SPACING,
    DEFAULT_SHELF_DEPTH_NORMAL, DEFAULT_SHELF_DEPTH_BOTTOM,
    DEFAULT_PANEL_THICK, DEFAULT_BACK_THICK,
)
from scene_geometry import (
    DELIVERY_TABLE_HEIGHT as COUNTER_HEIGHT,
    DELIVERY_TABLE_LENGTH as COUNTER_LENGTH,
    DELIVERY_TABLE_TOP_THICKNESS as COUNTER_THICK,
    DELIVERY_TABLE_WIDTH as COUNTER_WIDTH,
)

# ============================================================
# 路径配置
# ============================================================
MUJOCO_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MUJOCO_DIR)
SCANNED_DIR = os.path.join(MUJOCO_DIR, "assets/scanned/models")

# ============================================================
# 全局参数 (单位：米)
# ============================================================
MAP_WIDTH = 6.0
MAP_LENGTH = 8.0

FLOOR_SIZE = (MAP_WIDTH, MAP_LENGTH, 0.02)

# 围墙参数
WALL_HEIGHT = 0.25
WALL_THICK = 0.05
WALL_INNER_X = MAP_WIDTH / 2
WALL_INNER_Y = MAP_LENGTH / 2
WALL_CENTER_X = WALL_INNER_X + WALL_THICK / 2
WALL_CENTER_Y = WALL_INNER_Y + WALL_THICK / 2

# 起点区域
START_ZONE_TOP_POS = (0, 3.6, 0.01)
START_ZONE_BOTTOM_POS = (0, -3.6, 0.01)
START_ZONE_RADIUS = 0.3

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

# 中间分割线
DIVIDER_Y = 0.0

# ============================================================
# 商品模型定义 (与 generate_scene.py 一致)
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
# Assets
# ============================================================

def create_assets():
    assets = ET.Element("asset")

    ET.SubElement(assets, "texture", {
        "name": "floor_tex", "type": "2d", "builtin": "checker",
        "width": "256", "height": "256",
        "rgb1": "0.85 0.82 0.78", "rgb2": "0.75 0.72 0.68",
    })

    materials = {
        "floor_mat":       ("floor_tex", "0.05", "12 16"),
        "shelf_mat":       ("", "0.3", ""),
        "shelf_layer_mat": ("", "0.3", ""),
        "frame_mat":       ("", "0.4", ""),
        "counter_mat":     ("", "0.15", ""),
        "counter_top_mat": ("", "0.15", ""),
        "basket_mat":      ("", "0.1", ""),
        "receipt_mat":     ("", "0.05", ""),
        "boundary_mat":    ("", "0.1", ""),
        "divider_mat":     ("", "0.1", ""),
        "start_zone_mat":  ("", "0.05", ""),
        "wall_mat":        ("", "0.2", ""),
        "shelf_side_red":  ("", "0.3", ""),
        "shelf_side_black":("", "0.3", ""),
    }
    mat_rgba = {
        "floor_mat": "0.85 0.82 0.78 1.0",
        "shelf_mat": "0.15 0.15 0.18 1.0",
        "shelf_layer_mat": "0.75 0.15 0.15 1.0",
        "frame_mat": "0.25 0.25 0.30 1.0",
        "counter_mat": "0.50 0.50 0.52 1.0",
        "counter_top_mat": "0.05 0.50 0.25 1.0",
        "basket_mat": "0.15 0.25 0.55 1.0",
        "receipt_mat": "0.98 0.98 0.95 1.0",
        "boundary_mat": "0.9 0.9 0.9 1.0",
        "divider_mat": "1.0 0.85 0.0 1.0",
        "start_zone_mat": "0.0 0.7 0.2 0.8",
        "wall_mat": "1.0 1.0 1.0 1.0",
        "shelf_side_red": "0.75 0.15 0.15 1.0",
        "shelf_side_black": "0.08 0.08 0.08 1.0",
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
# 场景结构 (与 generate_scene.py 一致)
# ============================================================

def build_floor(worldbody):
    body = make_body(worldbody, "floor", pos=fmt_vec((0, 0, -FLOOR_SIZE[2]/2)))
    add_geom(body, "floor_geom", (0, 0, 0),
             (FLOOR_SIZE[0]/2, FLOOR_SIZE[1]/2, FLOOR_SIZE[2]/2), material="floor_mat")

def build_divider_line(worldbody):
    body = make_body(worldbody, "divider_line", pos=fmt_vec((0, DIVIDER_Y, 0.001)))
    add_geom(body, "divider_geom", (0, 0, 0),
             (MAP_WIDTH/2, 0.03, 0.001), material="divider_mat")
    n_dashes = 30
    dash_len = MAP_WIDTH / n_dashes
    for i in range(n_dashes):
        if i % 2 == 1:
            x = -MAP_WIDTH/2 + dash_len * (i + 0.5)
            db = make_body(worldbody, f"divider_black_{i}", pos=fmt_vec((x, DIVIDER_Y, 0.002)))
            add_geom(db, f"divider_black_{i}_geom", (0, 0, 0),
                     (dash_len/2 * 0.8, 0.03, 0.001), rgba="0.1 0.1 0.1 1.0")

def build_start_zones(worldbody):
    for prefix, pos in [("start_zone_top", START_ZONE_TOP_POS),
                        ("start_zone_bottom", START_ZONE_BOTTOM_POS)]:
        body = make_body(worldbody, prefix, pos=fmt_vec(pos))
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
    hh = WALL_HEIGHT / 2
    ht = WALL_THICK / 2
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

def build_counter(worldbody, name, world_x, world_y, yaw):
    """Build a delivery table from its database local-origin pose."""
    import math
    counter = make_body(
        worldbody,
        name,
        pos=fmt_vec((world_x, world_y, 0)),
        euler=f"0 0 {math.degrees(yaw):.4f}",
    )
    cl2 = COUNTER_LENGTH / 2
    cw2 = COUNTER_WIDTH / 2
    ch2 = COUNTER_HEIGHT / 2
    ct = COUNTER_THICK
    table_body = make_body(counter, f"{name}_body", pos=fmt_vec((cl2, cw2, ch2)))
    add_geom(table_body, f"{name}_top", (0, 0, ch2), (cl2, cw2, ct/2), material="counter_top_mat")
    for i, (lx, ly) in enumerate([
        (-cl2+0.03, -cw2+0.03), (-cl2+0.03, cw2-0.03),
        (cl2-0.03, -cw2+0.03), (cl2-0.03, cw2-0.03),
    ]):
        leg = make_body(table_body, f"{name}_leg_{i}", pos=fmt_vec((lx, ly, 0)))
        add_geom(leg, f"{name}_leg_{i}_geom", (0, 0, 0), (0.02, 0.02, ch2), material="counter_mat")
    return counter


def place_receipt_on_counter(counter, counter_name):
    """A static paper list; delivery goods themselves remain out of scope."""
    receipt = make_body(
        counter,
        f"receipt_{counter_name}",
        pos=fmt_vec((COUNTER_LENGTH / 2, COUNTER_WIDTH / 2,
                     COUNTER_HEIGHT + COUNTER_THICK / 2 + RECEIPT_THICK / 2 + 0.001)),
    )
    add_geom(receipt, f"receipt_{counter_name}_geom", (0, 0, 0),
             (RECEIPT_LENGTH / 2, RECEIPT_WIDTH / 2, RECEIPT_THICK / 2), material="receipt_mat")

def place_receipt(worldbody, counter_pos, counter_name, ox=0, oy=0):
    rx = counter_pos[0] + ox
    ry = counter_pos[1] + oy
    rz = counter_pos[2] + COUNTER_HEIGHT/2 + COUNTER_THICK/2 + RECEIPT_THICK/2 + 0.001
    r = make_body(worldbody, f"receipt_{counter_name}", pos=fmt_vec((rx, ry, rz)))
    add_geom(r, f"receipt_{counter_name}_geom", (0, 0, 0),
             (RECEIPT_LENGTH/2, RECEIPT_WIDTH/2, RECEIPT_THICK/2), material="receipt_mat")

def place_basket_on_counter(worldbody, counter_name, counter_pos, ox=0, oy=0):
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
# 货架构建 (与 generate_scene.py 一致)
# ============================================================

def build_shelf_group(worldbody, group_name, world_x, world_y, yaw,
                       shelf_type: ShelfType = None):
    """
    构建双面货架组。

    注意: world_x, world_y 是数据库中的货架局部原点 map 坐标。
    但 MuJoCo 中 body 的 pos 是 body 自身的原点位置。
    我们以货架中心为 body 原点，需要转换。

    货架中心 = 局部原点 + (shelf_width/2, shelf_length/2) 在局部坐标系中,
    然后旋转 yaw 到世界坐标系。
    """
    import math
    if shelf_type is None:
        shelf_type = ShelfType(
            id=0, name="_default",
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

    sw = shelf_type.shelf_width
    sl = shelf_type.shelf_length
    sh = shelf_type.shelf_height
    nl = shelf_type.num_levels
    bc = shelf_type.bottom_clearance
    ls = shelf_type.level_spacing
    pt = shelf_type.panel_thick
    bt = shelf_type.back_thick

    # 局部坐标系中: 中心相对于货架局部原点的偏移
    local_center_x = sw / 2
    local_center_y = sl / 2

    # 转换到世界坐标
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    center_wx = world_x + local_center_x * cos_yaw - local_center_y * sin_yaw
    center_wy = world_y + local_center_x * sin_yaw + local_center_y * cos_yaw

    group_body = make_body(worldbody, group_name, pos=fmt_vec((center_wx, center_wy, 0)))
    if abs(yaw) > 1e-6:
        group_body.set("euler", f"0 0 {math.degrees(yaw):.4f}")

    half_y = sl / 2
    prefix = group_name + "_"

    # 中间背板
    back_panel = make_body(group_body, f"{prefix}back_panel",
                           pos=fmt_vec((0, 0, sh/2)))
    add_geom(back_panel, f"{prefix}back_geom", (0, 0, 0),
             (bt/2, half_y, sh/2), material="shelf_mat")

    # 各层板
    for level in range(nl):
        z = bc + level * ls
        depth = shelf_type.shelf_depth_bottom if level == 0 else shelf_type.shelf_depth_normal

        shelf_body = make_body(group_body, f"{prefix}shelf_level_{level}",
                               pos=fmt_vec((0, 0, z)))

        # -X 侧层板
        shelf_cx_neg = -(bt/2 + depth/2)
        add_geom(shelf_body, f"{prefix}shelf_L{level}_neg_body",
                 (shelf_cx_neg, 0, 0),
                 (depth/2, half_y, pt/2),
                 material="shelf_side_black")
        add_geom(shelf_body, f"{prefix}shelf_L{level}_neg_face",
                 (shelf_cx_neg - depth/2 + 0.001, 0, 0),
                 (0.002, half_y, pt/2),
                 material="shelf_side_red")

        # +X 侧层板
        shelf_cx_pos = (bt/2 + depth/2)
        add_geom(shelf_body, f"{prefix}shelf_L{level}_pos_body",
                 (shelf_cx_pos, 0, 0),
                 (depth/2, half_y, pt/2),
                 material="shelf_side_black")
        add_geom(shelf_body, f"{prefix}shelf_L{level}_pos_face",
                 (shelf_cx_pos + depth/2 - 0.001, 0, 0),
                 (0.002, half_y, pt/2),
                 material="shelf_side_red")

        # 导航 sites
        add_site(group_body, f"{group_name}_L{level}_face_neg",
                 (shelf_cx_neg - depth/2 - 0.05, 0, z),
                 size="0.03", rgba="0 0 0 0")
        add_site(group_body, f"{group_name}_L{level}_face_pos",
                 (shelf_cx_pos + depth/2 + 0.05, 0, z),
                 size="0.03", rgba="0 0 0 0")

    return group_body


# ============================================================
# 商品放置 (从数据库读取, 使用 mesh 碰撞)
# ============================================================

def place_mesh_product(parent, model_name, pos, euler, unique_id):
    """放置真实mesh模型（独立body + freejoint，mesh碰撞）"""
    if model_name not in ALL_MODELS:
        print(f"  [WARN] SKU '{model_name}' not found in ALL_MODELS, skipping")
        return None
    body_name = f"product_{model_name}_uid{unique_id}"
    body = make_body(parent, body_name, pos=fmt_vec(pos))
    if euler != "0 0 0":
        body.set("euler", euler)

    ET.SubElement(body, "freejoint", {"name": f"{body_name}_joint"})

    # 视觉几何体
    ET.SubElement(body, "geom", {
        "name": f"{body_name}_visual",
        "type": "mesh",
        "mesh": f"mesh_{model_name}",
        "material": f"mat_{model_name}",
        "contype": "0",
        "conaffinity": "0",
        "group": "2",
    })

    # 碰撞几何体 (mesh)
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


def place_products_from_database(worldbody, db: ShelfDatabase):
    """
    从数据库读取固定货位，为实际存在的商品生成 MuJoCo body。

    - Z = 层板表面 + height_cm / 2
    - Y = y_cm / 100 (相对于货架原点)

    返回: total_products
    """
    all_slots = db.get_all_slots_world()
    global_pid = 0

    for slot in all_slots:
        sku = slot["actual_sku"]
        if sku is None:
            continue
        world_x = slot["world_x"]
        world_y = slot["world_y"]
        world_z = slot["world_z"]

        place_mesh_product(worldbody, sku, (world_x, world_y, world_z), "0 0 0", global_pid)
        global_pid += 1

    return global_pid


# ============================================================
# 主生成函数
# ============================================================

def generate_mjcf_from_db(db_path: str):
    """
    从数据库生成 MuJoCo XML

    Args:
        db_path: SQLite 数据库文件路径
    """
    db = ShelfDatabase(db_path)

    mujoco = ET.Element("mujoco", {"model": "supermarket_scene_from_db"})

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

    # 光照
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

    # 场景结构
    build_floor(worldbody)
    build_divider_line(worldbody)
    build_start_zones(worldbody)
    build_boundary_lines(worldbody)
    build_walls(worldbody)

    # 从数据库读取货架组并构建
    shelf_groups = db.get_all_shelf_groups()
    print(f"从数据库读取到 {len(shelf_groups)} 个货架组")

    for sg in shelf_groups:
        group_name = f"shelf_group_{sg.id}"
        # 获取货架类型参数
        st = None
        if sg.shelf_type_id:
            st = db.get_shelf_type(sg.shelf_type_id)
        build_shelf_group(worldbody, group_name, sg.world_x, sg.world_y, sg.yaw, st)

    # 从数据库读取商品并放置
    total_products = place_products_from_database(worldbody, db)

    # Delivery tables are independent database fixtures.  They have a paper
    # list as a visual marker, but do not contain inventory at this stage.
    delivery_tables = db.get_all_delivery_tables()
    print(f"从数据库读取到 {len(delivery_tables)} 张交付桌")
    for table in delivery_tables:
        table_name = f"delivery_table_{table.id}"
        counter = build_counter(worldbody, table_name, table.world_x, table.world_y, table.yaw)
        place_receipt_on_counter(counter, table_name)

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

    db.close()
    return mujoco, total_products


def prettify_xml(elem):
    rough = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough)
    return reparsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")


if __name__ == "__main__":
    db_path = os.path.join(PROJECT_ROOT, "shelf_inventory.db")
    output_path = os.path.join(MUJOCO_DIR, "supermarket_scene_from_db.xml")

    if not os.path.exists(db_path):
        print(f"❌ 数据库文件不存在: {db_path}")
        print("   请先运行 init_database.py 初始化数据库")
        exit(1)

    mjcf, total_products = generate_mjcf_from_db(db_path)
    xml_str = prettify_xml(mjcf)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    print(f"✅ 场景文件已生成: {output_path}")
    print(f"   文件大小: {len(xml_str):,} bytes")
    print(f"   商品数量: {total_products}")
    print(f"\n🚀 预览:")
    print(f"   python -m mujoco.viewer --mjcf={output_path}")
