#!/usr/bin/env python3
"""拍摄超市场景截图 - 使用 mujoco.Renderer"""
import os
import mujoco
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "supermarket_scene_6x8_competition.xml")
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

width, height = 1920, 1080

# ============================================================
# 使用 XML 中定义的相机
# ============================================================
cam_names = ["top_view", "aisle_view", "shelf_close", "overview_angle"]

for cam_name in cam_names:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id == -1:
        print(f"⚠️ Camera '{cam_name}' not found")
        continue
    
    renderer = mujoco.Renderer(model, height, width)
    
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = cam_id
    
    renderer.update_scene(data, camera=cam)
    pixels = renderer.render()
    
    path = os.path.join(SCREENSHOT_DIR, f"v5_{cam_name}.png")
    img = Image.fromarray(pixels)
    img.save(path)
    print(f"✅ Saved: {path}")
    renderer.close()

# ============================================================
# 自由相机 - 通过直接设置 cam.pos 和 cam.mat 来精确定位
# ============================================================
free_cams = [
    # (name, pos_x, pos_y, pos_z, lookat_x, lookat_y, lookat_z)
    # 俯视图 - 正上方俯瞰全局（Z轴指向下）
    ("v5_top_view",     0, 0, 7,     0, 0, 0.8),
    # 斜角全景 - 从右前上方看全场景  
    ("v5_overview",     5, 6, 4,     0, 0, 0.8),
    # 正面看货架 - 从Y轴负方向看
    ("v5_front",        0, -5, 2.5,  0, 0, 0.8),
    # 侧面看货架 - 从X轴正方向看
    ("v5_side",         5, 0, 2.0,   0, 0, 0.8),
    # 货架侧面特写 - 看层板结构
    ("v5_shelf_side",   3.0, 0.4, 1.2,  -1.5, 0.4, 0.8),
    # 低角度看货架底部
    ("v5_low_angle",    1.5, 2.0, 0.3,  -1.5, 0.4, 0.8),
    # 过道视角
    ("v5_aisle",        0, 0, 1.8,   0, 2.0, 0.8),
]

for name, px, py, pz, lx, ly, lz in free_cams:
    renderer = mujoco.Renderer(model, height, width)
    
    cam_pos = np.array([px, py, pz])
    lookat = np.array([lx, ly, lz])
    d = cam_pos - lookat
    
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = np.linalg.norm(d)
    cam.azimuth = np.degrees(np.arctan2(d[1], d[0]))
    # elevation: 负值=向下看，正值=向上看
    # 从相机位置看向目标点，elevation 应该是负的（向下看场景）
    cam.elevation = -np.degrees(np.arctan2(d[2], np.sqrt(d[0]**2 + d[1]**2)))

    renderer.update_scene(data, camera=cam)
    pixels = renderer.render()
    
    path = os.path.join(SCREENSHOT_DIR, f"{name}.png")
    img = Image.fromarray(pixels)
    img.save(path)
    print(f"✅ Saved: {path}")
    renderer.close()

print("\n🎉 All screenshots taken!")
