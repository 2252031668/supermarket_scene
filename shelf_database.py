#!/usr/bin/env python3
"""
货架数据库 - 基于 SQLite 的超市货架商品管理

货架局部坐标系（以 reference anchor 为原点）:
    +X 沿货架宽度方向
    +Y 沿货架长度方向
    +Z 向上

世界坐标系使用 ROS map 的右手系: +X × +Y = +Z。
world_x/world_y 是货架局部原点的 map 坐标，而非屏幕方位角。

面: 0 = -X 侧, 1 = +X 侧
层: 0 到 (num_levels-1), 从下到上

Slot ID 格式: {shelf_id}-{face}-{level}-{y_cm}-{z_offset_cm}
    每个 slot 对应一个商品实例, y_cm 和 z_offset_cm 精确到厘米
    例如: 1-1-2-9-4 = 1号货架, +X 侧, 第2层, 商品中心距原点Y轴9cm, 距层板表面4cm

世界坐标转换:
    货架组在世界坐标系中有 (world_x, world_y, yaw)
    局部坐标 (lx, ly, lz) -> 世界坐标 (wx, wy, wz):
        wx = world_x + lx*cos(yaw) - ly*sin(yaw)
        wy = world_y + lx*sin(yaw) + ly*cos(yaw)
        wz = lz
"""

import sqlite3
import math
import os
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field


# ============================================================
# 货架物理参数默认值 (单位: 米) — fallback, 优先使用 shelf_types 表
# ============================================================
DEFAULT_SHELF_LENGTH = 1.86       # 货架组长度 (沿Y)
DEFAULT_SHELF_WIDTH = 0.80        # 货架组宽度 (沿X)
DEFAULT_SHELF_HEIGHT = 1.65       # 货架组高度 (沿Z)
DEFAULT_NUM_LEVELS = 5            # 层数
DEFAULT_BOTTOM_CLEARANCE = 0.05   # 底层离地高度
DEFAULT_LEVEL_SPACING = 0.40      # 层间距
DEFAULT_PANEL_THICK = 0.02        # 层板厚度
DEFAULT_BACK_THICK = 0.005        # 背板厚度
DEFAULT_SHELF_DEPTH_NORMAL = 0.30 # 层板深度(每面, 上层)
DEFAULT_SHELF_DEPTH_BOTTOM = 0.40 # 层板深度(每面, 底层)

# 向后兼容别名
SHELF_LENGTH = DEFAULT_SHELF_LENGTH
SHELF_WIDTH = DEFAULT_SHELF_WIDTH
SHELF_HEIGHT = DEFAULT_SHELF_HEIGHT
NUM_LEVELS = DEFAULT_NUM_LEVELS
BOTTOM_CLEARANCE = DEFAULT_BOTTOM_CLEARANCE
LEVEL_SPACING = DEFAULT_LEVEL_SPACING
PANEL_THICK = DEFAULT_PANEL_THICK
BACK_THICK = DEFAULT_BACK_THICK
SHELF_DEPTH_NORMAL = DEFAULT_SHELF_DEPTH_NORMAL
SHELF_DEPTH_BOTTOM = DEFAULT_SHELF_DEPTH_BOTTOM


# ============================================================
# 数据类
# ============================================================

@dataclass
class ShelfType:
    """货架类型 — 物理尺寸参数 (10 个字段)"""
    id: int
    name: str                           # 类型名, 如 "standard"
    shelf_length: float = DEFAULT_SHELF_LENGTH
    shelf_width: float = DEFAULT_SHELF_WIDTH
    shelf_height: float = DEFAULT_SHELF_HEIGHT
    num_levels: int = DEFAULT_NUM_LEVELS
    bottom_clearance: float = DEFAULT_BOTTOM_CLEARANCE
    level_spacing: float = DEFAULT_LEVEL_SPACING
    panel_thick: float = DEFAULT_PANEL_THICK
    back_thick: float = DEFAULT_BACK_THICK
    shelf_depth_normal: float = DEFAULT_SHELF_DEPTH_NORMAL
    shelf_depth_bottom: float = DEFAULT_SHELF_DEPTH_BOTTOM


@dataclass
class ShelfGroup:
    """货架组在世界中的位置"""
    id: int
    name: str
    world_x: float              # 货架局部原点的 map X
    world_y: float              # 货架局部原点的 map Y
    yaw: float = 0.0            # 绕 map +Z 的逆时针旋转 (弧度)
    shelf_type_id: int = 0      # 关联 shelf_types.id
    created_at: str = ""


@dataclass
class SkuInfo:
    """商品SKU信息"""
    sku: str
    category: str = ""
    mesh_file: str = ""
    tex_file: str = ""


@dataclass
class ShelfSlot:
    """货架上的一个商品槽位 (一个 slot = 一个商品实例)"""
    id: int = -1               # 数据库自增ID
    shelf_id: int = 0
    face: int = 0              # 0=-X 侧, 1=+X 侧
    level: int = 0             # 0 到 (num_levels-1), 从下到上
    y_cm: float = 0.0          # 商品中心距货架原点Y轴距离 (厘米)
    z_offset_cm: float = 0.0   # 商品中心距当前层板表面的高度 (厘米)
    sku: str = ""


@dataclass
class LocalPos:
    """货架局部坐标"""
    x: float
    y: float
    z: float


@dataclass
class WorldPos:
    """世界坐标"""
    x: float
    y: float
    z: float


# ============================================================
# 数据库核心类
# ============================================================

class ShelfDatabase:
    """货架数据库"""

    def __init__(self, db_path: str = ":memory:"):
        """
        初始化数据库连接

        Args:
            db_path: 数据库文件路径, 默认内存数据库
                     如 "shelf_inventory.db" 则持久化到文件
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._create_tables()

    def _create_tables(self):
        """创建数据库表结构"""
        cur = self.conn.cursor()

        # 货架类型表 (10 个物理参数)
        cur.execute("""
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
            )
        """)

        # 货架组表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS shelf_groups (
                id              INTEGER PRIMARY KEY,
                name            TEXT NOT NULL DEFAULT '',
                world_x         REAL NOT NULL DEFAULT 0.0,
                world_y         REAL NOT NULL DEFAULT 0.0,
                yaw             REAL NOT NULL DEFAULT 0.0,
                shelf_type_id   INTEGER DEFAULT NULL,
                created_at      TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (shelf_type_id) REFERENCES shelf_types(id) ON DELETE SET NULL
            )
        """)

        # SKU目录表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sku_catalog (
                sku         TEXT PRIMARY KEY,
                category    TEXT NOT NULL DEFAULT '',
                mesh_file   TEXT NOT NULL DEFAULT '',
                tex_file    TEXT NOT NULL DEFAULT ''
            )
        """)

        # 货架库存表 (核心)
        # 一个 slot = 一个商品实例, y_cm 和 z_offset_cm 精确到厘米
        # 同一位置 (shelf_id, face, level, y_cm) 只能有一个商品
        cur.execute("""
            CREATE TABLE IF NOT EXISTS shelf_inventory (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                shelf_id        INTEGER NOT NULL,
                face            INTEGER NOT NULL CHECK(face IN (0, 1)),
                level           INTEGER NOT NULL CHECK(level >= 0),
                y_cm            REAL NOT NULL CHECK(y_cm >= 0),
                z_offset_cm     REAL NOT NULL DEFAULT 0.0,
                sku             TEXT NOT NULL,
                FOREIGN KEY (shelf_id) REFERENCES shelf_groups(id) ON DELETE CASCADE,
                FOREIGN KEY (sku) REFERENCES sku_catalog(sku) ON DELETE CASCADE,
                UNIQUE(shelf_id, face, level, y_cm)
            )
        """)

        # 索引
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_inventory_shelf
                ON shelf_inventory(shelf_id, face, level, y_cm)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_inventory_sku
                ON shelf_inventory(sku)
        """)

        self.conn.commit()

    # ================================================================
    # 货架类型管理
    # ================================================================

    def add_shelf_type(self, name: str, shelf_length: float, shelf_width: float,
                       shelf_height: float, num_levels: int, bottom_clearance: float,
                       level_spacing: float, panel_thick: float, back_thick: float,
                       shelf_depth_normal: float, shelf_depth_bottom: float) -> int:
        """添加一个货架类型 (10 个物理参数), 返回新类型的ID"""
        cur = self.conn.execute(
            "INSERT INTO shelf_types (name, shelf_length, shelf_width, shelf_height, "
            "num_levels, bottom_clearance, level_spacing, panel_thick, back_thick, "
            "shelf_depth_normal, shelf_depth_bottom) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (name, shelf_length, shelf_width, shelf_height,
             num_levels, bottom_clearance, level_spacing, panel_thick, back_thick,
             shelf_depth_normal, shelf_depth_bottom)
        )
        self.conn.commit()
        return cur.lastrowid

    def get_shelf_type(self, type_id: int) -> Optional[ShelfType]:
        """获取货架类型信息"""
        row = self.conn.execute(
            "SELECT * FROM shelf_types WHERE id = ?", (type_id,)
        ).fetchone()
        if row is None:
            return None
        return ShelfType(**dict(row))

    def get_shelf_type_by_name(self, name: str) -> Optional[ShelfType]:
        """按名称获取货架类型"""
        row = self.conn.execute(
            "SELECT * FROM shelf_types WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            return None
        return ShelfType(**dict(row))

    def get_all_shelf_types(self) -> List[ShelfType]:
        """获取所有货架类型"""
        rows = self.conn.execute(
            "SELECT * FROM shelf_types ORDER BY id"
        ).fetchall()
        return [ShelfType(**dict(r)) for r in rows]

    def _resolve_shelf_params(self, shelf_id: int) -> ShelfType:
        """
        解析货架的物理参数, 优先从 shelf_type_id 获取, 否则使用默认值

        Returns:
            ShelfType — 始终可用的货架类型参数
        """
        shelf = self.get_shelf_group(shelf_id)
        if shelf is not None and shelf.shelf_type_id:
            st = self.get_shelf_type(shelf.shelf_type_id)
            if st is not None:
                return st
        # 构造默认类型
        return ShelfType(
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

    # ================================================================
    # 坐标计算 (基于 shelf_type + y_cm / z_offset_cm)
    # ================================================================

    def level_surface_z(self, level: int, shelf_type: ShelfType = None) -> float:
        """
        计算指定层的层板上表面在货架局部坐标系中的 Z 坐标 (单位: 米)
        """
        if shelf_type is None:
            return DEFAULT_BOTTOM_CLEARANCE + level * DEFAULT_LEVEL_SPACING + DEFAULT_PANEL_THICK / 2
        return shelf_type.bottom_clearance + level * shelf_type.level_spacing + shelf_type.panel_thick / 2

    def face_center_x(self, face: int, level: int, shelf_type: ShelfType = None) -> float:
        """
        计算指定面、指定层的层板中心在货架局部坐标系中的 X 坐标 (单位: 米)
        """
        if shelf_type is None:
            depth = DEFAULT_SHELF_DEPTH_BOTTOM if level == 0 else DEFAULT_SHELF_DEPTH_NORMAL
            back_thick = DEFAULT_BACK_THICK
            shelf_width = DEFAULT_SHELF_WIDTH
        else:
            depth = shelf_type.shelf_depth_bottom if level == 0 else shelf_type.shelf_depth_normal
            back_thick = shelf_type.back_thick
            shelf_width = shelf_type.shelf_width
        if face == 0:
            return back_thick / 2 + depth / 2
        else:
            return shelf_width - back_thick / 2 - depth / 2

    def slot_id_to_local(self, shelf_id: int, face: int, level: int,
                         y_cm: float, z_offset_cm: float) -> LocalPos:
        """
        将槽位ID转换为货架局部坐标 (商品空间中心, 单位: 米)

        Args:
            shelf_id: 货架组ID (用于查询货架类型参数)
            face: 面号 0=左, 1=右
            level: 层号
            y_cm: 商品中心距货架原点Y轴距离 (厘米)
            z_offset_cm: 商品中心距当前层板表面的高度 (厘米)
        """
        st = self._resolve_shelf_params(shelf_id)
        return LocalPos(
            x=self.face_center_x(face, level, st),
            y=y_cm / 100.0,   # 厘米 → 米
            z=self.level_surface_z(level, st) + z_offset_cm / 100.0,
        )

    @staticmethod
    def local_to_world(local: LocalPos, world_x: float, world_y: float,
                       yaw: float) -> WorldPos:
        """
        将货架局部坐标转换为世界坐标

        Args:
            local: 局部坐标
            world_x: 货架局部原点的 map X
            world_y: 货架局部原点的 map Y
            yaw: 绕 map +Z 的逆时针旋转 (弧度)

        Returns:
            世界坐标
        """
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        return WorldPos(
            x=world_x + local.x * cos_yaw - local.y * sin_yaw,
            y=world_y + local.x * sin_yaw + local.y * cos_yaw,
            z=local.z,
        )

    # ================================================================
    # 货架组管理
    # ================================================================

    def add_shelf_group(self, name: str = "", world_x: float = 0.0,
                        world_y: float = 0.0, yaw: float = 0.0,
                        shelf_type_id: int = None) -> int:
        """
        添加一个货架组

        Args:
            name: 货架名称
            world_x: 货架局部原点的 map X
            world_y: 货架局部原点的 map Y
            yaw: 绕 map +Z 的逆时针旋转 (弧度)
            shelf_type_id: 货架类型ID

        Returns:
            新货架组的ID
        """
        cur = self.conn.execute(
            "INSERT INTO shelf_groups (name, world_x, world_y, yaw, shelf_type_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, world_x, world_y, yaw, shelf_type_id)
        )
        self.conn.commit()
        return cur.lastrowid

    def update_shelf_group(self, shelf_id: int, name: str = None,
                           world_x: float = None, world_y: float = None,
                           yaw: float = None, shelf_type_id: int = None):
        """更新货架组信息"""
        fields = []
        values = []
        if name is not None:
            fields.append("name = ?")
            values.append(name)
        if world_x is not None:
            fields.append("world_x = ?")
            values.append(world_x)
        if world_y is not None:
            fields.append("world_y = ?")
            values.append(world_y)
        if yaw is not None:
            fields.append("yaw = ?")
            values.append(yaw)
        if shelf_type_id is not None:
            fields.append("shelf_type_id = ?")
            values.append(shelf_type_id)
        if not fields:
            return
        values.append(shelf_id)
        self.conn.execute(
            f"UPDATE shelf_groups SET {', '.join(fields)} WHERE id = ?",
            values
        )
        self.conn.commit()

    def remove_shelf_group(self, shelf_id: int) -> int:
        """删除货架组并返回级联删除的库存数量。"""
        removed = self.conn.execute(
            "SELECT COUNT(*) FROM shelf_inventory WHERE shelf_id = ?", (shelf_id,)
        ).fetchone()[0]
        self.conn.execute("DELETE FROM shelf_groups WHERE id = ?", (shelf_id,))
        self.conn.commit()
        return removed

    def get_shelf_group(self, shelf_id: int) -> Optional[ShelfGroup]:
        """获取货架组信息"""
        row = self.conn.execute(
            "SELECT id, name, world_x, world_y, yaw, shelf_type_id, created_at "
            "FROM shelf_groups WHERE id = ?",
            (shelf_id,)
        ).fetchone()
        if row is None:
            return None
        return ShelfGroup(**dict(row))

    def get_all_shelf_groups(self) -> List[ShelfGroup]:
        """获取所有货架组"""
        rows = self.conn.execute(
            "SELECT id, name, world_x, world_y, yaw, shelf_type_id, created_at "
            "FROM shelf_groups ORDER BY id"
        ).fetchall()
        return [ShelfGroup(**dict(r)) for r in rows]

    def get_shelf_world_pos(self, shelf_id: int) -> Optional[WorldPos]:
        """获取货架局部原点在 map 坐标系中的位置"""
        row = self.conn.execute(
            "SELECT world_x, world_y FROM shelf_groups WHERE id = ?",
            (shelf_id,)
        ).fetchone()
        if row is None:
            return None
        return WorldPos(x=row["world_x"], y=row["world_y"], z=0.0)

    # ================================================================
    # SKU 目录管理
    # ================================================================

    def register_sku(self, sku: str, category: str = "",
                     mesh_file: str = "", tex_file: str = ""):
        """注册或更新一个商品 SKU，不替换行以避免触发库存级联删除。"""
        self.conn.execute(
            "INSERT INTO sku_catalog (sku, category, mesh_file, tex_file) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(sku) DO UPDATE SET "
            "category = excluded.category, mesh_file = excluded.mesh_file, tex_file = excluded.tex_file",
            (sku, category, mesh_file, tex_file)
        )
        self.conn.commit()

    def register_skus_batch(self, skus: List[Dict[str, str]]):
        """批量注册SKU"""
        data = [(s["sku"], s.get("category", ""),
                 s.get("mesh_file", ""), s.get("tex_file", ""))
                for s in skus]
        self.conn.executemany(
            "INSERT INTO sku_catalog (sku, category, mesh_file, tex_file) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(sku) DO UPDATE SET "
            "category = excluded.category, mesh_file = excluded.mesh_file, tex_file = excluded.tex_file", data
        )
        self.conn.commit()

    def get_sku_info(self, sku: str) -> Optional[SkuInfo]:
        """获取SKU信息"""
        row = self.conn.execute(
            "SELECT sku, category, mesh_file, tex_file FROM sku_catalog WHERE sku = ?",
            (sku,)
        ).fetchone()
        if row is None:
            return None
        return SkuInfo(**dict(row))

    def get_all_skus(self) -> List[SkuInfo]:
        """获取所有SKU"""
        rows = self.conn.execute(
            "SELECT sku, category, mesh_file, tex_file FROM sku_catalog ORDER BY sku"
        ).fetchall()
        return [SkuInfo(**dict(r)) for r in rows]

    def remove_sku_from_catalog(self, sku: str):
        """从目录中删除SKU (级联删除库存)"""
        self.conn.execute("DELETE FROM sku_catalog WHERE sku = ?", (sku,))
        self.conn.commit()

    # ================================================================
    # 库存管理 (核心) — 一个 slot = 一个商品实例
    # ================================================================

    def set_slot(self, shelf_id: int, face: int, level: int,
                 y_cm: float, z_offset_cm: float, sku: str):
        """
        在货架上放置一个商品 (一个 slot = 一个商品实例)

        Args:
            shelf_id: 货架组ID
            face: 面 0=-X 侧, 1=+X 侧
            level: 层号 (从下到上)
            y_cm: 商品中心距货架原点Y轴距离 (厘米)
            z_offset_cm: 商品中心距当前层板表面的高度 (厘米)
            sku: 商品SKU名
        """
        self.conn.execute(
            "INSERT INTO shelf_inventory (shelf_id, face, level, y_cm, z_offset_cm, sku) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(shelf_id, face, level, y_cm) DO UPDATE SET "
            "z_offset_cm = excluded.z_offset_cm, sku = excluded.sku",
            (shelf_id, face, level, y_cm, z_offset_cm, sku)
        )
        self.conn.commit()

    def remove_slot(self, shelf_id: int, face: int, level: int, y_cm: float):
        """删除指定位置的商品"""
        self.conn.execute(
            "DELETE FROM shelf_inventory "
            "WHERE shelf_id=? AND face=? AND level=? AND y_cm=?",
            (shelf_id, face, level, y_cm)
        )
        self.conn.commit()

    def clear_shelf(self, shelf_id: int) -> int:
        """清空指定货架组的所有库存，并返回删除数量。"""
        cursor = self.conn.execute(
            "DELETE FROM shelf_inventory WHERE shelf_id=?", (shelf_id,)
        )
        self.conn.commit()
        return cursor.rowcount

    def clear_shelf_face(self, shelf_id: int, face: int) -> int:
        """清空货架指定局部 X 侧的所有库存，并返回删除数量。"""
        cursor = self.conn.execute(
            "DELETE FROM shelf_inventory WHERE shelf_id=? AND face=?", (shelf_id, face)
        )
        self.conn.commit()
        return cursor.rowcount

    def clear_shelf_face_level(self, shelf_id: int, face: int, level: int) -> int:
        """清空货架指定局部 X 侧和层号的库存，并返回删除数量。"""
        cursor = self.conn.execute(
            "DELETE FROM shelf_inventory WHERE shelf_id=? AND face=? AND level=?",
            (shelf_id, face, level)
        )
        self.conn.commit()
        return cursor.rowcount

    def get_slot(self, shelf_id: int, face: int, level: int,
                 y_cm: float) -> Optional[ShelfSlot]:
        """获取指定位置的商品"""
        row = self.conn.execute(
            "SELECT id, shelf_id, face, level, y_cm, z_offset_cm, sku "
            "FROM shelf_inventory "
            "WHERE shelf_id=? AND face=? AND level=? AND y_cm=?",
            (shelf_id, face, level, y_cm)
        ).fetchone()
        if row is None:
            return None
        return ShelfSlot(**dict(row))

    def get_shelf_inventory(self, shelf_id: int) -> List[ShelfSlot]:
        """获取指定货架组的所有商品"""
        rows = self.conn.execute(
            "SELECT id, shelf_id, face, level, y_cm, z_offset_cm, sku "
            "FROM shelf_inventory WHERE shelf_id=? "
            "ORDER BY face, level, y_cm, sku",
            (shelf_id,)
        ).fetchall()
        return [ShelfSlot(**dict(r)) for r in rows]

    def get_shelf_sku_summary(self, shelf_id: int) -> List[Dict[str, Any]]:
        """
        查询: 指定货架有哪些SKU及总数量 (每个 slot = 1个商品)
        """
        rows = self.conn.execute(
            "SELECT sku, COUNT(*) as total_qty "
            "FROM shelf_inventory WHERE shelf_id=? "
            "GROUP BY sku ORDER BY sku",
            (shelf_id,)
        ).fetchall()
        return [{"sku": r["sku"], "total_quantity": r["total_qty"]} for r in rows]

    def find_sku_locations(self, sku: str) -> List[Dict[str, Any]]:
        """
        查询: 指定SKU在哪些货架的哪些位置
        """
        rows = self.conn.execute(
            "SELECT si.shelf_id, sg.name as shelf_name, "
            "si.face, si.level, si.y_cm, si.z_offset_cm "
            "FROM shelf_inventory si "
            "JOIN shelf_groups sg ON si.shelf_id = sg.id "
            "WHERE si.sku = ? "
            "ORDER BY si.shelf_id, si.face, si.level, si.y_cm",
            (sku,)
        ).fetchall()
        return [dict(r) for r in rows]

    def find_sku_world_positions(self, sku: str) -> List[Dict[str, Any]]:
        """
        查询: 指定SKU所在位置的世界坐标
        """
        rows = self.conn.execute(
            "SELECT si.shelf_id, sg.name as shelf_name, "
            "sg.world_x, sg.world_y, sg.yaw, "
            "si.face, si.level, si.y_cm, si.z_offset_cm "
            "FROM shelf_inventory si "
            "JOIN shelf_groups sg ON si.shelf_id = sg.id "
            "WHERE si.sku = ? "
            "ORDER BY si.shelf_id, si.face, si.level, si.y_cm",
            (sku,)
        ).fetchall()

        results = []
        for r in rows:
            local = self.slot_id_to_local(
                r["shelf_id"], r["face"], r["level"], r["y_cm"], r["z_offset_cm"]
            )
            world = self.local_to_world(
                local, r["world_x"], r["world_y"], r["yaw"]
            )
            results.append({
                "shelf_id": r["shelf_id"],
                "shelf_name": r["shelf_name"],
                "face": r["face"],
                "level": r["level"],
                "y_cm": r["y_cm"],
                "z_offset_cm": r["z_offset_cm"],
                "world_x": world.x,
                "world_y": world.y,
                "world_z": world.z,
                "slot_id_str": f"{r['shelf_id']}-{r['face']}-{r['level']}-"
                               f"{r['y_cm']:.0f}-{r['z_offset_cm']:.0f}",
            })
        return results

    # ================================================================
    # 查询 API
    # ================================================================

    def get_sku_total_quantity(self, sku: str) -> int:
        """
        查询指定 SKU 在所有货架的总数量 (每个 slot = 1个)

        Args:
            sku: 商品SKU名

        Returns:
            总数量
        """
        row = self.conn.execute(
            "SELECT COUNT(*) FROM shelf_inventory WHERE sku = ?",
            (sku,)
        ).fetchone()
        return row[0]

    def get_sku_total_quantity_by_shelf(self, sku: str) -> List[Dict[str, Any]]:
        """
        查询指定 SKU 在每个货架的总数量

        Returns:
            [{"shelf_id": 1, "total_quantity": 10}, ...]
        """
        rows = self.conn.execute(
            "SELECT shelf_id, COUNT(*) as total_quantity "
            "FROM shelf_inventory WHERE sku = ? "
            "GROUP BY shelf_id ORDER BY shelf_id",
            (sku,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ================================================================
    # 坐标查询
    # ================================================================

    def get_slot_world_pos(self, shelf_id: int, face: int,
                           level: int, y_cm: float, z_offset_cm: float) -> Optional[WorldPos]:
        """
        获取指定位置的世界坐标
        """
        shelf = self.get_shelf_group(shelf_id)
        if shelf is None:
            return None
        local = self.slot_id_to_local(shelf_id, face, level, y_cm, z_offset_cm)
        return self.local_to_world(local, shelf.world_x, shelf.world_y, shelf.yaw)

    def get_all_slots_world(self) -> List[Dict[str, Any]]:
        """
        获取所有商品的世界坐标
        用于生成 MuJoCo 场景
        """
        rows = self.conn.execute(
            "SELECT si.shelf_id, si.face, si.level, si.y_cm, si.z_offset_cm, si.sku, "
            "sg.world_x, sg.world_y, sg.yaw "
            "FROM shelf_inventory si "
            "JOIN shelf_groups sg ON si.shelf_id = sg.id "
            "ORDER BY si.shelf_id, si.face, si.level, si.y_cm"
        ).fetchall()

        results = []
        for r in rows:
            local = self.slot_id_to_local(
                r["shelf_id"], r["face"], r["level"], r["y_cm"], r["z_offset_cm"]
            )
            world = self.local_to_world(
                local, r["world_x"], r["world_y"], r["yaw"]
            )
            results.append({
                "shelf_id": r["shelf_id"],
                "face": r["face"],
                "level": r["level"],
                "y_cm": r["y_cm"],
                "z_offset_cm": r["z_offset_cm"],
                "sku": r["sku"],
                "world_x": world.x,
                "world_y": world.y,
                "world_z": world.z,
                "yaw": r["yaw"],
                "slot_id_str": f"{r['shelf_id']}-{r['face']}-{r['level']}-"
                               f"{r['y_cm']:.0f}-{r['z_offset_cm']:.0f}",
            })
        return results

    def get_shelf_group_all_slots_world(self, shelf_id: int) -> List[Dict[str, Any]]:
        """
        获取指定货架组所有商品的世界坐标
        """
        shelf = self.get_shelf_group(shelf_id)
        if shelf is None:
            return []

        rows = self.conn.execute(
            "SELECT si.face, si.level, si.y_cm, si.z_offset_cm, si.sku "
            "FROM shelf_inventory si "
            "WHERE si.shelf_id = ? "
            "ORDER BY si.face, si.level, si.y_cm",
            (shelf_id,)
        ).fetchall()

        results = []
        for r in rows:
            local = self.slot_id_to_local(shelf_id, r["face"], r["level"],
                                          r["y_cm"], r["z_offset_cm"])
            world = self.local_to_world(local, shelf.world_x, shelf.world_y, shelf.yaw)
            results.append({
                "face": r["face"],
                "level": r["level"],
                "y_cm": r["y_cm"],
                "z_offset_cm": r["z_offset_cm"],
                "sku": r["sku"],
                "world_x": world.x,
                "world_y": world.y,
                "world_z": world.z,
                "slot_id_str": f"{shelf_id}-{r['face']}-{r['level']}-"
                               f"{r['y_cm']:.0f}-{r['z_offset_cm']:.0f}",
            })
        return results

    # ================================================================
    # 工具方法
    # ================================================================

    def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        n_types = self.conn.execute(
            "SELECT COUNT(*) FROM shelf_types"
        ).fetchone()[0]
        n_shelves = self.conn.execute(
            "SELECT COUNT(*) FROM shelf_groups"
        ).fetchone()[0]
        n_skus = self.conn.execute(
            "SELECT COUNT(*) FROM sku_catalog"
        ).fetchone()[0]
        n_items = self.conn.execute(
            "SELECT COUNT(*) FROM shelf_inventory"
        ).fetchone()[0]
        return {
            "shelf_types": n_types,
            "shelf_groups": n_shelves,
            "sku_catalog": n_skus,
            "total_items": n_items,
        }

    def slot_id_str_to_tuple(self, slot_id_str: str) -> Tuple[int, int, int, float, float]:
        """将 '0-1-2-9-4' 格式解析为 (shelf_id, face, level, y_cm, z_offset_cm)"""
        parts = slot_id_str.split("-")
        if len(parts) != 5:
            raise ValueError(f"Invalid slot ID string: {slot_id_str}, "
                             f"expected format 'shelf-face-level-ycm-zoffset'")
        return (int(parts[0]), int(parts[1]), int(parts[2]),
                float(parts[3]), float(parts[4]))

    def close(self):
        """关闭数据库连接"""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================
# 便捷函数: 从现有 generate_scene.py 的参数初始化数据库
# ============================================================

def init_database_from_scene_params(db: ShelfDatabase,
                                    shelf_positions: List[Dict[str, Any]],
                                    skus: List[Dict[str, str]] = None):
    """
    根据场景参数初始化数据库

    Args:
        db: ShelfDatabase 实例
        shelf_positions: 货架组位置列表
            [{"name": "shelf_0", "world_x": -1.1, "world_y": 1.18,
              "yaw": 0.0, "shelf_type_id": 1}, ...]
        skus: SKU目录列表
            [{"sku": "cracker_box", "category": "food", "mesh_file": "...", "tex_file": "..."}, ...]
    """
    # 注册SKU
    if skus:
        db.register_skus_batch(skus)

    # 添加货架组
    for sp in shelf_positions:
        db.add_shelf_group(
            name=sp.get("name", ""),
            world_x=sp["world_x"],
            world_y=sp["world_y"],
            yaw=sp.get("yaw", 0.0),
            shelf_type_id=sp.get("shelf_type_id"),
        )


# ============================================================
# 测试/演示
# ============================================================

if __name__ == "__main__":
    # 使用内存数据库演示
    db = ShelfDatabase(":memory:")

    # 0. 创建货架类型
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
    print(f"已创建货架类型: standard (id={type_id})")

    # 1. 注册SKU
    db.register_skus_batch([
        {"sku": "cracker_box", "category": "food"},
        {"sku": "tomato_soup_can", "category": "food"},
        {"sku": "mustard_bottle", "category": "drink"},
        {"sku": "banana", "category": "food"},
    ])

    # 2. 添加货架组 (参考现有场景的4个货架组位置)
    shelf_centers = [
        ("1号货架", -1.1,  1.18),   # -> ID 1
        ("2号货架",  1.1,  1.18),   # -> ID 2
        ("3号货架", -1.1, -1.18),   # -> ID 3
        ("4号货架",  1.1, -1.18),   # -> ID 4
    ]
    shelf_ids = []
    for name, cx, cy in shelf_centers:
        sid = db.add_shelf_group(
            name=name,
            world_x=cx - 0.40,   # 局部原点 X = 中心X - 半宽
            world_y=cy - 0.93,   # 局部原点 Y = 中心Y - 半长
            yaw=0.0,
            shelf_type_id=type_id,
        )
        shelf_ids.append(sid)

    S0, S1, S2, S3 = shelf_ids

    # 3. 放置商品 (一个 slot = 一个商品)
    # 格式: set_slot(shelf_id, face, level, y_cm, z_offset_cm, sku)
    db.set_slot(S0, face=0, level=2, y_cm=50, z_offset_cm=4, sku="cracker_box")
    db.set_slot(S0, face=0, level=2, y_cm=65, z_offset_cm=4, sku="tomato_soup_can")
    db.set_slot(S0, face=1, level=4, y_cm=90, z_offset_cm=5, sku="banana")
    db.set_slot(S1, face=0, level=0, y_cm=5,  z_offset_cm=3, sku="mustard_bottle")
    db.set_slot(S3, face=1, level=1, y_cm=30, z_offset_cm=6, sku="cracker_box")
    # 同一位置覆盖: 更新 S0-0-2-50 为另一个 SKU
    db.set_slot(S0, face=0, level=2, y_cm=50, z_offset_cm=4, sku="banana")

    # 4. 查询演示
    print("=" * 60)
    print("数据库统计:", db.get_stats())
    print("货架类型:", [(t.id, t.name) for t in db.get_all_shelf_types()])

    print(f"\n--- 查询: 货架{S0} ({shelf_centers[0][0]})的所有SKU及数量 ---")
    for item in db.get_shelf_sku_summary(S0):
        print(f"  {item['sku']}: {item['total_quantity']}个")

    print("\n--- 查询: cracker_box在哪些位置 ---")
    for loc in db.find_sku_locations("cracker_box"):
        print(f"  货架{loc['shelf_id']} 面{loc['face']} 层{loc['level']} "
              f"y={loc['y_cm']:.0f}cm z_offset={loc['z_offset_cm']:.0f}cm")

    print("\n--- 查询: cracker_box的世界坐标 ---")
    for pos in db.find_sku_world_positions("cracker_box"):
        print(f"  {pos['slot_id_str']}: world=({pos['world_x']:.3f}, "
              f"{pos['world_y']:.3f}, {pos['world_z']:.3f})")

    print(f"\n--- 查询: 位置 {S0}-0-2-65 的商品 ---")
    slot = db.get_slot(S0, 0, 2, 65)
    if slot:
        print(f"  {slot.sku}")

    print("\n--- 所有商品世界坐标 (用于MuJoCo生成) ---")
    all_slots = db.get_all_slots_world()
    for s in all_slots:
        print(f"  {s['slot_id_str']} {s['sku']}: "
              f"world=({s['world_x']:.3f}, {s['world_y']:.3f}, {s['world_z']:.3f})")

    print(f"\n--- 货架{S0}的世界位置 ---")
    pos = db.get_shelf_world_pos(S0)
    print(f"  局部原点: ({pos.x:.3f}, {pos.y:.3f})")

    # 5. 测试删除
    print("\n--- 测试: 删除位置 S0-0-2-65 ---")
    db.remove_slot(S0, face=0, level=2, y_cm=65)
    slot = db.get_slot(S0, 0, 2, 65)
    print(f"  删除后查询: {slot}")

    print("\n--- 测试: 解析 slot ID 字符串 ---")
    parsed = db.slot_id_str_to_tuple("0-1-2-9-4")
    print(f"  '0-1-2-9-4' -> {parsed}")

    db.close()
