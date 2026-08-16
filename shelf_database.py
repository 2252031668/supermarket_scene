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

Slot ID 格式: {shelf_id}-{face}-{level}-{y_cm}
    每个 slot 对应一个固定货位, y_cm 精确到厘米
    例如: 1-1-2-9 = 1号货架, +X 侧, 第2层, 商品中心距原点Y轴9cm

世界坐标转换:
    货架组在世界坐标系中有 (world_x, world_y, yaw)
    局部坐标 (lx, ly, lz) -> 世界坐标 (wx, wy, wz):
        wx = world_x + lx*cos(yaw) - ly*sin(yaw)
        wy = world_y + lx*sin(yaw) + ly*cos(yaw)
        wz = lz
"""

import sqlite3
import math
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass


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
UNKNOWN_SKU = "unknown"


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
class DeliveryTable:
    """A named delivery-table pose; it deliberately has no inventory slots."""
    id: int
    name: str
    world_x: float              # local min-X/min-Y footprint anchor in map
    world_y: float
    yaw: float = 0.0            # counter-clockwise around map +Z, in radians
    created_at: str = ""


@dataclass
class SkuInfo:
    """商品SKU信息"""
    sku: str
    category: str = ""
    mesh_file: str = ""
    tex_file: str = ""
    owlv2_prompt: str = ""
    qwen_grounding_prompt: str = ""
    reference_image_path: str = ""
    grasp_method: str = "夹爪"


@dataclass
class ShelfSlot:
    """货架上的固定货位。"""
    slot_id: str = ""
    shelf_id: int = 0
    face: int = 0              # 0=-X 侧, 1=+X 侧
    level: int = 0             # 0 到 (num_levels-1), 从下到上
    y_cm: float = 0.0          # 商品中心距货架原点Y轴距离 (厘米)
    expected_sku: str = ""
    actual_sku: Optional[str] = None
    width_cm: Optional[float] = None   # 商品沿局部Y的宽度 (厘米)
    height_cm: Optional[float] = None  # 商品沿局部Z的高度 (厘米)
    image_dir: str = ""               # 实例图片相对目录
    status: str = ""


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


_UNSET = object()


def slot_status(expected_sku: str, actual_sku: Optional[str]) -> str:
    if actual_sku is None:
        return "缺货"
    return "正常" if actual_sku == expected_sku else "摆放错误"


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

        # Delivery tables are scene fixtures, not shelves.  In particular they
        # never own rows from shelf_inventory.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS delivery_tables (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT NOT NULL DEFAULT '',
                world_x         REAL NOT NULL DEFAULT 0.0,
                world_y         REAL NOT NULL DEFAULT 0.0,
                yaw             REAL NOT NULL DEFAULT 0.0,
                created_at      TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        # SKU目录表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sku_catalog (
                sku         TEXT PRIMARY KEY,
                category    TEXT NOT NULL DEFAULT '',
                mesh_file   TEXT NOT NULL DEFAULT '',
                tex_file    TEXT NOT NULL DEFAULT '',
                owlv2_prompt TEXT NOT NULL DEFAULT '',
                qwen_grounding_prompt TEXT NOT NULL DEFAULT '',
                reference_image_path TEXT NOT NULL DEFAULT '',
                grasp_method TEXT NOT NULL DEFAULT '夹爪' CHECK(grasp_method IN ('夹爪', '吸盘'))
            )
        """)

        self._ensure_sku_schema()
        self._ensure_inventory_schema()
        self.conn.commit()

    def _ensure_sku_schema(self):
        """Add the local OWLv2 prompt to catalogues created before this field existed."""
        columns = {
            row["name"] for row in self.conn.execute("PRAGMA table_info(sku_catalog)").fetchall()
        }
        if "owlv2_prompt" not in columns:
            self.conn.execute(
                "ALTER TABLE sku_catalog ADD COLUMN owlv2_prompt TEXT NOT NULL DEFAULT ''"
            )
        if "qwen_grounding_prompt" not in columns:
            self.conn.execute(
                "ALTER TABLE sku_catalog ADD COLUMN qwen_grounding_prompt TEXT NOT NULL DEFAULT ''"
            )
        if "reference_image_path" not in columns:
            self.conn.execute(
                "ALTER TABLE sku_catalog ADD COLUMN reference_image_path TEXT NOT NULL DEFAULT ''"
            )
        if "grasp_method" not in columns:
            self.conn.execute(
                "ALTER TABLE sku_catalog ADD COLUMN grasp_method TEXT NOT NULL DEFAULT '夹爪'"
            )

    def _create_inventory_table(self):
        self.conn.execute("""
            CREATE TABLE shelf_inventory (
                slot_id         TEXT PRIMARY KEY,
                shelf_id        INTEGER NOT NULL,
                face            INTEGER NOT NULL CHECK(face IN (0, 1)),
                level           INTEGER NOT NULL CHECK(level >= 0),
                y_cm            REAL NOT NULL CHECK(y_cm >= 0),
                expected_sku    TEXT NOT NULL,
                actual_sku      TEXT DEFAULT NULL,
                width_cm        REAL DEFAULT NULL CHECK(width_cm IS NULL OR width_cm >= 0),
                height_cm       REAL DEFAULT NULL CHECK(height_cm IS NULL OR height_cm >= 0),
                image_dir       TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (shelf_id) REFERENCES shelf_groups(id) ON DELETE CASCADE,
                FOREIGN KEY (expected_sku) REFERENCES sku_catalog(sku) ON DELETE RESTRICT,
                FOREIGN KEY (actual_sku) REFERENCES sku_catalog(sku) ON DELETE RESTRICT,
                UNIQUE(shelf_id, face, level, y_cm)
            )
        """)

    def _ensure_inventory_schema(self):
        """Create the current schema; legacy files require the one-time migrator."""
        exists = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='shelf_inventory'"
        ).fetchone()
        columns = [] if exists is None else [row["name"] for row in self.conn.execute(
            "PRAGMA table_info(shelf_inventory)"
        ).fetchall()]
        if not columns:
            self._create_inventory_table()
        elif not {"slot_id", "expected_sku", "actual_sku"}.issubset(columns):
            raise RuntimeError(
                "Legacy shelf_inventory schema detected; run the fixed-slot migration first"
            )

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_inventory_shelf
                ON shelf_inventory(shelf_id, face, level, y_cm)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_inventory_expected_sku
                ON shelf_inventory(expected_sku)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_inventory_actual_sku
                ON shelf_inventory(actual_sku)
        """)

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

    def update_shelf_type(self, type_id: int, shelf_length: float, shelf_width: float,
                          shelf_height: float, num_levels: int, bottom_clearance: float,
                          level_spacing: float, panel_thick: float, back_thick: float,
                          shelf_depth_normal: float, shelf_depth_bottom: float) -> None:
        """Update shared physical parameters while keeping assigned inventory valid."""
        if self.get_shelf_type(type_id) is None:
            raise ValueError(f"Shelf type {type_id} does not exist")
        positive_values = {
            "shelf_length": shelf_length,
            "shelf_width": shelf_width,
            "shelf_height": shelf_height,
            "level_spacing": level_spacing,
            "panel_thick": panel_thick,
            "back_thick": back_thick,
            "shelf_depth_normal": shelf_depth_normal,
            "shelf_depth_bottom": shelf_depth_bottom,
        }
        if any(value <= 0 for value in positive_values.values()):
            raise ValueError("Shelf dimensions and thicknesses must be greater than zero")
        if num_levels < 1:
            raise ValueError("num_levels must be at least one")
        if bottom_clearance < 0:
            raise ValueError("bottom_clearance must not be negative")

        invalid_slot = self.conn.execute(
            "SELECT si.shelf_id, si.level, si.y_cm "
            "FROM shelf_inventory si JOIN shelf_groups sg ON sg.id = si.shelf_id "
            "WHERE sg.shelf_type_id = ? AND (si.level >= ? OR si.y_cm > ?) LIMIT 1",
            (type_id, num_levels, shelf_length * 100),
        ).fetchone()
        if invalid_slot is not None:
            raise ValueError(
                f"Existing slot on shelf {invalid_slot['shelf_id']} would be outside the updated type "
                f"(level {invalid_slot['level']}, y={invalid_slot['y_cm']} cm)"
            )

        self.conn.execute(
            "UPDATE shelf_types SET shelf_length=?, shelf_width=?, shelf_height=?, num_levels=?, "
            "bottom_clearance=?, level_spacing=?, panel_thick=?, back_thick=?, "
            "shelf_depth_normal=?, shelf_depth_bottom=? WHERE id=?",
            (shelf_length, shelf_width, shelf_height, num_levels, bottom_clearance,
             level_spacing, panel_thick, back_thick, shelf_depth_normal,
             shelf_depth_bottom, type_id),
        )
        self.conn.commit()

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
    # 坐标计算 (基于 shelf_type + y_cm / height_cm)
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

    def level_opening_height(self, level: int, shelf_type: ShelfType = None) -> float:
        """Return the usable vertical calibration span for a shelf level in metres."""
        st = shelf_type or ShelfType(
            id=0, name="_default", shelf_length=DEFAULT_SHELF_LENGTH,
            shelf_width=DEFAULT_SHELF_WIDTH, shelf_height=DEFAULT_SHELF_HEIGHT,
            num_levels=DEFAULT_NUM_LEVELS, bottom_clearance=DEFAULT_BOTTOM_CLEARANCE,
            level_spacing=DEFAULT_LEVEL_SPACING, panel_thick=DEFAULT_PANEL_THICK,
            back_thick=DEFAULT_BACK_THICK, shelf_depth_normal=DEFAULT_SHELF_DEPTH_NORMAL,
            shelf_depth_bottom=DEFAULT_SHELF_DEPTH_BOTTOM,
        )
        # Shelf types currently define equal level pitch and do not provide an
        # independent top clearance. Reuse that type's pitch for every layer.
        return max(0.0, st.level_spacing - st.panel_thick)

    def get_shelf_calibration(self, shelf_id: int) -> Optional[Dict[str, Any]]:
        """Return the shelf-type dimensions used by the photo annotation workflow."""
        shelf = self.get_shelf_group(shelf_id)
        if shelf is None:
            return None
        st = self._resolve_shelf_params(shelf_id)
        return {
            "shelf_id": shelf_id,
            "shelf_length_cm": st.shelf_length * 100,
            "levels": [
                {
                    "level": level,
                    "surface_z_cm": self.level_surface_z(level, st) * 100,
                    "opening_height_cm": self.level_opening_height(level, st) * 100,
                }
                for level in range(st.num_levels)
            ],
        }

    def slot_id_to_local(self, shelf_id: int, face: int, level: int,
                         y_cm: float, height_cm: Optional[float] = None) -> LocalPos:
        """
        将槽位ID转换为货架局部坐标 (商品空间中心, 单位: 米)

        Args:
            shelf_id: 货架组ID (用于查询货架类型参数)
            face: 面号 0=左, 1=右
            level: 层号
            y_cm: 商品中心距货架原点Y轴距离 (厘米)
            height_cm: 商品沿局部Z的高度 (厘米), 商品中心位于其一半高度处
        """
        st = self._resolve_shelf_params(shelf_id)
        return LocalPos(
            x=self.face_center_x(face, level, st),
            y=y_cm / 100.0,   # 厘米 → 米
            z=self.level_surface_z(level, st) + (height_cm or 0.0) / 200.0,
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
    # 交付桌管理
    # ================================================================

    def add_delivery_table(self, name: str = "", world_x: float = 0.0,
                           world_y: float = 0.0, yaw: float = 0.0) -> int:
        """Add a delivery table at its stable local footprint anchor."""
        cur = self.conn.execute(
            "INSERT INTO delivery_tables (name, world_x, world_y, yaw) VALUES (?, ?, ?, ?)",
            (name, world_x, world_y, yaw),
        )
        self.conn.commit()
        return cur.lastrowid

    def update_delivery_table(self, table_id: int, name: str = None,
                              world_x: float = None, world_y: float = None,
                              yaw: float = None):
        """Update only the editable delivery-table pose fields."""
        fields = []
        values = []
        for column, value in (("name", name), ("world_x", world_x),
                              ("world_y", world_y), ("yaw", yaw)):
            if value is not None:
                fields.append(f"{column} = ?")
                values.append(value)
        if not fields:
            return
        values.append(table_id)
        self.conn.execute(
            f"UPDATE delivery_tables SET {', '.join(fields)} WHERE id = ?", values
        )
        self.conn.commit()

    def remove_delivery_table(self, table_id: int):
        """Remove a table fixture.  No inventory can be affected."""
        self.conn.execute("DELETE FROM delivery_tables WHERE id = ?", (table_id,))
        self.conn.commit()

    def get_delivery_table(self, table_id: int) -> Optional[DeliveryTable]:
        row = self.conn.execute(
            "SELECT id, name, world_x, world_y, yaw, created_at "
            "FROM delivery_tables WHERE id = ?", (table_id,)
        ).fetchone()
        return None if row is None else DeliveryTable(**dict(row))

    def get_all_delivery_tables(self) -> List[DeliveryTable]:
        rows = self.conn.execute(
            "SELECT id, name, world_x, world_y, yaw, created_at "
            "FROM delivery_tables ORDER BY id"
        ).fetchall()
        return [DeliveryTable(**dict(row)) for row in rows]

    # ================================================================
    # SKU 目录管理
    # ================================================================

    @staticmethod
    def _owlv2_prompt(value: str) -> str:
        prompt = str(value).strip()
        if len(prompt) > 240:
            raise ValueError("owlv2_prompt must be at most 240 characters")
        return prompt

    @staticmethod
    def _qwen_grounding_prompt(value: str) -> str:
        prompt = str(value).strip()
        if len(prompt) > 400:
            raise ValueError("qwen_grounding_prompt must be at most 400 characters")
        return prompt

    @staticmethod
    def _grasp_method(value: str) -> str:
        method = str(value).strip()
        if method not in {"夹爪", "吸盘"}:
            raise ValueError("grasp_method must be 夹爪 or 吸盘")
        return method

    def register_sku(self, sku: str, category: str = "",
                     mesh_file: str = "", tex_file: str = "", owlv2_prompt: Optional[str] = None,
                     reference_image_path: Optional[str] = None, grasp_method: Optional[str] = None,
                     qwen_grounding_prompt: Optional[str] = None):
        """注册或更新一个商品 SKU，不替换行以避免触发库存级联删除。"""
        existing = self.get_sku_info(sku)
        if existing is None:
            self.conn.execute(
                "INSERT INTO sku_catalog (sku, category, mesh_file, tex_file, owlv2_prompt, qwen_grounding_prompt, reference_image_path, grasp_method) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (sku, category, mesh_file, tex_file, self._owlv2_prompt(owlv2_prompt or ""),
                 self._qwen_grounding_prompt(qwen_grounding_prompt or ""),
                 str(reference_image_path or ""), self._grasp_method(grasp_method or "夹爪")),
            )
        else:
            self.conn.execute(
                "UPDATE sku_catalog SET category=?, mesh_file=?, tex_file=?, owlv2_prompt=?, qwen_grounding_prompt=?, reference_image_path=?, grasp_method=? WHERE sku=?",
                (category, mesh_file, tex_file,
                 existing.owlv2_prompt if owlv2_prompt is None else self._owlv2_prompt(owlv2_prompt),
                 existing.qwen_grounding_prompt if qwen_grounding_prompt is None else self._qwen_grounding_prompt(qwen_grounding_prompt),
                 existing.reference_image_path if reference_image_path is None else str(reference_image_path),
                 existing.grasp_method if grasp_method is None else self._grasp_method(grasp_method), sku),
            )
        self.conn.commit()

    def register_skus_batch(self, skus: List[Dict[str, str]]):
        """批量注册SKU"""
        for sku in skus:
            self.register_sku(
                sku["sku"], sku.get("category", ""), sku.get("mesh_file", ""),
                sku.get("tex_file", ""), sku.get("owlv2_prompt") if "owlv2_prompt" in sku else None,
                sku.get("reference_image_path") if "reference_image_path" in sku else None,
                sku.get("grasp_method") if "grasp_method" in sku else None,
                sku.get("qwen_grounding_prompt") if "qwen_grounding_prompt" in sku else None,
            )

    def get_sku_info(self, sku: str) -> Optional[SkuInfo]:
        """获取SKU信息"""
        row = self.conn.execute(
            "SELECT sku, category, mesh_file, tex_file, owlv2_prompt, qwen_grounding_prompt, reference_image_path, grasp_method FROM sku_catalog WHERE sku = ?",
            (sku,)
        ).fetchone()
        if row is None:
            return None
        return SkuInfo(**dict(row))

    def get_all_skus(self) -> List[SkuInfo]:
        """获取所有SKU"""
        rows = self.conn.execute(
            "SELECT sku, category, mesh_file, tex_file, owlv2_prompt, qwen_grounding_prompt, reference_image_path, grasp_method FROM sku_catalog ORDER BY sku"
        ).fetchall()
        return [SkuInfo(**dict(r)) for r in rows]

    def update_sku(self, sku: str, category: str, mesh_file: str, tex_file: str,
                   owlv2_prompt: str, reference_image_path: Optional[str] = None,
                   grasp_method: Optional[str] = None, qwen_grounding_prompt: Optional[str] = None) -> SkuInfo:
        current = self.get_sku_info(sku)
        if current is None:
            raise ValueError(f"SKU {sku} does not exist")
        self.conn.execute(
            "UPDATE sku_catalog SET category=?, mesh_file=?, tex_file=?, owlv2_prompt=?, qwen_grounding_prompt=?, reference_image_path=?, grasp_method=? WHERE sku=?",
            (category, mesh_file, tex_file, self._owlv2_prompt(owlv2_prompt),
             current.qwen_grounding_prompt if qwen_grounding_prompt is None else self._qwen_grounding_prompt(qwen_grounding_prompt),
             current.reference_image_path if reference_image_path is None else str(reference_image_path),
             current.grasp_method if grasp_method is None else self._grasp_method(grasp_method), sku),
        )
        self.conn.commit()
        return self.get_sku_info(sku)

    def remove_sku_from_catalog(self, sku: str):
        """删除未被固定货位引用的 SKU。"""
        if sku == UNKNOWN_SKU:
            raise ValueError("unknown cannot be deleted")
        referenced = self.conn.execute(
            "SELECT COUNT(*) FROM shelf_inventory "
            "WHERE expected_sku = ? OR actual_sku = ?",
            (sku, sku),
        ).fetchone()[0]
        if referenced:
            raise ValueError("SKU is referenced by shelf slots")
        self.conn.execute("DELETE FROM sku_catalog WHERE sku = ?", (sku,))
        self.conn.commit()

    def ensure_unknown_sku(self):
        self.conn.execute(
            "INSERT INTO sku_catalog (sku, category, mesh_file, tex_file, owlv2_prompt, qwen_grounding_prompt, reference_image_path, grasp_method) "
            "VALUES (?, '', '', '', '', '', '', '夹爪') ON CONFLICT(sku) DO NOTHING",
            (UNKNOWN_SKU,),
        )

    def rename_sku(self, old_sku: str, new_sku: str,
                   reference_image_path: Optional[str] = None) -> SkuInfo:
        old_sku, new_sku = str(old_sku).strip(), str(new_sku).strip()
        if not old_sku or not new_sku:
            raise ValueError("SKU name cannot be empty")
        if old_sku == UNKNOWN_SKU or new_sku == UNKNOWN_SKU:
            raise ValueError("unknown is reserved")
        with self.conn:
            current = self.get_sku_info(old_sku)
            if current is None:
                raise ValueError(f"SKU {old_sku} does not exist")
            if self.get_sku_info(new_sku) is not None:
                raise ValueError(f"SKU {new_sku} already exists")
            self.conn.execute(
                "INSERT INTO sku_catalog (sku, category, mesh_file, tex_file, owlv2_prompt, qwen_grounding_prompt, reference_image_path, grasp_method) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (new_sku, current.category, current.mesh_file, current.tex_file,
                 current.owlv2_prompt,
                 current.qwen_grounding_prompt,
                 current.reference_image_path if reference_image_path is None else reference_image_path,
                 current.grasp_method),
            )
            self.conn.execute("UPDATE shelf_inventory SET expected_sku=? WHERE expected_sku=?", (new_sku, old_sku))
            self.conn.execute("UPDATE shelf_inventory SET actual_sku=? WHERE actual_sku=?", (new_sku, old_sku))
            self.conn.execute("DELETE FROM sku_catalog WHERE sku=?", (old_sku,))
        return self.get_sku_info(new_sku)

    def delete_skus_to_unknown(self, skus: List[str]) -> List[str]:
        requested = list(dict.fromkeys(str(sku).strip() for sku in skus))
        if not requested or any(not sku for sku in requested):
            raise ValueError("At least one SKU is required")
        if UNKNOWN_SKU in requested:
            raise ValueError("unknown cannot be deleted")
        placeholders = ", ".join("?" for _ in requested)
        with self.conn:
            existing = {
                row[0] for row in self.conn.execute(
                    f"SELECT sku FROM sku_catalog WHERE sku IN ({placeholders})", requested
                )
            }
            missing = [sku for sku in requested if sku not in existing]
            if missing:
                raise ValueError(f"SKU does not exist: {missing[0]}")
            self.ensure_unknown_sku()
            self.conn.execute(
                f"UPDATE shelf_inventory SET expected_sku=? WHERE expected_sku IN ({placeholders})",
                (UNKNOWN_SKU, *requested),
            )
            self.conn.execute(
                f"UPDATE shelf_inventory SET actual_sku=? WHERE actual_sku IN ({placeholders})",
                (UNKNOWN_SKU, *requested),
            )
            self.conn.execute(f"DELETE FROM sku_catalog WHERE sku IN ({placeholders})", requested)
        return requested

    # ================================================================
    # 固定货位管理
    # ================================================================

    @staticmethod
    def _slot_from_row(row: sqlite3.Row) -> ShelfSlot:
        data = dict(row)
        data["status"] = slot_status(data["expected_sku"], data["actual_sku"])
        return ShelfSlot(**data)

    def _validate_slot_position(self, shelf_id: int, face: int,
                                level: int, y_cm: float) -> None:
        if face not in (0, 1) or level < 0 or y_cm < 0:
            raise ValueError("Invalid shelf position")
        if self.get_shelf_group(shelf_id) is None:
            raise ValueError(f"Shelf {shelf_id} does not exist")
        shelf_type = self._resolve_shelf_params(shelf_id)
        if level >= shelf_type.num_levels or y_cm > shelf_type.shelf_length * 100:
            raise ValueError(
                f"Slot {self.format_slot_id(shelf_id, face, level, y_cm)} "
                "is outside the shelf bounds"
            )

    def _validate_slot_skus(self, expected_sku: str,
                            actual_sku: Optional[str]) -> None:
        if not expected_sku or self.get_sku_info(expected_sku) is None:
            raise ValueError(f"SKU {expected_sku} does not exist")
        if actual_sku is not None and self.get_sku_info(actual_sku) is None:
            raise ValueError(f"SKU {actual_sku} does not exist")

    def create_slot(self, shelf_id: int, face: int, level: int, y_cm: float,
                    expected_sku: str, actual_sku: Any = _UNSET,
                    width_cm: Optional[float] = None,
                    height_cm: Optional[float] = None,
                    image_dir: str = "") -> str:
        shelf_id, face, level, y_cm = int(shelf_id), int(face), int(level), float(y_cm)
        actual_sku = expected_sku if actual_sku is _UNSET else actual_sku
        self._validate_slot_position(shelf_id, face, level, y_cm)
        self._validate_slot_skus(expected_sku, actual_sku)
        slot_id = self.format_slot_id(shelf_id, face, level, y_cm)
        try:
            self.conn.execute(
                "INSERT INTO shelf_inventory "
                "(slot_id, shelf_id, face, level, y_cm, expected_sku, actual_sku, "
                "width_cm, height_cm, image_dir) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (slot_id, shelf_id, face, level, y_cm, expected_sku, actual_sku,
                 width_cm, height_cm, image_dir),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError(f"Slot {slot_id} already exists") from error
        self.conn.commit()
        return slot_id

    def update_slot(self, slot_id: str, **changes: Any) -> ShelfSlot:
        allowed = {"expected_sku", "actual_sku", "width_cm", "height_cm", "image_dir"}
        invalid = set(changes) - allowed
        if invalid:
            raise ValueError(f"Slot location fields cannot be changed: {', '.join(sorted(invalid))}")
        slot = self.get_slot_by_id(slot_id)
        if slot is None:
            raise ValueError(f"Slot {slot_id} does not exist")
        if not changes:
            return slot

        expected_sku = str(changes.get("expected_sku", slot.expected_sku)).strip()
        actual_sku = changes.get("actual_sku", slot.actual_sku)
        if actual_sku is not None:
            actual_sku = str(actual_sku).strip()
        self._validate_slot_skus(expected_sku, actual_sku)

        values = {
            "expected_sku": expected_sku,
            "actual_sku": actual_sku,
            "width_cm": changes.get("width_cm", slot.width_cm),
            "height_cm": changes.get("height_cm", slot.height_cm),
            "image_dir": str(changes.get("image_dir", slot.image_dir)),
        }
        self.conn.execute(
            "UPDATE shelf_inventory SET expected_sku=?, actual_sku=?, width_cm=?, "
            "height_cm=?, image_dir=? WHERE slot_id=?",
            (*values.values(), slot_id),
        )
        self.conn.commit()
        return self.get_slot_by_id(slot_id)

    def set_actual_sku(self, slot_id: str, actual_sku: Optional[str]) -> ShelfSlot:
        return self.update_slot(slot_id, actual_sku=actual_sku)

    def set_actual_sku_batch(self, changes: List[tuple[str, Optional[str]]]) -> List[ShelfSlot]:
        """Update selected fixed-slot observations in one SQLite transaction."""
        if not changes:
            raise ValueError("At least one slot change is required")
        slot_ids = [slot_id for slot_id, _ in changes]
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError("Duplicate slot IDs are not allowed")
        with self.conn:
            for slot_id, actual_sku in changes:
                slot = self.get_slot_by_id(slot_id)
                if slot is None:
                    raise ValueError(f"Slot {slot_id} does not exist")
                if actual_sku is not None:
                    actual_sku = str(actual_sku).strip()
                self._validate_slot_skus(slot.expected_sku, actual_sku)
                self.conn.execute(
                    "UPDATE shelf_inventory SET actual_sku=? WHERE slot_id=?",
                    (actual_sku, slot_id),
                )
        return [self.get_slot_by_id(slot_id) for slot_id in slot_ids]

    def take_slot(self, slot_id: str) -> ShelfSlot:
        return self.set_actual_sku(slot_id, None)

    def restock_slot(self, slot_id: str) -> ShelfSlot:
        slot = self.get_slot_by_id(slot_id)
        if slot is None:
            raise ValueError(f"Slot {slot_id} does not exist")
        return self.set_actual_sku(slot_id, slot.expected_sku)

    def import_slots_batch(self, new_skus: List[Dict[str, str]], slots: List[Dict[str, Any]],
                           sku_prompts: Optional[List[Dict[str, str]]] = None,
                           sku_metadata: Optional[List[Dict[str, str]]] = None) -> List[str]:
        """Atomically create SKUs and fixed shelf slots."""
        if not slots:
            raise ValueError("At least one shelf slot is required")
        keys = [(int(slot["shelf_id"]), int(slot["face"]), int(slot["level"]), float(slot["y_cm"])) for slot in slots]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate positions exist in this import")
        prompt_by_sku: Dict[str, str] = {}
        for item in sku_prompts or []:
            if not isinstance(item, dict):
                raise ValueError("sku_prompts must contain objects")
            sku_name = str(item.get("sku", "")).strip()
            if not sku_name or sku_name in prompt_by_sku:
                raise ValueError("sku_prompts contains an invalid or duplicate SKU")
            prompt = self._owlv2_prompt(item.get("owlv2_prompt", ""))
            if prompt:
                prompt_by_sku[sku_name] = prompt
        metadata_by_sku: Dict[str, Dict[str, str]] = {}
        for item in sku_metadata or []:
            if not isinstance(item, dict):
                raise ValueError("sku_metadata must contain objects")
            sku_name = str(item.get("sku", "")).strip()
            if not sku_name or sku_name in metadata_by_sku:
                raise ValueError("sku_metadata contains an invalid or duplicate SKU")
            metadata_by_sku[sku_name] = {
                "reference_image_path": str(item.get("reference_image_path", "")),
                "grasp_method": self._grasp_method(item.get("grasp_method", "夹爪")),
            }
        with self.conn:
            for sku in new_skus:
                sku_name = str(sku["sku"]).strip()
                if not sku_name:
                    raise ValueError("SKU name cannot be empty")
                self.conn.execute(
                    "INSERT INTO sku_catalog (sku, category, mesh_file, tex_file) VALUES (?, ?, '', '') "
                    "ON CONFLICT(sku) DO NOTHING",
                    (sku_name, str(sku.get("category", "")))
                )
            for sku_name, prompt in prompt_by_sku.items():
                if self.get_sku_info(sku_name) is None:
                    raise ValueError(f"SKU {sku_name} does not exist")
                self.conn.execute("UPDATE sku_catalog SET owlv2_prompt=? WHERE sku=?", (prompt, sku_name))
            for sku_name, metadata in metadata_by_sku.items():
                current = self.get_sku_info(sku_name)
                if current is None:
                    raise ValueError(f"SKU {sku_name} does not exist")
                self.conn.execute(
                    "UPDATE sku_catalog SET reference_image_path=?, grasp_method=? WHERE sku=?",
                    (metadata["reference_image_path"] or current.reference_image_path,
                     metadata["grasp_method"], sku_name),
                )
            for slot, key in zip(slots, keys):
                shelf_id, face, level, y_cm = key
                expected_sku = str(slot["expected_sku"]).strip()
                actual_sku = slot.get("actual_sku", expected_sku)
                if actual_sku is not None:
                    actual_sku = str(actual_sku).strip()
                self._validate_slot_position(*key)
                self._validate_slot_skus(expected_sku, actual_sku)
                slot_id = self.format_slot_id(*key)
                self.conn.execute(
                    "INSERT INTO shelf_inventory "
                    "(slot_id, shelf_id, face, level, y_cm, expected_sku, actual_sku, "
                    "width_cm, height_cm, image_dir) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (slot_id, shelf_id, face, level, y_cm, expected_sku, actual_sku,
                     slot.get("width_cm"), slot.get("height_cm"), str(slot.get("image_dir", ""))),
                )
        return [self.format_slot_id(*key) for key in keys]

    def delete_slot(self, slot_id: str) -> Optional[ShelfSlot]:
        """删除固定货位。"""
        slot = self.get_slot_by_id(slot_id)
        self.conn.execute("DELETE FROM shelf_inventory WHERE slot_id=?", (slot_id,))
        self.conn.commit()
        return slot

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
        """按位置获取固定货位。"""
        row = self.conn.execute(
            "SELECT * FROM shelf_inventory "
            "WHERE shelf_id=? AND face=? AND level=? AND y_cm=?",
            (shelf_id, face, level, y_cm)
        ).fetchone()
        return None if row is None else self._slot_from_row(row)

    def get_slot_by_id(self, slot_id: str) -> Optional[ShelfSlot]:
        row = self.conn.execute(
            "SELECT * FROM shelf_inventory WHERE slot_id=?", (slot_id,)
        ).fetchone()
        return None if row is None else self._slot_from_row(row)

    def get_shelf_inventory(self, shelf_id: int) -> List[ShelfSlot]:
        """获取指定货架组的所有固定货位。"""
        rows = self.conn.execute(
            "SELECT * "
            "FROM shelf_inventory WHERE shelf_id=? "
            "ORDER BY face, level, y_cm, slot_id",
            (shelf_id,)
        ).fetchall()
        return [self._slot_from_row(row) for row in rows]

    def get_all_slots(self) -> List[ShelfSlot]:
        rows = self.conn.execute(
            "SELECT * FROM shelf_inventory ORDER BY shelf_id, face, level, y_cm"
        ).fetchall()
        return [self._slot_from_row(row) for row in rows]

    def get_shortage_slots(self) -> List[ShelfSlot]:
        rows = self.conn.execute(
            "SELECT * FROM shelf_inventory WHERE actual_sku IS NULL "
            "ORDER BY shelf_id, face, level, y_cm"
        ).fetchall()
        return [self._slot_from_row(row) for row in rows]

    def get_misplaced_slots(self) -> List[ShelfSlot]:
        rows = self.conn.execute(
            "SELECT * FROM shelf_inventory "
            "WHERE actual_sku IS NOT NULL AND actual_sku != expected_sku "
            "ORDER BY shelf_id, face, level, y_cm"
        ).fetchall()
        return [self._slot_from_row(row) for row in rows]

    def get_shelf_sku_summary(self, shelf_id: int) -> List[Dict[str, Any]]:
        """
        查询指定货架当前实际 SKU 及数量。
        """
        rows = self.conn.execute(
            "SELECT actual_sku AS sku, COUNT(*) as total_qty "
            "FROM shelf_inventory WHERE shelf_id=? AND actual_sku IS NOT NULL "
            "GROUP BY actual_sku ORDER BY actual_sku",
            (shelf_id,)
        ).fetchall()
        return [{"sku": r["sku"], "total_quantity": r["total_qty"]} for r in rows]

    def find_sku_locations(self, sku: str) -> List[Dict[str, Any]]:
        """
        查询: 指定SKU在哪些货架的哪些位置
        """
        rows = self.conn.execute(
            "SELECT si.slot_id, si.shelf_id, sg.name as shelf_name, "
            "si.face, si.level, si.y_cm, si.expected_sku, si.actual_sku, "
            "si.width_cm, si.height_cm, si.image_dir "
            "FROM shelf_inventory si "
            "JOIN shelf_groups sg ON si.shelf_id = sg.id "
            "WHERE si.actual_sku = ? "
            "ORDER BY si.shelf_id, si.face, si.level, si.y_cm",
            (sku,)
        ).fetchall()
        return [{**dict(row), "status": slot_status(row["expected_sku"], row["actual_sku"])} for row in rows]

    def find_expected_sku_locations(self, sku: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT slot_id, shelf_id, face, level, y_cm, expected_sku, actual_sku "
            "FROM shelf_inventory WHERE expected_sku = ? "
            "ORDER BY shelf_id, face, level, y_cm",
            (sku,),
        ).fetchall()
        return [{**dict(row), "status": slot_status(row["expected_sku"], row["actual_sku"])} for row in rows]

    def find_sku_world_positions(self, sku: str) -> List[Dict[str, Any]]:
        """
        查询: 指定SKU所在位置的世界坐标
        """
        rows = self.conn.execute(
            "SELECT si.slot_id, si.shelf_id, sg.name as shelf_name, "
            "sg.world_x, sg.world_y, sg.yaw, "
            "si.face, si.level, si.y_cm, si.expected_sku, si.actual_sku, "
            "si.width_cm, si.height_cm, si.image_dir "
            "FROM shelf_inventory si "
            "JOIN shelf_groups sg ON si.shelf_id = sg.id "
            "WHERE si.actual_sku = ? "
            "ORDER BY si.shelf_id, si.face, si.level, si.y_cm",
            (sku,)
        ).fetchall()

        results = []
        for r in rows:
            local = self.slot_id_to_local(
                r["shelf_id"], r["face"], r["level"], r["y_cm"], r["height_cm"]
            )
            world = self.local_to_world(
                local, r["world_x"], r["world_y"], r["yaw"]
            )
            results.append({
                "slot_id": r["slot_id"],
                "shelf_id": r["shelf_id"],
                "shelf_name": r["shelf_name"],
                "face": r["face"],
                "level": r["level"],
                "y_cm": r["y_cm"],
                "width_cm": r["width_cm"],
                "height_cm": r["height_cm"],
                "image_dir": r["image_dir"],
                "expected_sku": r["expected_sku"],
                "actual_sku": r["actual_sku"],
                "status": slot_status(r["expected_sku"], r["actual_sku"]),
                "world_x": world.x,
                "world_y": world.y,
                "world_z": world.z,
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
            "SELECT COUNT(*) FROM shelf_inventory WHERE actual_sku = ?",
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
            "FROM shelf_inventory WHERE actual_sku = ? "
            "GROUP BY shelf_id ORDER BY shelf_id",
            (sku,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ================================================================
    # 坐标查询
    # ================================================================

    def get_slot_world_pos(self, shelf_id: int, face: int,
                           level: int, y_cm: float,
                           height_cm: Optional[float] = None) -> Optional[WorldPos]:
        """
        获取指定位置的世界坐标
        """
        shelf = self.get_shelf_group(shelf_id)
        if shelf is None:
            return None
        local = self.slot_id_to_local(shelf_id, face, level, y_cm, height_cm)
        return self.local_to_world(local, shelf.world_x, shelf.world_y, shelf.yaw)

    def get_all_slots_world(self) -> List[Dict[str, Any]]:
        """获取所有固定货位的世界坐标和当前状态。"""
        rows = self.conn.execute(
            "SELECT si.slot_id, si.shelf_id, si.face, si.level, si.y_cm, "
            "si.expected_sku, si.actual_sku, si.width_cm, si.height_cm, si.image_dir, "
            "sg.world_x, sg.world_y, sg.yaw "
            "FROM shelf_inventory si "
            "JOIN shelf_groups sg ON si.shelf_id = sg.id "
            "ORDER BY si.shelf_id, si.face, si.level, si.y_cm"
        ).fetchall()

        results = []
        for r in rows:
            local = self.slot_id_to_local(
                r["shelf_id"], r["face"], r["level"], r["y_cm"], r["height_cm"]
            )
            world = self.local_to_world(
                local, r["world_x"], r["world_y"], r["yaw"]
            )
            results.append({
                "slot_id": r["slot_id"],
                "shelf_id": r["shelf_id"],
                "face": r["face"],
                "level": r["level"],
                "y_cm": r["y_cm"],
                "expected_sku": r["expected_sku"],
                "actual_sku": r["actual_sku"],
                "status": slot_status(r["expected_sku"], r["actual_sku"]),
                "width_cm": r["width_cm"],
                "height_cm": r["height_cm"],
                "image_dir": r["image_dir"],
                "world_x": world.x,
                "world_y": world.y,
                "world_z": world.z,
                "yaw": r["yaw"],
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
            "SELECT si.slot_id, si.face, si.level, si.y_cm, si.expected_sku, si.actual_sku, "
            "si.width_cm, si.height_cm, si.image_dir "
            "FROM shelf_inventory si "
            "WHERE si.shelf_id = ? "
            "ORDER BY si.face, si.level, si.y_cm",
            (shelf_id,)
        ).fetchall()

        results = []
        for r in rows:
            local = self.slot_id_to_local(shelf_id, r["face"], r["level"],
                                          r["y_cm"], r["height_cm"])
            world = self.local_to_world(local, shelf.world_x, shelf.world_y, shelf.yaw)
            results.append({
                "slot_id": r["slot_id"],
                "face": r["face"],
                "level": r["level"],
                "y_cm": r["y_cm"],
                "expected_sku": r["expected_sku"],
                "actual_sku": r["actual_sku"],
                "status": slot_status(r["expected_sku"], r["actual_sku"]),
                "width_cm": r["width_cm"],
                "height_cm": r["height_cm"],
                "image_dir": r["image_dir"],
                "world_x": world.x,
                "world_y": world.y,
                "world_z": world.z,
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
        n_delivery_tables = self.conn.execute(
            "SELECT COUNT(*) FROM delivery_tables"
        ).fetchone()[0]
        n_skus = self.conn.execute(
            "SELECT COUNT(*) FROM sku_catalog"
        ).fetchone()[0]
        total_positions = self.conn.execute("SELECT COUNT(*) FROM shelf_inventory").fetchone()[0]
        actual_items = self.conn.execute(
            "SELECT COUNT(*) FROM shelf_inventory WHERE actual_sku IS NOT NULL"
        ).fetchone()[0]
        shortages = self.conn.execute(
            "SELECT COUNT(*) FROM shelf_inventory WHERE actual_sku IS NULL"
        ).fetchone()[0]
        misplacements = self.conn.execute(
            "SELECT COUNT(*) FROM shelf_inventory "
            "WHERE actual_sku IS NOT NULL AND actual_sku != expected_sku"
        ).fetchone()[0]
        return {
            "shelf_types": n_types,
            "shelf_groups": n_shelves,
            "delivery_tables": n_delivery_tables,
            "sku_catalog": n_skus,
            "total_positions": total_positions,
            "actual_items": actual_items,
            "shortages": shortages,
            "misplacements": misplacements,
        }

    @staticmethod
    def format_slot_id(shelf_id: int, face: int, level: int, y_cm: float) -> str:
        """Format the stable four-part, centimetre-based instance identifier."""
        return f"{shelf_id}-{face}-{level}-{y_cm:g}"

    def slot_id_str_to_tuple(self, slot_id_str: str) -> Tuple[int, int, int, float]:
        """将 '0-1-2-9' 格式解析为 (shelf_id, face, level, y_cm)。"""
        parts = slot_id_str.split("-")
        if len(parts) != 4:
            raise ValueError(f"Invalid slot ID string: {slot_id_str}, "
                             f"expected format 'shelf-face-level-ycm'")
        try:
            parsed = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
        except ValueError as error:
            raise ValueError(f"Invalid slot ID string: {slot_id_str}") from error
        if self.format_slot_id(*parsed) != slot_id_str:
            raise ValueError(f"Non-canonical slot ID string: {slot_id_str}")
        return parsed

    def close(self):
        """关闭数据库连接"""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================
# 便捷函数: 从 MuJoCo 场景生成器的参数初始化数据库
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

    # 3. 创建固定货位，示例数据默认均为正常状态
    db.create_slot(S0, 0, 2, 50, "cracker_box", "cracker_box", height_cm=8)
    db.create_slot(S0, 0, 2, 65, "tomato_soup_can", "tomato_soup_can", height_cm=8)
    db.create_slot(S0, 1, 4, 90, "banana", "banana", height_cm=10)
    db.create_slot(S1, 0, 0, 5, "mustard_bottle", "mustard_bottle", height_cm=6)
    db.create_slot(S3, 1, 1, 30, "cracker_box", "cracker_box", height_cm=12)

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
              f"y={loc['y_cm']:.0f}cm height={loc['height_cm'] or 0:.0f}cm")

    print("\n--- 查询: cracker_box的世界坐标 ---")
    for pos in db.find_sku_world_positions("cracker_box"):
        print(f"  {pos['slot_id']}: world=({pos['world_x']:.3f}, "
              f"{pos['world_y']:.3f}, {pos['world_z']:.3f})")

    print(f"\n--- 查询: 位置 {S0}-0-2-65 的商品 ---")
    slot = db.get_slot(S0, 0, 2, 65)
    if slot:
        print(f"  expected={slot.expected_sku}, actual={slot.actual_sku}, status={slot.status}")

    print("\n--- 所有商品世界坐标 (用于MuJoCo生成) ---")
    all_slots = db.get_all_slots_world()
    for s in all_slots:
        print(f"  {s['slot_id']} {s['actual_sku']}: "
              f"world=({s['world_x']:.3f}, {s['world_y']:.3f}, {s['world_z']:.3f})")

    print(f"\n--- 货架{S0}的世界位置 ---")
    pos = db.get_shelf_world_pos(S0)
    print(f"  局部原点: ({pos.x:.3f}, {pos.y:.3f})")

    # 5. 测试删除
    print("\n--- 测试: 删除位置 S0-0-2-65 ---")
    db.delete_slot(db.format_slot_id(S0, 0, 2, 65))
    slot = db.get_slot(S0, 0, 2, 65)
    print(f"  删除后查询: {slot}")

    print("\n--- 测试: 解析 slot ID 字符串 ---")
    parsed = db.slot_id_str_to_tuple("0-1-2-9")
    print(f"  '0-1-2-9' -> {parsed}")

    db.close()
