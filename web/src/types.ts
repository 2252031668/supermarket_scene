export type ShelfType = {
  id: number
  name: string
  shelf_length: number
  shelf_width: number
  shelf_height: number
  num_levels: number
  bottom_clearance: number
  level_spacing: number
  panel_thick: number
  back_thick: number
  shelf_depth_normal: number
  shelf_depth_bottom: number
}

export type Shelf = {
  id: number
  name: string
  world_x: number
  world_y: number
  yaw: number
  shelf_type_id: number | null
  created_at: string
}

export type Sku = {
  sku: string
  category: string
  mesh_file: string
  tex_file: string
}

export type Slot = {
  shelf_id: number
  face: number
  level: number
  y_cm: number
  sku: string
  width_cm: number | null
  height_cm: number | null
  image_dir: string
  world_x: number
  world_y: number
  world_z: number
  yaw: number
  slot_id_str: string
}

export type WarehouseState = {
  stats: { shelf_types: number; shelf_groups: number; sku_catalog: number; total_items: number }
  shelf_types: ShelfType[]
  shelves: Shelf[]
  skus: Sku[]
  slots: Slot[]
}

export type Selection =
  | { kind: 'none' }
  | { kind: 'shelf'; shelfId: number }
  | { kind: 'slot'; slot: Slot }
