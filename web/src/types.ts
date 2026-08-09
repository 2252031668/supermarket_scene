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

export type DeliveryTable = {
  id: number
  name: string
  world_x: number
  world_y: number
  yaw: number
  created_at: string
}

export type DeliveryTableSpec = {
  length: number
  width: number
  height: number
  top_thickness: number
}

export type ShelfImage = {
  shelf_id: number
  face: 0 | 1
}

export type Sku = {
  sku: string
  category: string
  mesh_file: string
  tex_file: string
}

export type SlotStatus = '正常' | '缺货' | '摆放错误'

export type Slot = {
  slot_id: string
  shelf_id: number
  face: number
  level: number
  y_cm: number
  expected_sku: string
  actual_sku: string | null
  status: SlotStatus
  width_cm: number | null
  height_cm: number | null
  image_dir: string
  world_x: number
  world_y: number
  world_z: number
  yaw: number
}

export type VisionConfig = {
  min_current_coverage: number
  analysis_center_ratio: number
  lab_distance_threshold: number
  slot_change_ratio_threshold: number
  dino_confidence_threshold: number
  ambiguity_margin: number
  vlm_fallback: boolean
  vlm_top_k: number
}

export type VisionInspectionRow = {
  slot_id: string
  expected_sku: string
  actual_sku: string | null
  status: SlotStatus
  source: string
  confidence: number | null
  reason: string
  selected: boolean
  difference_ratio: number
}

export type VisionInspectionReport = {
  run_id: string
  shelf_id: number
  face: number
  slots: VisionInspectionRow[]
  analysis?: { center_ratio: number; roi: { x: number; y: number; width: number; height: number }; skipped_edge_slots: number }
  artifacts: Record<string, string>
}

export type SkuQueryBox = {
  index: number
  label: string
  normalized: [number, number, number, number]
  pixels: [number, number, number, number]
}

export type SkuQueryConfig = {
  max_boxes: number
  dino_fallback: boolean
  dino_confidence_threshold: number
}

export type SkuQueryReport = {
  run_id: string
  query: string
  sku: string
  reference_slot_id: string
  provider: string
  model: string
  request_seconds: number
  total_seconds: number
  raw_response: string
  vlm_detected_boxes: SkuQueryBox[]
  detected_boxes: SkuQueryBox[]
  dino_scores: { index: number; confidence: number }[]
  config: SkuQueryConfig
  artifacts: Record<string, string>
}

export type ImageStitchReport = {
  run_id: string
  width: number
  height: number
  main_index: number
  pairs: Array<{ source_index: number; destination_index: number; matches: number; inliers: number; inlier_ratio: number }>
  used_indices: number[]
  skipped: Array<{ index: number; reason: string }>
  rendering: { seam_method: string; blend_bands: number; rejected_indices: number[]; seam_overlap_pixels: number[] }
  artifacts: Record<string, string>
}

export type WarehouseState = {
  stats: { shelf_types: number; shelf_groups: number; delivery_tables: number; sku_catalog: number; total_positions: number; actual_items: number; shortages: number; misplacements: number }
  shelf_types: ShelfType[]
  shelves: Shelf[]
  shelf_images: ShelfImage[]
  delivery_tables: DeliveryTable[]
  delivery_table_spec: DeliveryTableSpec
  skus: Sku[]
  slots: Slot[]
}

export type Selection =
  | { kind: 'none' }
  | { kind: 'shelf'; shelfId: number }
  | { kind: 'delivery-table'; tableId: number }
  | { kind: 'slot'; slot: Slot }
