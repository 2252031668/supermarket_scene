import { Edges, Html, OrbitControls, OrthographicCamera, PerspectiveCamera, useTexture } from '@react-three/drei'
import { Canvas, ThreeEvent } from '@react-three/fiber'
import { Suspense, useEffect, useMemo } from 'react'
import { Color, MOUSE } from 'three'
import type { DeliveryTable, DeliveryTableSpec, Selection, Shelf, ShelfType, Slot, Sku } from './types'
import { skuColor } from './skuColors'

type SceneProps = {
  shelves: Shelf[]
  deliveryTables: DeliveryTable[]
  deliveryTableSpec: DeliveryTableSpec
  shelfTypes: ShelfType[]
  slots: Slot[]
  skus: Sku[]
  selection: Selection
  camera: 'top' | 'perspective'
  onSelect: (selection: Selection) => void
}

const EMPTY_TEXTURE = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=='

function ShelfModel({ shelf, type, selected, onClick }: {
  shelf: Shelf
  type?: ShelfType
  selected: boolean
  onClick: () => void
}) {
  const width = type?.shelf_width ?? 0.8
  const length = type?.shelf_length ?? 1.86
  const height = type?.shelf_height ?? 1.65
  const levels = type?.num_levels ?? 5
  const clearance = type?.bottom_clearance ?? 0.05
  const spacing = type?.level_spacing ?? 0.4
  const panel = type?.panel_thick ?? 0.02
  const normalDepth = type?.shelf_depth_normal ?? 0.3
  const bottomDepth = type?.shelf_depth_bottom ?? 0.4

  const click = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation()
    onClick()
  }

  return (
    <group position={[shelf.world_x, shelf.world_y, 0]} rotation={[0, 0, shelf.yaw]} onClick={click}>
      <mesh position={[width / 2, length / 2, height / 2]} castShadow receiveShadow>
        <boxGeometry args={[0.03, length, height]} />
        <meshStandardMaterial color={selected ? '#f5b544' : '#4d5964'} metalness={0.35} roughness={0.48} />
      </mesh>
      {Array.from({ length: levels }, (_, level) => {
        const depth = level === 0 ? bottomDepth : normalDepth
        const z = clearance + level * spacing
        return (
          <group key={level}>
            <mesh position={[width / 2 - depth / 2, length / 2, z]} castShadow receiveShadow>
              <boxGeometry args={[depth, length, panel]} />
              <meshStandardMaterial color="#9da8b1" metalness={0.25} roughness={0.65} />
            </mesh>
            <mesh position={[width / 2 + depth / 2, length / 2, z]} castShadow receiveShadow>
              <boxGeometry args={[depth, length, panel]} />
              <meshStandardMaterial color="#9da8b1" metalness={0.25} roughness={0.65} />
            </mesh>
          </group>
        )
      })}
      {[0.05, length - 0.05].map((y) => [width / 2 - 0.03, width / 2 + 0.03].map((x) => (
        <mesh key={`${x}-${y}`} position={[x, y, height / 2]} castShadow>
          <boxGeometry args={[0.04, 0.04, height]} />
          <meshStandardMaterial color="#2d3640" metalness={0.45} roughness={0.4} />
        </mesh>
      )))}
      <CoordinateFrame label={`货架 ${shelf.id} 局部原点`} size={0.28} />
    </group>
  )
}

function DeliveryTableModel({ table, spec, selected, onClick }: {
  table: DeliveryTable
  spec: DeliveryTableSpec
  selected: boolean
  onClick: () => void
}) {
  const click = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation()
    onClick()
  }
  const legOffset = 0.04
  return (
    <group position={[table.world_x, table.world_y, 0]} rotation={[0, 0, table.yaw]} onClick={click}>
      <mesh position={[spec.length / 2, spec.width / 2, spec.height]} castShadow receiveShadow>
        <boxGeometry args={[spec.length, spec.width, spec.top_thickness]} />
        <meshStandardMaterial color={selected ? '#f5b544' : '#65d8d1'} metalness={0.2} roughness={0.46} />
      </mesh>
      {[legOffset, spec.length - legOffset].map((x) => [legOffset, spec.width - legOffset].map((y) => (
        <mesh key={`${x}-${y}`} position={[x, y, spec.height / 2]} castShadow>
          <boxGeometry args={[0.04, 0.04, spec.height]} />
          <meshStandardMaterial color="#4d5964" metalness={0.5} roughness={0.42} />
        </mesh>
      )))}
      <mesh position={[spec.length / 2, spec.width / 2, spec.height + spec.top_thickness / 2 + 0.004]} castShadow>
        <boxGeometry args={[0.16, 0.1, 0.004]} />
        <meshStandardMaterial color="#fbfaf2" roughness={0.88} />
      </mesh>
      <CoordinateFrame label={`交付桌 ${table.id} 局部原点`} size={0.25} />
    </group>
  )
}

function CoordinateFrame({ label, size, world = false }: { label: string; size: number; world?: boolean }) {
  const offset = size + 0.04
  return (
    <group>
      <axesHelper args={[size]} />
      <mesh>
        <sphereGeometry args={[world ? 0.042 : 0.025, 12, 12]} />
        <meshBasicMaterial color={world ? '#ffffff' : '#f5b544'} />
      </mesh>
      <Html position={[0.025, 0.025, 0.04]} center sprite pointerEvents="none">
        <span className="axis-origin-label">{label}</span>
      </Html>
      <Html position={[offset, 0, 0]} center sprite pointerEvents="none"><span className="axis-label axis-x">X</span></Html>
      <Html position={[0, offset, 0]} center sprite pointerEvents="none"><span className="axis-label axis-y">Y</span></Html>
      <Html position={[0, 0, offset]} center sprite pointerEvents="none"><span className="axis-label axis-z">Z</span></Html>
    </group>
  )
}

function ItemBox({ slot, color, selected, onClick }: {
  slot: Slot
  color: string
  selected: boolean
  onClick: () => void
}) {
  const displayColor = selected ? '#fff1b8' : color
  const imageUrl = slot.image_dir ? `/api/item-images/${encodeURIComponent(slot.slot_id)}/0.png` : EMPTY_TEXTURE
  const texture = useTexture(imageUrl)
  useEffect(() => {
    if (!slot.image_dir) return
    // X-facing BoxGeometry UVs use Z as U and Y as V. Rotate so the photo's
    // horizontal axis follows product width (local Y), and its vertical axis Z.
    texture.center.set(0.5, 0.5)
    texture.rotation = slot.face === 1 ? Math.PI / 2 : -Math.PI / 2
    texture.needsUpdate = true
  }, [slot.face, slot.image_dir, texture])
  const click = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation()
    onClick()
  }
  return (
    <mesh
      position={[slot.world_x, slot.world_y, slot.world_z]}
      rotation={[0, 0, slot.yaw]}
      castShadow
      onClick={click}
    >
      <boxGeometry args={[0.03, Math.max(0.04, (slot.width_cm ?? 7.5) / 100), Math.max(0.04, (slot.height_cm ?? 14) / 100)]} />
      {/* BoxGeometry material 0 is +X and material 1 is -X. Only the outward shelf face carries 0.png. */}
      <meshStandardMaterial attach="material-0" color={slot.image_dir && slot.face === 1 ? '#ffffff' : displayColor} map={slot.image_dir && slot.face === 1 ? texture : undefined} emissive={selected ? new Color('#9c6e12') : new Color('#000000')} emissiveIntensity={selected ? 0.26 : 0} roughness={0.48} />
      <meshStandardMaterial attach="material-1" color={slot.image_dir && slot.face === 0 ? '#ffffff' : displayColor} map={slot.image_dir && slot.face === 0 ? texture : undefined} emissive={selected ? new Color('#9c6e12') : new Color('#000000')} emissiveIntensity={selected ? 0.26 : 0} roughness={0.48} />
      <meshStandardMaterial attach="material-2" color={displayColor} emissive={selected ? new Color('#9c6e12') : new Color('#000000')} emissiveIntensity={selected ? 0.26 : 0} roughness={0.48} />
      <meshStandardMaterial attach="material-3" color={displayColor} emissive={selected ? new Color('#9c6e12') : new Color('#000000')} emissiveIntensity={selected ? 0.26 : 0} roughness={0.48} />
      <meshStandardMaterial attach="material-4" color={displayColor} emissive={selected ? new Color('#9c6e12') : new Color('#000000')} emissiveIntensity={selected ? 0.26 : 0} roughness={0.48} />
      <meshStandardMaterial attach="material-5" color={displayColor} emissive={selected ? new Color('#9c6e12') : new Color('#000000')} emissiveIntensity={selected ? 0.26 : 0} roughness={0.48} />
      {slot.status === '摆放错误' && <Edges color="#d94b46" lineWidth={2} />}
    </mesh>
  )
}

function EmptySlotMarker({ slot, selected, onClick }: { slot: Slot; selected: boolean; onClick: () => void }) {
  const click = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation()
    onClick()
  }
  return <mesh position={[slot.world_x, slot.world_y, slot.world_z]} rotation={[0, 0, slot.yaw]} onClick={click}>
    <boxGeometry args={[0.035, Math.max(0.05, (slot.width_cm ?? 7.5) / 100), Math.max(0.05, (slot.height_cm ?? 14) / 100)]} />
    <meshStandardMaterial color={selected ? '#f5b544' : '#f0b14a'} transparent opacity={selected ? 0.34 : 0.16} wireframe />
    <Edges color={selected ? '#8c5b05' : '#c47d18'} lineWidth={selected ? 2 : 1} />
  </mesh>
}

function SceneContent(props: SceneProps) {
  const colorBySku = useMemo(() => new Map(props.skus.map((sku, index) => [sku.sku, skuColor(index)])), [props.skus])
  const typeById = useMemo(() => new Map(props.shelfTypes.map((type) => [type.id, type])), [props.shelfTypes])
  const selectedSlotId = props.selection.kind === 'slot' ? props.selection.slot.slot_id : undefined
  const selectedShelfId = props.selection.kind === 'shelf'
    ? props.selection.shelfId
    : props.selection.kind === 'slot' ? props.selection.slot.shelf_id : undefined
  const selectedDeliveryTableId = props.selection.kind === 'delivery-table' ? props.selection.tableId : undefined

  return (
    <>
      <color attach="background" args={['#e7edf0']} />
      <ambientLight intensity={0.62} />
      <directionalLight position={[5, -4, 9]} intensity={1.35} castShadow shadow-mapSize={[2048, 2048]} />
      <hemisphereLight args={['#dce9f3', '#8c9a87', 0.55]} />
      {props.camera === 'top'
        ? <OrthographicCamera makeDefault position={[0, 0, 9]} up={[0, 0, 1]} zoom={92} near={0.1} far={100} />
        : <PerspectiveCamera makeDefault position={[5.6, -7.4, 7.8]} up={[0, 0, 1]} fov={39} />}
      <OrbitControls
        enableRotate={props.camera === 'perspective'}
        enablePan
        enableDamping
        dampingFactor={0.08}
        minDistance={2.5}
        maxDistance={15}
        minPolarAngle={0.08}
        maxPolarAngle={Math.PI - 0.12}
        minZoom={45}
        maxZoom={180}
        target={[0, 0, 0.7]}
        mouseButtons={{ LEFT: MOUSE.ROTATE, MIDDLE: MOUSE.PAN, RIGHT: undefined }}
      />
      <gridHelper args={[8, 32, '#9fafb6', '#cbd4d8']} position={[0, 0, -0.012]} rotation={[Math.PI / 2, 0, 0]} />
      <mesh position={[0, 0, -0.04]} receiveShadow onClick={() => props.onSelect({ kind: 'none' })}>
        <boxGeometry args={[6, 8, 0.05]} />
        <meshStandardMaterial color="#dfe6e5" roughness={0.9} />
      </mesh>
      <CoordinateFrame label="WORLD (0, 0, 0)" size={0.72} world />
      {props.shelves.map((shelf) => (
        <ShelfModel
          key={shelf.id}
          shelf={shelf}
          type={shelf.shelf_type_id ? typeById.get(shelf.shelf_type_id) : undefined}
          selected={shelf.id === selectedShelfId}
          onClick={() => props.onSelect({ kind: 'shelf', shelfId: shelf.id })}
        />
      ))}
      {props.deliveryTables.map((table) => (
        <DeliveryTableModel
          key={table.id}
          table={table}
          spec={props.deliveryTableSpec}
          selected={table.id === selectedDeliveryTableId}
          onClick={() => props.onSelect({ kind: 'delivery-table', tableId: table.id })}
        />
      ))}
      <Suspense fallback={null}>{props.slots.map((slot) => slot.actual_sku === null
        ? <EmptySlotMarker key={slot.slot_id} slot={slot} selected={slot.slot_id === selectedSlotId} onClick={() => props.onSelect({ kind: 'slot', slot })} />
        : <ItemBox key={slot.slot_id} slot={slot} color={colorBySku.get(slot.actual_sku) ?? '#a9b5c1'} selected={slot.slot_id === selectedSlotId} onClick={() => props.onSelect({ kind: 'slot', slot })} />
      )}</Suspense>
    </>
  )
}

export function WarehouseScene(props: SceneProps) {
  return (
    <Canvas shadows dpr={[1, 2]} gl={{ antialias: true }}>
      <SceneContent {...props} />
    </Canvas>
  )
}
