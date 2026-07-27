import { Html, OrbitControls, OrthographicCamera, PerspectiveCamera } from '@react-three/drei'
import { Canvas, ThreeEvent } from '@react-three/fiber'
import { useMemo } from 'react'
import { Color, MOUSE } from 'three'
import type { Selection, Shelf, ShelfType, Slot, Sku } from './types'
import { skuColor } from './skuColors'

type SceneProps = {
  shelves: Shelf[]
  shelfTypes: ShelfType[]
  slots: Slot[]
  skus: Sku[]
  selection: Selection
  camera: 'top' | 'perspective'
  onSelect: (selection: Selection) => void
}

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
  const click = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation()
    onClick()
  }
  return (
    <mesh
      position={[slot.world_x, slot.world_y, slot.world_z + 0.07]}
      rotation={[0, 0, slot.yaw]}
      castShadow
      onClick={click}
    >
      <boxGeometry args={[0.13, 0.075, 0.14]} />
      <meshStandardMaterial color={displayColor} emissive={selected ? new Color('#9c6e12') : new Color('#000000')} emissiveIntensity={selected ? 0.26 : 0} roughness={0.48} />
    </mesh>
  )
}

function SceneContent(props: SceneProps) {
  const colorBySku = useMemo(() => new Map(props.skus.map((sku, index) => [sku.sku, skuColor(index)])), [props.skus])
  const typeById = useMemo(() => new Map(props.shelfTypes.map((type) => [type.id, type])), [props.shelfTypes])
  const selectedSlotId = props.selection.kind === 'slot' ? props.selection.slot.slot_id_str : undefined
  const selectedShelfId = props.selection.kind === 'shelf'
    ? props.selection.shelfId
    : props.selection.kind === 'slot' ? props.selection.slot.shelf_id : undefined

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
      {props.slots.map((slot) => (
        <ItemBox
          key={slot.slot_id_str}
          slot={slot}
          color={colorBySku.get(slot.sku) ?? '#a9b5c1'}
          selected={slot.slot_id_str === selectedSlotId}
          onClick={() => props.onSelect({ kind: 'slot', slot })}
        />
      ))}
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
