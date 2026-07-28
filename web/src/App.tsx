import { useEffect, useMemo, useState } from 'react'
import { Box, Boxes, ChevronDown, ChevronRight, Eye, Layers3, LoaderCircle, Map, PackagePlus, Pencil, Plus, RefreshCw, Save, Trash2, Upload, X } from 'lucide-react'
import { WarehouseScene } from './WarehouseScene'
import { ManualImportPage } from './ManualImportPage'
import { skuColor } from './skuColors'
import type { Selection, Shelf, Slot, WarehouseState } from './types'

const blankState: WarehouseState = {
  stats: { shelf_types: 0, shelf_groups: 0, sku_catalog: 0, total_items: 0 },
  shelf_types: [], shelves: [], skus: [], slots: [],
}

type SlotDraft = Pick<Slot, 'shelf_id' | 'face' | 'level' | 'y_cm' | 'sku' | 'width_cm' | 'height_cm' | 'image_dir'>
type SkuDraft = Pick<WarehouseState['skus'][number], 'sku' | 'category' | 'mesh_file' | 'tex_file'>
type CleanupScope = 'level' | 'face' | 'all' | 'shelf'
type SlotWorldPosition = Pick<Slot, 'slot_id_str' | 'shelf_id' | 'face' | 'level' | 'y_cm' | 'sku' | 'width_cm' | 'height_cm' | 'image_dir' | 'world_x' | 'world_y' | 'world_z'> & { frame: string }

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options?.headers ?? {}) },
  })
  const body = await response.json() as T & { error?: string }
  if (!response.ok) throw new Error(body.error ?? 'Request failed')
  return body
}

function number(value: string) {
  return Number.isFinite(Number(value)) ? Number(value) : 0
}

export function App() {
  const [state, setState] = useState<WarehouseState>(blankState)
  const [selection, setSelection] = useState<Selection>({ kind: 'none' })
  const [camera, setCamera] = useState<'top' | 'perspective'>('perspective')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [notice, setNotice] = useState('')
  const [slotDraft, setSlotDraft] = useState<SlotDraft | null>(null)
  const [shelfDraft, setShelfDraft] = useState<Shelf | null>(null)
  const [skuDraft, setSkuDraft] = useState<SkuDraft | null>(null)
  const [cleanupShelfId, setCleanupShelfId] = useState<number | null>(null)
  const [slotWorldPosition, setSlotWorldPosition] = useState<SlotWorldPosition | null>(null)
  const [manualImport, setManualImport] = useState(false)

  const selectedShelf = useMemo(() => {
    if (selection.kind === 'shelf') return state.shelves.find((shelf) => shelf.id === selection.shelfId)
    if (selection.kind === 'slot') return state.shelves.find((shelf) => shelf.id === selection.slot.shelf_id)
    return undefined
  }, [selection, state.shelves])

  const refresh = async (showNotice = false) => {
    setLoading(true)
    try {
      const next = await request<WarehouseState>('/api/state')
      setState(next)
      setSelection((current) => {
        if (current.kind === 'slot') {
          const slot = next.slots.find((item) => item.slot_id_str === current.slot.slot_id_str)
          return slot ? { kind: 'slot', slot } : { kind: 'none' }
        }
        if (current.kind === 'shelf') {
          return next.shelves.some((shelf) => shelf.id === current.shelfId) ? current : { kind: 'none' }
        }
        return current
      })
      if (showNotice) setNotice('数据已从数据库刷新')
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '无法读取数据库')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { void refresh() }, [])
  useEffect(() => {
    if (selection.kind === 'shelf') setShelfDraft(state.shelves.find((shelf) => shelf.id === selection.shelfId) ?? null)
    else setShelfDraft(null)
    if (selection.kind === 'slot') {
      const { shelf_id, face, level, y_cm, sku, width_cm, height_cm, image_dir } = selection.slot
      setSlotDraft({ shelf_id, face, level, y_cm, sku, width_cm, height_cm, image_dir })
    }
  }, [selection, state.shelves])
  useEffect(() => {
    if (selection.kind !== 'slot') {
      setSlotWorldPosition(null)
      return
    }
    let active = true
    void request<{ slot: SlotWorldPosition }>(`/api/slots/${encodeURIComponent(selection.slot.slot_id_str)}/world-position`)
      .then((result) => { if (active) setSlotWorldPosition(result.slot) })
      .catch(() => { if (active) setSlotWorldPosition(null) })
    return () => { active = false }
  }, [selection])

  const commit = async (path: string, method: string, payload?: unknown) => {
    setSaving(true)
    setNotice('')
    try {
      const result = await request<{ state: WarehouseState; id?: number; removed?: number }>(path, {
        method,
        body: payload ? JSON.stringify(payload) : undefined,
      })
      setState(result.state)
      return result
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '操作失败')
      return undefined
    } finally {
      setSaving(false)
    }
  }

  const beginNewSlot = () => {
    if (!selectedShelf || state.skus.length === 0) return
    setSelection({ kind: 'none' })
    setSlotDraft({ shelf_id: selectedShelf.id, face: 0, level: 0, y_cm: 20, sku: state.skus[0].sku, width_cm: null, height_cm: 16, image_dir: '' })
  }

  const saveSlot = async () => {
    if (!slotDraft) return
    const previous = selection.kind === 'slot'
      ? { shelf_id: selection.slot.shelf_id, face: selection.slot.face, level: selection.slot.level, y_cm: selection.slot.y_cm }
      : undefined
    const result = await commit('/api/slots', 'PUT', { ...slotDraft, previous })
    if (result) {
      const match = result.state.slots.find((slot) => slot.shelf_id === slotDraft.shelf_id && slot.face === slotDraft.face && slot.level === slotDraft.level && slot.y_cm === slotDraft.y_cm)
      setSelection(match ? { kind: 'slot', slot: match } : { kind: 'shelf', shelfId: slotDraft.shelf_id })
    }
  }

  const deleteSlot = async () => {
    if (selection.kind !== 'slot' || !window.confirm('删除这个商品货位？')) return
    const result = await commit('/api/slots', 'DELETE', selection.slot)
    if (result) setSelection({ kind: 'shelf', shelfId: selection.slot.shelf_id })
  }

  const saveShelf = async () => {
    if (!shelfDraft) return
    const result = await commit(`/api/shelves/${shelfDraft.id}`, 'PUT', shelfDraft)
    if (result) setSelection({ kind: 'shelf', shelfId: shelfDraft.id })
  }

  const beginNewSku = () => {
    setSkuDraft({ sku: '', category: '', mesh_file: '', tex_file: '' })
  }

  const saveSku = async () => {
    if (!skuDraft) return
    const sku = skuDraft.sku.trim()
    if (!sku) {
      setNotice('请输入 SKU 名称')
      return
    }
    const result = await commit('/api/skus', 'POST', { ...skuDraft, sku })
    if (result) {
      setSkuDraft(null)
      setNotice(`已新增 SKU: ${sku}`)
    }
  }

  const deleteFromShelf = async (shelf: Shelf, scope: CleanupScope, face: number, level: number, itemCount: number) => {
    const description = scope === 'level'
      ? `${face === 0 ? '-X 侧' : '+X 侧'}第 ${level} 层`
      : scope === 'face'
        ? `${face === 0 ? '-X 侧' : '+X 侧'}整面`
        : scope === 'all' ? '全部货位' : '整个货架及其货位'
    if (!window.confirm(`确认删除“${shelf.name}”的${description}？将删除 ${itemCount} 个商品实例。`)) return
    const result = scope === 'shelf'
      ? await commit(`/api/shelves/${shelf.id}`, 'DELETE')
      : await commit(`/api/shelves/${shelf.id}/inventory`, 'DELETE', { scope, face, level })
    if (result) {
      setCleanupShelfId(null)
      setNotice(`已删除 ${result.removed ?? itemCount} 个商品实例`)
      setSelection(scope === 'shelf' ? { kind: 'none' } : { kind: 'shelf', shelfId: shelf.id })
    }
  }

  const addShelf = async () => {
    const offset = state.shelves.length * 0.3
    const type = state.shelf_types[0]
    const nextShelfNumber = Math.max(0, ...state.shelves.map((shelf) => shelf.id)) + 1
    const result = await commit('/api/shelves', 'POST', {
      name: `${nextShelfNumber}号货架`,
      world_x: -0.4 + offset,
      world_y: -0.9 - offset,
      yaw: 0,
      shelf_type_id: type?.id ?? null,
    })
    if (result?.id) setSelection({ kind: 'shelf', shelfId: result.id })
  }

  if (manualImport) return <ManualImportPage state={state} onBack={() => setManualImport(false)} onImported={(next) => setState(next)} />

  return (
    <main className="app-shell">
      <header className="app-header">
        <div className="brand"><Boxes size={22} strokeWidth={2.4} /><span>仓库货架管理</span><small>LIVE DATABASE</small></div>
        <div className="header-actions">
          <div className="view-toggle" role="group" aria-label="视图模式">
            <button className={camera === 'perspective' ? 'active' : ''} aria-pressed={camera === 'perspective'} onClick={() => setCamera('perspective')}><Eye size={16} />3D 视图</button>
            <button className={camera === 'top' ? 'active' : ''} aria-pressed={camera === 'top'} onClick={() => setCamera('top')}><Map size={16} />平面图</button>
          </div>
          <button className="refresh-button" title="重新读取数据库" onClick={() => void refresh(true)}><RefreshCw size={16} className={loading ? 'spin' : ''} /><span>刷新数据</span></button>
          <button className="manual-entry-button" title="通过照片人工批量录入商品" onClick={() => setManualImport(true)}><Upload size={16} /><span>人工批量录入</span></button>
        </div>
      </header>

      <section className="workspace">
        <section className="visual-panel" aria-label="仓库三维视图">
          <WarehouseScene shelves={state.shelves} shelfTypes={state.shelf_types} slots={state.slots} skus={state.skus} selection={selection} camera={camera} onSelect={(next) => { setSelection(next); if (next.kind === 'none') setSlotDraft(null) }} />
          <div className="scene-caption"><Layers3 size={15} />点击货架或商品以编辑</div>
        </section>

        <aside className="control-panel">
          {notice && <div className="notice"><span>{notice}</span><button title="关闭提示" onClick={() => setNotice('')}><X size={15} /></button></div>}
          {saving && <div className="saving"><LoaderCircle size={16} className="spin" /> 正在保存</div>}
          {cleanupShelfId !== null && state.shelves.find((shelf) => shelf.id === cleanupShelfId) ? (
            <CleanupEditor
              shelf={state.shelves.find((shelf) => shelf.id === cleanupShelfId)!}
              shelfType={state.shelf_types.find((type) => type.id === state.shelves.find((shelf) => shelf.id === cleanupShelfId)!.shelf_type_id)}
              slots={state.slots}
              onConfirm={(scope, face, level, itemCount) => void deleteFromShelf(state.shelves.find((shelf) => shelf.id === cleanupShelfId)!, scope, face, level, itemCount)}
              onClose={() => setCleanupShelfId(null)}
            />
          ) : skuDraft ? (
            <SkuEditor draft={skuDraft} onChange={setSkuDraft} onSave={() => void saveSku()} onClose={() => setSkuDraft(null)} />
          ) : selection.kind === 'slot' && slotDraft ? (
            <SlotEditor draft={slotDraft} skus={state.skus} worldPosition={slotWorldPosition} onChange={setSlotDraft} onSave={() => void saveSlot()} onDelete={() => void deleteSlot()} onClose={() => setSelection({ kind: 'shelf', shelfId: slotDraft.shelf_id })} />
          ) : selection.kind === 'shelf' && shelfDraft ? (
            <ShelfEditor draft={shelfDraft} types={state.shelf_types} itemCount={state.slots.filter((slot) => slot.shelf_id === shelfDraft.id).length} onChange={setShelfDraft} onSave={() => void saveShelf()} onCleanup={() => setCleanupShelfId(shelfDraft.id)} onAddSlot={beginNewSlot} />
          ) : slotDraft ? (
            <SlotEditor draft={slotDraft} skus={state.skus} onChange={setSlotDraft} onSave={() => void saveSlot()} onDelete={() => setSlotDraft(null)} onClose={() => setSlotDraft(null)} isNew />
          ) : (
            <Overview state={state} onSelectShelf={(shelfId) => setSelection({ kind: 'shelf', shelfId })} onAddShelf={() => void addShelf()} onAddSku={beginNewSku} />
          )}
        </aside>
      </section>
    </main>
  )
}

function PanelHeader({ icon, title, eyebrow, action }: { icon: React.ReactNode; title: string; eyebrow: string; action?: React.ReactNode }) {
  return <div className="panel-header"><div className="panel-title"><span className="panel-icon">{icon}</span><div><p>{eyebrow}</p><h1>{title}</h1></div></div>{action}</div>
}

function Overview({ state, onSelectShelf, onAddShelf, onAddSku }: { state: WarehouseState; onSelectShelf: (id: number) => void; onAddShelf: () => void; onAddSku: () => void }) {
  const [skuLegendOpen, setSkuLegendOpen] = useState(true)

  return <div className="overview-panel">
    <PanelHeader icon={<Boxes size={19} />} title="仓库总览" eyebrow="SHELF INVENTORY" action={<button className="icon-button" title="新建货架" onClick={onAddShelf}><Plus size={18} /></button>} />
    <div className="metrics">
      <Metric label="货架组" value={state.stats.shelf_groups} />
      <Metric label="库存货位" value={state.stats.total_items} />
      <Metric label="SKU 种类" value={state.stats.sku_catalog} />
    </div>
    <div className="section-heading"><span>货架组</span><span>{state.shelves.length}</span></div>
    <div className="shelf-list">
      {state.shelves.map((shelf) => <button className="shelf-row" key={shelf.id} onClick={() => onSelectShelf(shelf.id)}><span className="shelf-id">{String(shelf.id).padStart(2, '0')}</span><span><strong>{shelf.name}</strong><small>X {shelf.world_x.toFixed(2)} / Y {shelf.world_y.toFixed(2)}</small></span><ChevronRight size={17} /></button>)}
    </div>
    <div className="sku-legend-header"><button className="sku-legend-toggle" type="button" aria-expanded={skuLegendOpen} onClick={() => setSkuLegendOpen((open) => !open)}>
      <span>SKU 图例</span><span className="sku-legend-count">{state.skus.length}<ChevronDown size={15} className={skuLegendOpen ? '' : 'collapsed'} /></span>
    </button><button className="icon-button sku-add-button" title="新增 SKU" onClick={onAddSku}><Plus size={15} /></button></div>
    {skuLegendOpen && <div className="sku-summary" aria-label="SKU 图例列表">{state.skus.map((sku, index) => {
      const itemCount = state.slots.filter((slot) => slot.sku === sku.sku).length
      return <div className="sku-row" key={sku.sku}><span className="sku-swatch" style={{ backgroundColor: skuColor(index) }} /><strong>{sku.sku}</strong><small>{itemCount} 个实例</small></div>
    })}</div>}
  </div>
}

function Metric({ label, value }: { label: string; value: number }) { return <div className="metric"><strong>{value}</strong><span>{label}</span></div> }

function ShelfEditor({ draft, types, itemCount, onChange, onSave, onCleanup, onAddSlot }: { draft: Shelf; types: WarehouseState['shelf_types']; itemCount: number; onChange: (value: Shelf) => void; onSave: () => void; onCleanup: () => void; onAddSlot: () => void }) {
  const set = <K extends keyof Shelf>(key: K, value: Shelf[K]) => onChange({ ...draft, [key]: value })
  return <>
    <PanelHeader icon={<Layers3 size={19} />} title={draft.name || `货架 #${draft.id}`} eyebrow={`SHELF GROUP · ${itemCount} ITEMS`} action={<button className="danger-icon" title="删除或清理库存" onClick={onCleanup}><Trash2 size={17} /></button>} />
    <div className="field-grid"><Field label="名称" value={draft.name} onChange={(value) => set('name', value)} wide /><Field label="世界 X (m)" value={draft.world_x} onChange={(value) => set('world_x', number(value))} /><Field label="世界 Y (m)" value={draft.world_y} onChange={(value) => set('world_y', number(value))} /><Field label="朝向 (rad)" value={draft.yaw} onChange={(value) => set('yaw', number(value))} /><label className="field"><span>货架类型</span><select value={draft.shelf_type_id ?? ''} onChange={(event) => set('shelf_type_id', event.target.value ? Number(event.target.value) : null)}>{types.map((type) => <option key={type.id} value={type.id}>{type.name}</option>)}</select></label></div>
    <div className="panel-actions"><button className="primary-button" onClick={onSave}><Save size={16} />保存货架</button><button className="secondary-button" onClick={onAddSlot}><PackagePlus size={16} />添加商品</button></div>
  </>
}

function SkuEditor({ draft, onChange, onSave, onClose }: { draft: SkuDraft; onChange: (value: SkuDraft) => void; onSave: () => void; onClose: () => void }) {
  const set = <K extends keyof SkuDraft>(key: K, value: SkuDraft[K]) => onChange({ ...draft, [key]: value })
  return <>
    <PanelHeader icon={<Box size={19} />} title="新增 SKU" eyebrow="SKU CATALOG" action={<button className="icon-button" title="返回总览" onClick={onClose}><X size={17} /></button>} />
    <div className="field-grid"><Field label="SKU 名称" value={draft.sku} onChange={(value) => set('sku', value)} wide /><Field label="分类（可选）" value={draft.category} onChange={(value) => set('category', value)} /><Field label="网格文件（可选）" value={draft.mesh_file} onChange={(value) => set('mesh_file', value)} wide /><Field label="纹理文件（可选）" value={draft.tex_file} onChange={(value) => set('tex_file', value)} wide /></div>
    <div className="panel-actions"><button className="primary-button" onClick={onSave}><Save size={16} />创建 SKU</button></div>
  </>
}

function CleanupEditor({ shelf, shelfType, slots, onConfirm, onClose }: { shelf: Shelf; shelfType?: WarehouseState['shelf_types'][number]; slots: Slot[]; onConfirm: (scope: CleanupScope, face: number, level: number, itemCount: number) => void; onClose: () => void }) {
  const [scope, setScope] = useState<CleanupScope>('level')
  const [face, setFace] = useState(0)
  const [level, setLevel] = useState(0)
  const levels = shelfType?.num_levels ?? 5
  const shelfSlots = slots.filter((slot) => slot.shelf_id === shelf.id)
  const targetSlots = shelfSlots.filter((slot) => {
    if (scope === 'level') return slot.face === face && slot.level === level
    if (scope === 'face') return slot.face === face
    return true
  })
  const targetName = scope === 'level'
    ? `${face === 0 ? '-X 侧' : '+X 侧'}第 ${level} 层`
    : scope === 'face' ? `${face === 0 ? '-X 侧' : '+X 侧'}整面`
      : scope === 'all' ? '全部货位' : '整个货架及其货位'
  const actionLabel = scope === 'shelf' ? '删除货架' : '删除商品实例'

  return <>
    <PanelHeader icon={<Trash2 size={19} />} title="删除与清理" eyebrow={shelf.name} action={<button className="icon-button" title="返回货架" onClick={onClose}><X size={17} /></button>} />
    <div className="cleanup-scope" role="group" aria-label="删除范围">
      <button className={scope === 'level' ? 'active' : ''} onClick={() => setScope('level')}>指定层</button><button className={scope === 'face' ? 'active' : ''} onClick={() => setScope('face')}>整面</button><button className={scope === 'all' ? 'active' : ''} onClick={() => setScope('all')}>全部货位</button><button className={scope === 'shelf' ? 'active' : ''} onClick={() => setScope('shelf')}>整个货架</button>
    </div>
    {(scope === 'level' || scope === 'face') && <div className="field-grid cleanup-fields"><label className="field"><span>货架面</span><select value={face} onChange={(event) => setFace(Number(event.target.value))}><option value={0}>-X 侧</option><option value={1}>+X 侧</option></select></label>{scope === 'level' && <label className="field"><span>层号</span><select value={level} onChange={(event) => setLevel(Number(event.target.value))}>{Array.from({ length: levels }, (_, index) => <option key={index} value={index}>第 {index} 层</option>)}</select></label>}</div>}
    <div className="cleanup-preview"><span>{targetName}</span><strong>将删除 {targetSlots.length} 个商品实例</strong></div>
    <button className="danger-button" onClick={() => onConfirm(scope, face, level, targetSlots.length)}><Trash2 size={16} />{actionLabel}</button>
  </>
}

function SlotEditor({ draft, skus, worldPosition, onChange, onSave, onDelete, onClose, isNew = false }: { draft: SlotDraft; skus: WarehouseState['skus']; worldPosition?: SlotWorldPosition | null; onChange: (value: SlotDraft) => void; onSave: () => void; onDelete: () => void; onClose: () => void; isNew?: boolean }) {
  const set = <K extends keyof SlotDraft>(key: K, value: SlotDraft[K]) => onChange({ ...draft, [key]: value })
  return <>
    <PanelHeader icon={<Box size={19} />} title={isNew ? '新增商品货位' : draft.sku} eyebrow={isNew ? 'NEW INVENTORY SLOT' : `SLOT · ${draft.shelf_id}-${draft.face}-${draft.level}-${draft.y_cm}`} action={<button className="icon-button" title="返回货架" onClick={onClose}><X size={17} /></button>} />
    <div className="field-grid"><label className="field wide"><span>SKU</span><select value={draft.sku} onChange={(event) => set('sku', event.target.value)}>{skus.map((sku) => <option key={sku.sku} value={sku.sku}>{sku.sku}</option>)}</select></label><Field label="货架 ID" value={draft.shelf_id} onChange={(value) => set('shelf_id', number(value))} /><label className="field"><span>货架面</span><select value={draft.face} onChange={(event) => set('face', Number(event.target.value))}><option value={0}>-X 侧 (0)</option><option value={1}>+X 侧 (1)</option></select></label><Field label="层号" value={draft.level} onChange={(value) => set('level', number(value))} /><Field label="Y 中心 (cm)" value={draft.y_cm} onChange={(value) => set('y_cm', number(value))} /><Field label="宽度 (cm，可选)" value={draft.width_cm ?? ''} onChange={(value) => set('width_cm', value === '' ? null : number(value))} /><Field label="高度 (cm，可选)" value={draft.height_cm ?? ''} onChange={(value) => set('height_cm', value === '' ? null : number(value))} /></div>
    <p className="helper-text">商品世界坐标指向几何中心；中心高度由本层层板表面加上商品高度的一半计算。</p>
    {!isNew && <div className="world-position"><span>世界坐标（{worldPosition?.frame ?? 'map'}，m）</span>{worldPosition ? <div><strong>X {worldPosition.world_x.toFixed(3)}</strong><strong>Y {worldPosition.world_y.toFixed(3)}</strong><strong>Z {worldPosition.world_z.toFixed(3)}</strong></div> : <small>正在读取坐标</small>}</div>}
    <div className="panel-actions"><button className="primary-button" onClick={onSave}><Save size={16} />{isNew ? '创建货位' : '保存货位'}</button>{!isNew && <button className="danger-icon" title="删除商品货位" onClick={onDelete}><Trash2 size={17} /></button>}</div>
  </>
}

function Field({ label, value, onChange, wide = false }: { label: string; value: string | number; onChange: (value: string) => void; wide?: boolean }) { return <label className={`field ${wide ? 'wide' : ''}`}><span>{label}</span><input value={value} onChange={(event) => onChange(event.target.value)} /></label> }
