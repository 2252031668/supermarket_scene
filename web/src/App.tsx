import { useEffect, useMemo, useState } from 'react'
import { Box, Boxes, ChevronDown, ChevronRight, Eye, Images, Layers3, LoaderCircle, Map, PackageCheck, PackageMinus, PackagePlus, Pencil, Plus, RefreshCw, Save, ScanLine, Search, Table2, Trash2, Trophy, Upload, X } from 'lucide-react'
import { WarehouseScene } from './WarehouseScene'
import { ManualImportPage } from './ManualImportPage'
import { VisionInspectionPage } from './VisionInspectionPage'
import { RgbdStockoutPage } from './RgbdStockoutPage'
import { SkuQueryPage } from './SkuQueryPage'
import { ImageStitchPage } from './ImageStitchPage'
import { skuColor } from './skuColors'
import type { DeliveryTable, Selection, Shelf, Slot, WarehouseState } from './types'

const blankState: WarehouseState = {
  stats: { shelf_types: 0, shelf_groups: 0, delivery_tables: 0, sku_catalog: 0, total_positions: 0, actual_items: 0, shortages: 0, misplacements: 0 },
  shelf_types: [], shelves: [], shelf_images: [], delivery_tables: [], delivery_table_spec: { length: 1.2, width: 0.8, height: 0.75, top_thickness: 0.03 }, skus: [], slots: [],
}

type SlotDraft = Pick<Slot, 'slot_id' | 'shelf_id' | 'face' | 'level' | 'y_cm' | 'expected_sku' | 'actual_sku' | 'width_cm' | 'height_cm' | 'image_dir'>
type SkuDraft = Pick<WarehouseState['skus'][number], 'sku' | 'category' | 'mesh_file' | 'tex_file' | 'owlv2_prompt'>
type CleanupScope = 'level' | 'face' | 'all' | 'shelf'
type SlotWorldPosition = Slot & { frame: string }

function statusOf(slot: Pick<Slot, 'expected_sku' | 'actual_sku'>) {
  if (slot.actual_sku === null) return '缺货'
  return slot.actual_sku === slot.expected_sku ? '正常' : '摆放错误'
}

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
  const [shelfTypeDraft, setShelfTypeDraft] = useState<WarehouseState['shelf_types'][number] | null>(null)
  const [deliveryTableDraft, setDeliveryTableDraft] = useState<DeliveryTable | null>(null)
  const [skuDraft, setSkuDraft] = useState<SkuDraft | null>(null)
  const [cleanupShelfId, setCleanupShelfId] = useState<number | null>(null)
  const [slotWorldPosition, setSlotWorldPosition] = useState<SlotWorldPosition | null>(null)
  const [manualImport, setManualImport] = useState(false)
  const [visionInspection, setVisionInspection] = useState(false)
  const [rgbdStockout, setRgbdStockout] = useState(false)
  const [skuQuery, setSkuQuery] = useState(false)
  const [imageStitch, setImageStitch] = useState(false)
  const [manualImportImage, setManualImportImage] = useState<string | null>(null)

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
          const slot = next.slots.find((item) => item.slot_id === current.slot.slot_id)
          return slot ? { kind: 'slot', slot } : { kind: 'none' }
        }
        if (current.kind === 'shelf') {
          return next.shelves.some((shelf) => shelf.id === current.shelfId) ? current : { kind: 'none' }
        }
        if (current.kind === 'delivery-table') {
          return next.delivery_tables.some((table) => table.id === current.tableId) ? current : { kind: 'none' }
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
    if (selection.kind === 'delivery-table') setDeliveryTableDraft(state.delivery_tables.find((table) => table.id === selection.tableId) ?? null)
    else setDeliveryTableDraft(null)
    if (selection.kind === 'delivery-table') setSlotDraft(null)
    if (selection.kind === 'slot') {
      const { slot_id, shelf_id, face, level, y_cm, expected_sku, actual_sku, width_cm, height_cm, image_dir } = selection.slot
      setSlotDraft({ slot_id, shelf_id, face, level, y_cm, expected_sku, actual_sku, width_cm, height_cm, image_dir })
    }
  }, [selection, state.shelves, state.delivery_tables])
  useEffect(() => {
    if (selection.kind !== 'slot') {
      setSlotWorldPosition(null)
      return
    }
    let active = true
    void request<{ slot: SlotWorldPosition }>(`/api/slots/${encodeURIComponent(selection.slot.slot_id)}/world-position`)
      .then((result) => { if (active) setSlotWorldPosition(result.slot) })
      .catch(() => { if (active) setSlotWorldPosition(null) })
    return () => { active = false }
  }, [selection])

  const commit = async (path: string, method: string, payload?: unknown) => {
    setSaving(true)
    setNotice('')
    try {
      const result = await request<{ state: WarehouseState; id?: number; removed?: number; slot?: Slot }>(path, {
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
    setSlotDraft({ slot_id: '', shelf_id: selectedShelf.id, face: 0, level: 0, y_cm: 20, expected_sku: state.skus[0].sku, actual_sku: state.skus[0].sku, width_cm: null, height_cm: 16, image_dir: '' })
  }

  const saveSlot = async () => {
    if (!slotDraft) return
    const existing = selection.kind === 'slot'
    const path = existing ? `/api/slots/${encodeURIComponent(slotDraft.slot_id)}` : '/api/slots'
    const payload = existing
      ? { expected_sku: slotDraft.expected_sku, actual_sku: slotDraft.actual_sku, width_cm: slotDraft.width_cm, height_cm: slotDraft.height_cm, image_dir: slotDraft.image_dir }
      : { shelf_id: slotDraft.shelf_id, face: slotDraft.face, level: slotDraft.level, y_cm: slotDraft.y_cm, expected_sku: slotDraft.expected_sku, actual_sku: slotDraft.actual_sku, width_cm: slotDraft.width_cm, height_cm: slotDraft.height_cm, image_dir: slotDraft.image_dir }
    const result = await commit(path, existing ? 'PUT' : 'POST', payload)
    if (result) {
      const match = result.slot ?? result.state.slots.find((slot) => slot.shelf_id === slotDraft.shelf_id && slot.face === slotDraft.face && slot.level === slotDraft.level && slot.y_cm === slotDraft.y_cm)
      setSelection(match ? { kind: 'slot', slot: match } : { kind: 'shelf', shelfId: slotDraft.shelf_id })
    }
  }

  const deleteSlot = async () => {
    if (selection.kind !== 'slot' || !window.confirm('删除这个商品货位？')) return
    const result = await commit(`/api/slots/${encodeURIComponent(selection.slot.slot_id)}`, 'DELETE')
    if (result) setSelection({ kind: 'shelf', shelfId: selection.slot.shelf_id })
  }

  const runSlotAction = async (action: 'take' | 'restock') => {
    if (selection.kind !== 'slot') return
    const result = await commit(`/api/slots/${encodeURIComponent(selection.slot.slot_id)}/${action}`, 'POST', {})
    if (result?.slot) setSelection({ kind: 'slot', slot: result.slot })
  }

  const saveShelf = async () => {
    if (!shelfDraft) return
    const result = await commit(`/api/shelves/${shelfDraft.id}`, 'PUT', shelfDraft)
    if (result) setSelection({ kind: 'shelf', shelfId: shelfDraft.id })
  }

  const saveShelfType = async () => {
    if (!shelfTypeDraft) return
    const { id, name, ...parameters } = shelfTypeDraft
    const result = await commit(`/api/shelf-types/${id}`, 'PUT', parameters)
    if (result) {
      setShelfTypeDraft(null)
      setNotice(`已更新货架类型“${name}”；关联货架已按新尺寸刷新`)
    }
  }

  const saveDeliveryTable = async () => {
    if (!deliveryTableDraft) return
    const result = await commit(`/api/delivery-tables/${deliveryTableDraft.id}`, 'PUT', deliveryTableDraft)
    if (result) setSelection({ kind: 'delivery-table', tableId: deliveryTableDraft.id })
  }

  const deleteDeliveryTable = async () => {
    if (!deliveryTableDraft || !window.confirm(`删除“${deliveryTableDraft.name}”？该操作不会影响货架库存。`)) return
    const result = await commit(`/api/delivery-tables/${deliveryTableDraft.id}`, 'DELETE')
    if (result) {
      setSelection({ kind: 'none' })
      setNotice(`已删除 ${deliveryTableDraft.name}`)
    }
  }

  const beginNewSku = () => {
    setSkuDraft({ sku: '', category: '', mesh_file: '', tex_file: '', owlv2_prompt: '' })
  }

  const beginEditSku = (sku: string) => {
    const current = state.skus.find((item) => item.sku === sku)
    if (current) setSkuDraft({ ...current })
  }

  const saveSku = async () => {
    if (!skuDraft) return
    const sku = skuDraft.sku.trim()
    if (!sku) {
      setNotice('请输入 SKU 名称')
      return
    }
    const existing = state.skus.some((item) => item.sku === sku)
    const result = await commit(existing ? `/api/skus/${encodeURIComponent(sku)}` : '/api/skus', existing ? 'PUT' : 'POST', { ...skuDraft, sku })
    if (result) {
      setSkuDraft(null)
      setNotice(existing ? `已更新 SKU: ${sku}` : `已新增 SKU: ${sku}`)
    }
  }

  const deleteFromShelf = async (shelf: Shelf, scope: CleanupScope, face: number, level: number, itemCount: number) => {
    const description = scope === 'level'
      ? `${face === 0 ? '-X 侧' : '+X 侧'}第 ${level} 层`
      : scope === 'face'
        ? `${face === 0 ? '-X 侧' : '+X 侧'}整面`
        : scope === 'all' ? '全部货位' : '整个货架及其货位'
    if (!window.confirm(`确认删除“${shelf.name}”的${description}？将删除 ${itemCount} 个固定货位。`)) return
    const result = scope === 'shelf'
      ? await commit(`/api/shelves/${shelf.id}`, 'DELETE')
      : await commit(`/api/shelves/${shelf.id}/inventory`, 'DELETE', { scope, face, level })
    if (result) {
      setCleanupShelfId(null)
      setNotice(`已删除 ${result.removed ?? itemCount} 个固定货位`)
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

  const addDeliveryTable = async () => {
    const offset = state.delivery_tables.length * 0.16
    const nextTableNumber = Math.max(0, ...state.delivery_tables.map((table) => table.id)) + 1
    const result = await commit('/api/delivery-tables', 'POST', {
      name: `${nextTableNumber}号交付桌`,
      world_x: -0.6 + offset,
      world_y: -0.4 - offset,
      yaw: 0,
    })
    if (result?.id) setSelection({ kind: 'delivery-table', tableId: result.id })
  }

  const deleteSku = async (sku: string, itemCount: number) => {
    const description = itemCount ? `该 SKU 仍被 ${itemCount} 个固定货位引用，服务器将拒绝删除。` : '该 SKU 当前没有货位引用。'
    if (!window.confirm(`删除 SKU “${sku}”？\n${description}`)) return
    setSaving(true)
    try {
      const result = await request<{ removed: string; state: WarehouseState }>(`/api/skus/${encodeURIComponent(sku)}`, { method: 'DELETE' })
      setState(result.state)
      setSelection({ kind: 'none' })
      setNotice(`已删除 ${result.removed}`)
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '删除 SKU 失败')
    } finally {
      setSaving(false)
    }
  }

  if (manualImport) return <ManualImportPage state={state} initialImageUrl={manualImportImage ?? undefined} onBack={() => { setManualImport(false); setManualImportImage(null) }} onImported={(next) => setState(next)} />
  if (visionInspection) return <VisionInspectionPage onBack={() => setVisionInspection(false)} onApplied={(next) => setState(next)} />
  if (rgbdStockout) return <RgbdStockoutPage onBack={() => setRgbdStockout(false)} />
  if (skuQuery) return <SkuQueryPage state={state} onBack={() => setSkuQuery(false)} />
  if (imageStitch) return <ImageStitchPage onBack={() => setImageStitch(false)} onUseForImport={(url) => { setManualImportImage(url); setImageStitch(false); setManualImport(true) }} />

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
          <button className="manual-entry-button" title="上传局部照片并识别异常货位" onClick={() => setVisionInspection(true)}><ScanLine size={16} /><span>巡检识别</span></button>
          <button className="manual-entry-button" title="测试样本中的 RGB-D 后排缺货识别" onClick={() => setRgbdStockout(true)}><Trophy size={16} /><span>比赛巡检</span></button>
          <button className="manual-entry-button" title="用 SKU 或固定位置 ID 在局部照片中查询商品" onClick={() => setSkuQuery(true)}><Search size={16} /><span>货物查询</span></button>
          <button className="manual-entry-button" title="拼接多张同一货架面的局部照片" onClick={() => setImageStitch(true)}><Images size={16} /><span>图片拼接</span></button>
        </div>
      </header>

      <section className="workspace">
        <section className="visual-panel" aria-label="仓库三维视图">
          <WarehouseScene shelves={state.shelves} deliveryTables={state.delivery_tables} deliveryTableSpec={state.delivery_table_spec} shelfTypes={state.shelf_types} slots={state.slots} skus={state.skus} selection={selection} camera={camera} onSelect={(next) => { setSelection(next); if (next.kind === 'none') setSlotDraft(null) }} />
          <div className="scene-caption"><Layers3 size={15} />点击货架、交付桌或商品以编辑</div>
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
            <SkuEditor draft={skuDraft} existing={state.skus.some((item) => item.sku === skuDraft.sku)} onChange={setSkuDraft} onSave={() => void saveSku()} onClose={() => setSkuDraft(null)} />
          ) : shelfTypeDraft ? (
            <ShelfTypeEditor draft={shelfTypeDraft} assignedShelfCount={state.shelves.filter((shelf) => shelf.shelf_type_id === shelfTypeDraft.id).length} onChange={setShelfTypeDraft} onSave={() => void saveShelfType()} onClose={() => setShelfTypeDraft(null)} />
          ) : selection.kind === 'slot' && slotDraft ? (
            <SlotEditor draft={slotDraft} skus={state.skus} worldPosition={slotWorldPosition} onChange={setSlotDraft} onSave={() => void saveSlot()} onTake={() => void runSlotAction('take')} onRestock={() => void runSlotAction('restock')} onDelete={() => void deleteSlot()} onClose={() => setSelection({ kind: 'shelf', shelfId: slotDraft.shelf_id })} />
          ) : selection.kind === 'shelf' && shelfDraft ? (
            <ShelfEditor draft={shelfDraft} types={state.shelf_types} shelfImages={state.shelf_images} itemCount={state.slots.filter((slot) => slot.shelf_id === shelfDraft.id).length} onChange={setShelfDraft} onSave={() => void saveShelf()} onEditType={(typeId) => setShelfTypeDraft(state.shelf_types.find((type) => type.id === typeId) ?? null)} onCleanup={() => setCleanupShelfId(shelfDraft.id)} onAddSlot={beginNewSlot} />
          ) : selection.kind === 'delivery-table' && deliveryTableDraft ? (
            <DeliveryTableEditor draft={deliveryTableDraft} onChange={setDeliveryTableDraft} onSave={() => void saveDeliveryTable()} onDelete={() => void deleteDeliveryTable()} onClose={() => setSelection({ kind: 'none' })} />
          ) : slotDraft ? (
            <SlotEditor draft={slotDraft} skus={state.skus} onChange={setSlotDraft} onSave={() => void saveSlot()} onTake={() => {}} onRestock={() => {}} onDelete={() => setSlotDraft(null)} onClose={() => setSlotDraft(null)} isNew />
          ) : (
            <Overview state={state} onSelectSlot={(slot) => setSelection({ kind: 'slot', slot })} onSelectShelf={(shelfId) => setSelection({ kind: 'shelf', shelfId })} onAddShelf={() => void addShelf()} onSelectDeliveryTable={(tableId) => setSelection({ kind: 'delivery-table', tableId })} onAddDeliveryTable={() => void addDeliveryTable()} onAddSku={beginNewSku} onEditSku={beginEditSku} onDeleteSku={(sku, itemCount) => void deleteSku(sku, itemCount)} />
          )}
        </aside>
      </section>
    </main>
  )
}

function PanelHeader({ icon, title, eyebrow, action }: { icon: React.ReactNode; title: string; eyebrow: string; action?: React.ReactNode }) {
  return <div className="panel-header"><div className="panel-title"><span className="panel-icon">{icon}</span><div><p>{eyebrow}</p><h1>{title}</h1></div></div>{action}</div>
}

function Overview({ state, onSelectSlot, onSelectShelf, onAddShelf, onSelectDeliveryTable, onAddDeliveryTable, onAddSku, onEditSku, onDeleteSku }: { state: WarehouseState; onSelectSlot: (slot: Slot) => void; onSelectShelf: (id: number) => void; onAddShelf: () => void; onSelectDeliveryTable: (id: number) => void; onAddDeliveryTable: () => void; onAddSku: () => void; onEditSku: (sku: string) => void; onDeleteSku: (sku: string, itemCount: number) => void }) {
  const [shelfListOpen, setShelfListOpen] = useState(true)
  const [deliveryTableListOpen, setDeliveryTableListOpen] = useState(true)
  const [skuLegendOpen, setSkuLegendOpen] = useState(true)
  const [statusFilter, setStatusFilter] = useState<'all' | '缺货' | '摆放错误'>('all')
  const visibleSlots = statusFilter === 'all' ? state.slots : state.slots.filter((slot) => slot.status === statusFilter)

  return <div className="overview-panel">
    <PanelHeader icon={<Boxes size={19} />} title="仓库总览" eyebrow="SHELF INVENTORY" />
    <div className="metrics status-metrics">
      <Metric label="固定货位" value={state.stats.total_positions} />
      <Metric label="实际商品" value={state.stats.actual_items} />
      <Metric label="缺货" value={state.stats.shortages} />
      <Metric label="错放" value={state.stats.misplacements} />
    </div>
    <div className="status-filter" role="group" aria-label="货位状态筛选">
      <button className={statusFilter === 'all' ? 'active' : ''} onClick={() => setStatusFilter('all')}>全部</button>
      <button className={statusFilter === '缺货' ? 'active' : ''} onClick={() => setStatusFilter('缺货')}>缺货 {state.stats.shortages}</button>
      <button className={statusFilter === '摆放错误' ? 'active' : ''} onClick={() => setStatusFilter('摆放错误')}>错放 {state.stats.misplacements}</button>
    </div>
    <div className="status-list">{visibleSlots.map((slot) => <button key={slot.slot_id} className="status-row" onClick={() => onSelectSlot(slot)}><span className={`status-badge status-${slot.status}`}>{slot.status}</span><span><strong>{slot.slot_id}</strong><small>应摆 {slot.expected_sku} · 实际 {slot.actual_sku ?? '空'}</small></span><ChevronRight size={15} /></button>)}</div>
    <div className="section-heading"><button className="section-toggle" type="button" aria-expanded={shelfListOpen} onClick={() => setShelfListOpen((open) => !open)}><span>货架组</span><span>{state.shelves.length}<ChevronDown size={15} className={shelfListOpen ? '' : 'collapsed'} /></span></button><button className="section-add-button" title="新建货架" onClick={onAddShelf}><Plus size={15} /></button></div>
    {shelfListOpen && <div className="shelf-list">
      {state.shelves.map((shelf) => <button className="shelf-row" key={shelf.id} onClick={() => onSelectShelf(shelf.id)}><span className="shelf-id">{String(shelf.id).padStart(2, '0')}</span><span><strong>{shelf.name}</strong><small>X {shelf.world_x.toFixed(2)} / Y {shelf.world_y.toFixed(2)}</small></span><ChevronRight size={17} /></button>)}
    </div>}
    <div className="section-heading"><button className="section-toggle" type="button" aria-expanded={deliveryTableListOpen} onClick={() => setDeliveryTableListOpen((open) => !open)}><span>交付桌</span><span>{state.delivery_tables.length}<ChevronDown size={15} className={deliveryTableListOpen ? '' : 'collapsed'} /></span></button><button className="section-add-button" title="新建交付桌" onClick={onAddDeliveryTable}><Plus size={15} /></button></div>
    {deliveryTableListOpen && <div className="shelf-list">
      {state.delivery_tables.map((table) => <button className="shelf-row" key={table.id} onClick={() => onSelectDeliveryTable(table.id)}><span className="shelf-id delivery-table-id">{String(table.id).padStart(2, '0')}</span><span><strong>{table.name}</strong><small>X {table.world_x.toFixed(2)} / Y {table.world_y.toFixed(2)}</small></span><ChevronRight size={17} /></button>)}
    </div>}
    <div className="sku-legend-header"><button className="sku-legend-toggle" type="button" aria-expanded={skuLegendOpen} onClick={() => setSkuLegendOpen((open) => !open)}>
      <span>SKU 图例</span><span className="sku-legend-count">{state.skus.length}<ChevronDown size={15} className={skuLegendOpen ? '' : 'collapsed'} /></span>
    </button><button className="icon-button sku-add-button" title="新增 SKU" onClick={onAddSku}><Plus size={15} /></button></div>
    {skuLegendOpen && <div className="sku-summary" aria-label="SKU 图例列表">{state.skus.map((sku, index) => {
      const itemCount = state.slots.filter((slot) => slot.expected_sku === sku.sku || slot.actual_sku === sku.sku).length
      return <div className="sku-row" key={sku.sku}><span className="sku-swatch" style={{ backgroundColor: skuColor(index) }} /><strong>{sku.sku}</strong><small>{itemCount} 个引用</small><button className="icon-button" title={`编辑 ${sku.sku}`} onClick={() => onEditSku(sku.sku)}><Pencil size={14} /></button>{sku.sku !== 'unknown' && <button className="icon-button sku-delete-button" title={`删除 ${sku.sku}`} onClick={() => onDeleteSku(sku.sku, itemCount)}><Trash2 size={14} /></button>}</div>
    })}</div>}
  </div>
}

function Metric({ label, value }: { label: string; value: number }) { return <div className="metric"><strong>{value}</strong><span>{label}</span></div> }

function ShelfEditor({ draft, types, shelfImages, itemCount, onChange, onSave, onEditType, onCleanup, onAddSlot }: { draft: Shelf; types: WarehouseState['shelf_types']; shelfImages: WarehouseState['shelf_images']; itemCount: number; onChange: (value: Shelf) => void; onSave: () => void; onEditType: (typeId: number) => void; onCleanup: () => void; onAddSlot: () => void }) {
  const set = <K extends keyof Shelf>(key: K, value: Shelf[K]) => onChange({ ...draft, [key]: value })
  const hasImage = (face: 0 | 1) => shelfImages.some((image) => image.shelf_id === draft.id && image.face === face)
  return <>
    <PanelHeader icon={<Layers3 size={19} />} title={draft.name || `货架 #${draft.id}`} eyebrow={`SHELF GROUP · ${itemCount} SLOTS`} action={<button className="danger-icon" title="删除或清理货位" onClick={onCleanup}><Trash2 size={17} /></button>} />
    <div className="field-grid"><Field label="名称" value={draft.name} onChange={(value) => set('name', value)} wide /><Field label="世界 X (m)" value={draft.world_x} onChange={(value) => set('world_x', number(value))} /><Field label="世界 Y (m)" value={draft.world_y} onChange={(value) => set('world_y', number(value))} /><Field label="朝向 (rad)" value={draft.yaw} onChange={(value) => set('yaw', number(value))} /><label className="field"><span>货架类型</span><div className="type-select-row"><select value={draft.shelf_type_id ?? ''} onChange={(event) => set('shelf_type_id', event.target.value ? Number(event.target.value) : null)}>{types.map((type) => <option key={type.id} value={type.id}>{type.name}</option>)}</select><button className="icon-button" title="编辑当前货架类型参数" disabled={!draft.shelf_type_id} onClick={() => draft.shelf_type_id && onEditType(draft.shelf_type_id)}><Pencil size={15} /></button></div></label></div>
    <section className="shelf-image-section" aria-label="货架面照片">
      <div className="section-heading"><span>货架面照片</span><span>原图 0.png</span></div>
      <div className="shelf-image-grid">{([0, 1] as const).map((face) => hasImage(face) ? <figure className="shelf-image-card" key={face}><img src={`/api/shelf-images/${draft.id}/${face}/0.png`} alt={`${draft.name} ${face === 0 ? '-X' : '+X'} 面照片`} /><figcaption>{face === 0 ? '-X 侧' : '+X 侧'}</figcaption></figure> : <div className="shelf-image-empty" key={face}><strong>{face === 0 ? '-X 侧' : '+X 侧'}</strong><span>暂无原图</span></div>)}</div>
    </section>
    <div className="panel-actions"><button className="primary-button" onClick={onSave}><Save size={16} />保存货架</button><button className="secondary-button" onClick={onAddSlot}><PackagePlus size={16} />添加货位</button></div>
  </>
}

function ShelfTypeEditor({ draft, assignedShelfCount, onChange, onSave, onClose }: { draft: WarehouseState['shelf_types'][number]; assignedShelfCount: number; onChange: (value: WarehouseState['shelf_types'][number]) => void; onSave: () => void; onClose: () => void }) {
  const set = <K extends keyof WarehouseState['shelf_types'][number]>(key: K, value: WarehouseState['shelf_types'][number][K]) => onChange({ ...draft, [key]: value })
  return <>
    <PanelHeader icon={<Layers3 size={19} />} title={draft.name} eyebrow="SHELF TYPE · SHARED PARAMETERS" action={<button className="icon-button" title="返回货架" onClick={onClose}><X size={17} /></button>} />
    <div className="shared-type-warning"><strong>共享类型参数</strong><span>当前有 {assignedShelfCount} 个货架使用该类型。保存后它们的尺寸、层板位置和商品世界坐标会立即按新参数计算。</span></div>
    <div className="field-grid type-parameter-grid">
      <TypeParameterField label="货架长度 (m)" value={draft.shelf_length} step="0.01" description="局部 +Y 方向的总长度，决定商品 y_cm 的有效范围。" onChange={(value) => set('shelf_length', number(value))} />
      <TypeParameterField label="货架宽度 (m)" value={draft.shelf_width} step="0.01" description="局部 +X 方向的总宽度，包含两侧货架面。" onChange={(value) => set('shelf_width', number(value))} />
      <TypeParameterField label="货架总高度 (m)" value={draft.shelf_height} step="0.01" description="货架从地面到最高处的设计总高度。" onChange={(value) => set('shelf_height', number(value))} />
      <TypeParameterField label="层数" value={draft.num_levels} step="1" description="层号从下到上编号为 0 至层数减 1。" onChange={(value) => set('num_levels', Math.max(1, Math.round(number(value))))} />
      <TypeParameterField label="底层离地高度 (m)" value={draft.bottom_clearance} step="0.01" description="地面到第 0 层层板中心的基础高度。" onChange={(value) => set('bottom_clearance', number(value))} />
      <TypeParameterField label="层间距 (m)" value={draft.level_spacing} step="0.01" description="相邻两层层板中心在局部 +Z 方向的间距。" onChange={(value) => set('level_spacing', number(value))} />
      <TypeParameterField label="层板厚度 (m)" value={draft.panel_thick} step="0.001" description="每块水平层板的厚度；商品中心高度从层板上表面计算。" onChange={(value) => set('panel_thick', number(value))} />
      <TypeParameterField label="背板厚度 (m)" value={draft.back_thick} step="0.001" description="货架中部竖直背板在局部 +X 方向的厚度。" onChange={(value) => set('back_thick', number(value))} />
      <TypeParameterField label="普通层层板深度 (m)" value={draft.shelf_depth_normal} step="0.01" description="第 1 层及以上每一面的局部 X 向层板深度。" onChange={(value) => set('shelf_depth_normal', number(value))} />
      <TypeParameterField label="底层层板深度 (m)" value={draft.shelf_depth_bottom} step="0.01" description="第 0 层每一面的局部 X 向层板深度。" onChange={(value) => set('shelf_depth_bottom', number(value))} />
    </div>
    <p className="helper-text">减少层数或长度时，若已有商品会超出新的层号或长度范围，服务器将拒绝保存以保护库存数据。</p>
    <div className="panel-actions"><button className="primary-button" onClick={onSave}><Save size={16} />保存类型参数</button></div>
  </>
}

function DeliveryTableEditor({ draft, onChange, onSave, onDelete, onClose }: { draft: DeliveryTable; onChange: (value: DeliveryTable) => void; onSave: () => void; onDelete: () => void; onClose: () => void }) {
  const set = <K extends keyof DeliveryTable>(key: K, value: DeliveryTable[K]) => onChange({ ...draft, [key]: value })
  return <>
    <PanelHeader icon={<Table2 size={19} />} title={draft.name || `交付桌 #${draft.id}`} eyebrow="DELIVERY TABLE · NO INVENTORY" action={<button className="icon-button" title="返回总览" onClick={onClose}><X size={17} /></button>} />
    <div className="field-grid"><Field label="名称" value={draft.name} onChange={(value) => set('name', value)} wide /><Field label="世界 X (m)" value={draft.world_x} onChange={(value) => set('world_x', number(value))} /><Field label="世界 Y (m)" value={draft.world_y} onChange={(value) => set('world_y', number(value))} /><Field label="朝向 (rad)" value={draft.yaw} onChange={(value) => set('yaw', number(value))} /></div>
    <p className="helper-text">世界 X/Y 是交付桌局部原点。局部 +X 沿桌长，+Y 沿桌宽，+Z 向上；该实体目前不存储商品。</p>
    <div className="panel-actions"><button className="primary-button" onClick={onSave}><Save size={16} />保存交付桌</button><button className="danger-icon" title="删除交付桌" onClick={onDelete}><Trash2 size={17} /></button></div>
  </>
}

function SkuEditor({ draft, existing, onChange, onSave, onClose }: { draft: SkuDraft; existing: boolean; onChange: (value: SkuDraft) => void; onSave: () => void; onClose: () => void }) {
  const set = <K extends keyof SkuDraft>(key: K, value: SkuDraft[K]) => onChange({ ...draft, [key]: value })
  return <>
    <PanelHeader icon={<Box size={19} />} title={existing ? '编辑 SKU' : '新增 SKU'} eyebrow="SKU CATALOG" action={<button className="icon-button" title="返回总览" onClick={onClose}><X size={17} /></button>} />
    <div className="field-grid"><Field label="SKU 名称" value={draft.sku} onChange={(value) => set('sku', value)} readOnly={existing} wide /><Field label="分类（可选）" value={draft.category} onChange={(value) => set('category', value)} /><Field label="网格文件（可选）" value={draft.mesh_file} onChange={(value) => set('mesh_file', value)} wide /><Field label="纹理文件（可选）" value={draft.tex_file} onChange={(value) => set('tex_file', value)} wide /><Field label="OWLv2 英文提示词" value={draft.owlv2_prompt} onChange={(value) => set('owlv2_prompt', value)} wide /></div>
    <div className="panel-actions"><button className="primary-button" onClick={onSave}><Save size={16} />{existing ? '保存 SKU' : '创建 SKU'}</button></div>
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
  const actionLabel = scope === 'shelf' ? '删除货架' : '删除固定货位'

  return <>
    <PanelHeader icon={<Trash2 size={19} />} title="删除与清理" eyebrow={shelf.name} action={<button className="icon-button" title="返回货架" onClick={onClose}><X size={17} /></button>} />
    <div className="cleanup-scope" role="group" aria-label="删除范围">
      <button className={scope === 'level' ? 'active' : ''} onClick={() => setScope('level')}>指定层</button><button className={scope === 'face' ? 'active' : ''} onClick={() => setScope('face')}>整面</button><button className={scope === 'all' ? 'active' : ''} onClick={() => setScope('all')}>全部货位</button><button className={scope === 'shelf' ? 'active' : ''} onClick={() => setScope('shelf')}>整个货架</button>
    </div>
    {(scope === 'level' || scope === 'face') && <div className="field-grid cleanup-fields"><label className="field"><span>货架面</span><select value={face} onChange={(event) => setFace(Number(event.target.value))}><option value={0}>-X 侧</option><option value={1}>+X 侧</option></select></label>{scope === 'level' && <label className="field"><span>层号</span><select value={level} onChange={(event) => setLevel(Number(event.target.value))}>{Array.from({ length: levels }, (_, index) => <option key={index} value={index}>第 {index} 层</option>)}</select></label>}</div>}
    <div className="cleanup-preview"><span>{targetName}</span><strong>将删除 {targetSlots.length} 个固定货位</strong></div>
    <button className="danger-button" onClick={() => onConfirm(scope, face, level, targetSlots.length)}><Trash2 size={16} />{actionLabel}</button>
  </>
}

function SlotEditor({ draft, skus, worldPosition, onChange, onSave, onTake, onRestock, onDelete, onClose, isNew = false }: { draft: SlotDraft; skus: WarehouseState['skus']; worldPosition?: SlotWorldPosition | null; onChange: (value: SlotDraft) => void; onSave: () => void; onTake: () => void; onRestock: () => void; onDelete: () => void; onClose: () => void; isNew?: boolean }) {
  const set = <K extends keyof SlotDraft>(key: K, value: SlotDraft[K]) => onChange({ ...draft, [key]: value })
  const status = statusOf(draft)
  return <>
    <PanelHeader icon={<Box size={19} />} title={isNew ? '新增固定货位' : draft.expected_sku} eyebrow={isNew ? 'NEW FIXED SLOT' : `SLOT · ${draft.slot_id}`} action={<button className="icon-button" title="返回货架" onClick={onClose}><X size={17} /></button>} />
    <div className="slot-status-line"><span>当前状态</span><strong className={`status-badge status-${status}`}>{status}</strong></div>
    <div className="field-grid">
      <label className="field wide"><span>应摆 SKU</span><select value={draft.expected_sku} onChange={(event) => set('expected_sku', event.target.value)}>{skus.map((sku) => <option key={sku.sku} value={sku.sku}>{sku.sku}</option>)}</select></label>
      <label className="field wide"><span>实际 SKU</span><select value={draft.actual_sku ?? ''} onChange={(event) => set('actual_sku', event.target.value || null)}><option value="">空（缺货）</option>{skus.map((sku) => <option key={sku.sku} value={sku.sku}>{sku.sku}</option>)}</select></label>
      {isNew ? <><Field label="货架 ID" value={draft.shelf_id} onChange={(value) => set('shelf_id', number(value))} /><label className="field"><span>货架面</span><select value={draft.face} onChange={(event) => set('face', Number(event.target.value))}><option value={0}>-X 侧 (0)</option><option value={1}>+X 侧 (1)</option></select></label><Field label="层号" value={draft.level} onChange={(value) => set('level', number(value))} /><Field label="Y 中心 (cm)" value={draft.y_cm} onChange={(value) => set('y_cm', number(value))} /></> : <><ReadonlyField label="位置 ID" value={draft.slot_id} wide /><ReadonlyField label="货架 / 面" value={`${draft.shelf_id} / ${draft.face === 0 ? '-X' : '+X'}`} /><ReadonlyField label="层 / Y" value={`${draft.level} / ${draft.y_cm} cm`} /></>}
      <Field label="宽度 (cm，可选)" value={draft.width_cm ?? ''} onChange={(value) => set('width_cm', value === '' ? null : number(value))} /><Field label="高度 (cm，可选)" value={draft.height_cm ?? ''} onChange={(value) => set('height_cm', value === '' ? null : number(value))} />
    </div>
    <p className="helper-text">商品世界坐标指向几何中心；中心高度由本层层板表面加上商品高度的一半计算。</p>
    {!isNew && <div className="world-position"><span>世界坐标（{worldPosition?.frame ?? 'map'}，m）</span>{worldPosition ? <div><strong>X {worldPosition.world_x.toFixed(3)}</strong><strong>Y {worldPosition.world_y.toFixed(3)}</strong><strong>Z {worldPosition.world_z.toFixed(3)}</strong></div> : <small>正在读取坐标</small>}</div>}
    <div className="panel-actions"><button className="primary-button" onClick={onSave}><Save size={16} />{isNew ? '创建货位' : '保存状态'}</button>{!isNew && <><button className="icon-button" title="拿走实际商品并标记缺货" onClick={onTake}><PackageMinus size={17} /></button><button className="icon-button" title="按应摆 SKU 补货" onClick={onRestock}><PackageCheck size={17} /></button><button className="danger-icon" title="删除固定货位" onClick={onDelete}><Trash2 size={17} /></button></>}</div>
  </>
}

function Field({ label, value, onChange, wide = false, readOnly = false }: { label: string; value: string | number; onChange: (value: string) => void; wide?: boolean; readOnly?: boolean }) { return <label className={`field ${wide ? 'wide' : ''}`}><span>{label}</span><input value={value} readOnly={readOnly} onChange={(event) => onChange(event.target.value)} /></label> }

function ReadonlyField({ label, value, wide = false }: { label: string; value: string | number; wide?: boolean }) { return <label className={`field ${wide ? 'wide' : ''}`}><span>{label}</span><input value={value} readOnly /></label> }

function TypeParameterField({ label, value, step, description, onChange }: { label: string; value: number; step: string; description: string; onChange: (value: string) => void }) { return <label className="field type-parameter-field"><span>{label}</span><input type="number" min="0" step={step} value={value} onChange={(event) => onChange(event.target.value)} /><small>{description}</small></label> }
