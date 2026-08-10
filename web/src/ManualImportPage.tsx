import { useEffect, useMemo, useRef, useState } from 'react'
import { ArrowLeft, Check, ImagePlus, Layers3, LoaderCircle, MousePointer2, PackagePlus, PencilRuler, Plus, Sparkles, Trash2, Upload } from 'lucide-react'
import type { ShelfType, WarehouseState } from './types'

type Point = { x: number; y: number }
type LayerCalibration = { id: string; level: number; points: [Point, Point, Point, Point] }
type ProductBox = { id: string; expected_sku: string; actual_sku: string | null; x: number; y: number; width: number; height: number; crop_png: string; yOverride?: number; widthOverride?: number; heightOverride?: number }
type DerivedItem = ProductBox & { level: number; y_cm: number; width_cm: number; height_cm: number; slot_id: string; error?: string }
type ToolMode = 'calibrate' | 'draw' | 'edit'
type Handle = 'move' | 'nw' | 'ne' | 'se' | 'sw'
type BoxDrag = { id: string; handle: Handle; start: Point; original: ProductBox }
type GroundingResponse = { boxes: Array<{ label: string; x: number; y: number; width: number; height: number }>; detected: number; error?: string }

const UNKNOWN_SKU = 'unknown'

function valueFromEvent(event: React.PointerEvent<SVGSVGElement>, image: { width: number; height: number }): Point {
  const rect = event.currentTarget.getBoundingClientRect()
  return { x: (event.clientX - rect.left) * image.width / rect.width, y: (event.clientY - rect.top) * image.height / rect.height }
}

function valueFromSvg(event: React.PointerEvent<SVGGElement | SVGCircleElement>, svg: SVGSVGElement, image: { width: number; height: number }): Point {
  const rect = svg.getBoundingClientRect()
  return { x: (event.clientX - rect.left) * image.width / rect.width, y: (event.clientY - rect.top) * image.height / rect.height }
}

function solve(matrix: number[][], vector: number[]): number[] | null {
  const augmented = matrix.map((row, index) => [...row, vector[index]])
  for (let column = 0; column < augmented.length; column += 1) {
    let pivot = column
    for (let row = column + 1; row < augmented.length; row += 1) if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row
    if (Math.abs(augmented[pivot][column]) < 1e-9) return null
    ;[augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]]
    const divisor = augmented[column][column]
    for (let index = column; index <= augmented.length; index += 1) augmented[column][index] /= divisor
    for (let row = 0; row < augmented.length; row += 1) {
      if (row === column) continue
      const factor = augmented[row][column]
      for (let index = column; index <= augmented.length; index += 1) augmented[row][index] -= factor * augmented[column][index]
    }
  }
  return augmented.map((row) => row[augmented.length])
}

function homography(points: [Point, Point, Point, Point]): number[] | null {
  const destination: Point[] = [{ x: 0, y: 1 }, { x: 1, y: 1 }, { x: 1, y: 0 }, { x: 0, y: 0 }]
  const matrix: number[][] = []
  const vector: number[] = []
  points.forEach((source, index) => {
    const target = destination[index]
    matrix.push([source.x, source.y, 1, 0, 0, 0, -target.x * source.x, -target.x * source.y])
    vector.push(target.x)
    matrix.push([0, 0, 0, source.x, source.y, 1, -target.y * source.x, -target.y * source.y])
    vector.push(target.y)
  })
  return solve(matrix, vector)
}

function project(transform: number[] | null, point: Point): Point | null {
  if (!transform) return null
  const denominator = transform[6] * point.x + transform[7] * point.y + 1
  if (Math.abs(denominator) < 1e-9) return null
  return { x: (transform[0] * point.x + transform[1] * point.y + transform[2]) / denominator, y: (transform[3] * point.x + transform[4] * point.y + transform[5]) / denominator }
}

function openingHeightCm(type: ShelfType) { return Math.max(0, (type.level_spacing - type.panel_thick) * 100) }
function slotStatus(expectedSku: string, actualSku: string | null) { return actualSku === null ? '缺货' : actualSku === expectedSku ? '正常' : '摆放错误' }

function cropPng(image: HTMLImageElement, box: Pick<ProductBox, 'x' | 'y' | 'width' | 'height'>) {
  const scale = Math.min(1, 560 / Math.max(box.width, box.height))
  const canvas = document.createElement('canvas')
  canvas.width = Math.max(1, Math.round(box.width * scale))
  canvas.height = Math.max(1, Math.round(box.height * scale))
  const context = canvas.getContext('2d')
  if (!context) throw new Error('无法创建商品图片')
  context.drawImage(image, box.x, box.y, box.width, box.height, 0, 0, canvas.width, canvas.height)
  return canvas.toDataURL('image/png')
}

function fitBox(box: Pick<ProductBox, 'x' | 'y' | 'width' | 'height'>, image: { width: number; height: number }) {
  const width = Math.max(4, Math.min(box.width, image.width))
  const height = Math.max(4, Math.min(box.height, image.height))
  return { x: Math.max(0, Math.min(box.x, image.width - width)), y: Math.max(0, Math.min(box.y, image.height - height)), width, height }
}

function boxAfterDrag(drag: BoxDrag, point: Point, image: { width: number; height: number }) {
  const { original } = drag
  const right = original.x + original.width
  const bottom = original.y + original.height
  if (drag.handle === 'move') return fitBox({ ...original, x: original.x + point.x - drag.start.x, y: original.y + point.y - drag.start.y }, image)
  if (drag.handle === 'nw') return fitBox({ x: Math.min(point.x, right - 4), y: Math.min(point.y, bottom - 4), width: right - point.x, height: bottom - point.y }, image)
  if (drag.handle === 'ne') return fitBox({ x: original.x, y: Math.min(point.y, bottom - 4), width: Math.max(4, point.x - original.x), height: bottom - point.y }, image)
  if (drag.handle === 'se') return fitBox({ x: original.x, y: original.y, width: Math.max(4, point.x - original.x), height: Math.max(4, point.y - original.y) }, image)
  return fitBox({ x: Math.min(point.x, right - 4), y: original.y, width: right - point.x, height: Math.max(4, point.y - original.y) }, image)
}

export function ManualImportPage({ state, onBack, onImported, initialImageUrl }: { state: WarehouseState; onBack: () => void; onImported: (next: WarehouseState) => void; initialImageUrl?: string }) {
  const [shelfId, setShelfId] = useState(state.shelves[0]?.id ?? 0)
  const [face, setFace] = useState(0)
  const [step, setStep] = useState(1)
  const [toolMode, setToolMode] = useState<ToolMode>('calibrate')
  const [image, setImage] = useState<{ url: string; dataUrl: string; width: number; height: number } | null>(null)
  const sourceImage = useRef<HTMLImageElement | null>(null)
  const [activeLevel, setActiveLevel] = useState(0)
  const [layers, setLayers] = useState<LayerCalibration[]>([])
  const [pendingPoints, setPendingPoints] = useState<Point[]>([])
  const [sku, setSku] = useState(state.skus[0]?.sku ?? UNKNOWN_SKU)
  const [emptyPosition, setEmptyPosition] = useState(false)
  const [newSku, setNewSku] = useState('')
  const [createdSkus, setCreatedSkus] = useState<string[]>([])
  const [selectedDraftSkus, setSelectedDraftSkus] = useState<string[]>([])
  const [products, setProducts] = useState<ProductBox[]>([])
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null)
  const [dragStart, setDragStart] = useState<Point | null>(null)
  const [dragCurrent, setDragCurrent] = useState<Point | null>(null)
  const [boxDrag, setBoxDrag] = useState<BoxDrag | null>(null)
  const [notice, setNotice] = useState('')
  const [importing, setImporting] = useState(false)
  const [aiRunning, setAiRunning] = useState(false)
  const [aiStatus, setAiStatus] = useState('')
  const [skuPrompts, setSkuPrompts] = useState<Record<string, string>>({})
  const [promptRunning, setPromptRunning] = useState(false)

  const shelf = state.shelves.find((item) => item.id === shelfId)
  const shelfType = state.shelf_types.find((item) => item.id === shelf?.shelf_type_id) ?? state.shelf_types[0]
  const levels = shelfType ? Array.from({ length: shelfType.num_levels }, (_, index) => index) : []
  const allSkus = useMemo(() => Array.from(new Set([...state.skus.map((item) => item.sku), UNKNOWN_SKU, ...createdSkus])), [state.skus, createdSkus])
  const selectedProduct = products.find((product) => product.id === selectedProductId)

  const derivedItems = useMemo<DerivedItem[]>(() => products.map((product) => {
    if (!product.expected_sku.trim()) return { ...product, level: -1, y_cm: 0, width_cm: 0, height_cm: 0, slot_id: '', error: '缺少应摆 SKU' }
    if (!shelfType || layers.length === 0) return { ...product, level: -1, y_cm: 0, width_cm: 0, height_cm: 0, slot_id: '', error: '缺少层位标定' }
    const corners = [{ x: product.x, y: product.y }, { x: product.x + product.width, y: product.y }, { x: product.x + product.width, y: product.y + product.height }, { x: product.x, y: product.y + product.height }]
    const matches = layers.flatMap((candidate) => {
      const candidateTransform = homography(candidate.points)
      const mappedCorners = corners.map((corner) => project(candidateTransform, corner))
      return mappedCorners.every((corner) => corner && corner.x >= 0 && corner.x <= 1 && corner.y >= 0 && corner.y <= 1) && candidateTransform ? [{ layer: candidate, transform: candidateTransform }] : []
    })
    if (matches.length === 0) return { ...product, level: -1, y_cm: 0, width_cm: 0, height_cm: 0, slot_id: '', error: '商品框未完全位于任何已标定层位内' }
    if (matches.length > 1) return { ...product, level: -1, y_cm: 0, width_cm: 0, height_cm: 0, slot_id: '', error: '商品框同时位于多个标定层位内' }
    const { layer, transform } = matches[0]
    const center = project(transform, { x: product.x + product.width / 2, y: product.y + product.height / 2 })
    const left = project(transform, { x: product.x, y: product.y + product.height / 2 })
    const right = project(transform, { x: product.x + product.width, y: product.y + product.height / 2 })
    const top = project(transform, { x: product.x + product.width / 2, y: product.y })
    const bottom = project(transform, { x: product.x + product.width / 2, y: product.y + product.height })
    if (!center || !left || !right || !top || !bottom) return { ...product, level: layer.level, y_cm: 0, width_cm: 0, height_cm: 0, slot_id: '', error: '标定区域无效' }
    const lengthCm = shelfType.shelf_length * 100
    const mappedY = (face === 1 ? center.x : 1 - center.x) * lengthCm
    const y_cm = product.yOverride ?? Math.round(mappedY)
    const width_cm = product.widthOverride ?? Math.round(Math.abs(right.x - left.x) * lengthCm * 10) / 10
    const height_cm = product.heightOverride ?? Math.round(Math.abs(top.y - bottom.y) * openingHeightCm(shelfType) * 10) / 10
    return { ...product, level: layer.level, y_cm, width_cm, height_cm, slot_id: `${shelfId}-${face}-${layer.level}-${y_cm}` }
  }), [products, layers, shelfType, shelfId, face])

  const duplicateIds = useMemo(() => new Set(derivedItems.filter((item, index, items) => item.slot_id && items.findIndex((other) => other.slot_id === item.slot_id) !== index).map((item) => item.slot_id)), [derivedItems])
  const promptRows = useMemo(() => Array.from(new Set(products.flatMap((item) => [item.expected_sku, item.actual_sku].filter((name): name is string => Boolean(name && name !== UNKNOWN_SKU))))).sort().map((sku) => ({
    sku,
    samples: products.filter((item) => item.expected_sku === sku && item.actual_sku === sku).sort((left, right) => right.width * right.height - left.width * left.height).slice(0, 3).map((item) => item.crop_png),
  })), [products])
  const promptValue = (sku: string) => skuPrompts[sku] ?? state.skus.find((item) => item.sku === sku)?.owlv2_prompt ?? ''

  const updateProduct = (id: string, patch: Partial<ProductBox>) => setProducts((current) => current.map((item) => item.id === id ? { ...item, ...patch } : item))
  const updateProductBox = (id: string, box: Pick<ProductBox, 'x' | 'y' | 'width' | 'height'>) => {
    if (!sourceImage.current) return
    updateProduct(id, { ...box, crop_png: cropPng(sourceImage.current, box) })
  }

  const loadPhoto = (dataUrl: string) => {
    const loaded = new Image()
    loaded.onload = () => {
      sourceImage.current = loaded
      setImage({ url: dataUrl, dataUrl, width: loaded.naturalWidth, height: loaded.naturalHeight })
      setLayers([]); setProducts([]); setPendingPoints([]); setSelectedProductId(null); setStep(2); setToolMode('calibrate')
      setNotice('请先为每个需要录入的层位按左上、右上、右下、左下顺序点击四个角。')
    }
    loaded.src = dataUrl
  }

  const uploadPhoto = (file?: File) => {
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => loadPhoto(String(reader.result))
    reader.readAsDataURL(file)
  }

  useEffect(() => {
    if (!initialImageUrl) return
    void fetch(initialImageUrl).then((response) => response.blob()).then((blob) => {
      const reader = new FileReader()
      reader.onload = () => loadPhoto(String(reader.result))
      reader.readAsDataURL(blob)
    }).catch(() => setNotice('无法加载拼接图片'))
  }, [initialImageUrl])

  const addLayerPoint = (point: Point) => {
    if (pendingPoints.length >= 4) return
    const next = [...pendingPoints, point]
    if (next.length < 4) { setPendingPoints(next); return }
    const calibration: LayerCalibration = { id: `layer-${activeLevel}`, level: activeLevel, points: next as LayerCalibration['points'] }
    setLayers((current) => [...current.filter((item) => item.level !== activeLevel), calibration])
    setPendingPoints([])
    setNotice(`第 ${activeLevel} 层标定完成。可选择其他层号继续标定。`)
  }

  const addSku = () => {
    const name = newSku.trim()
    if (!name) return
    if (!allSkus.includes(name)) setCreatedSkus((current) => [...current, name])
    setSku(name); setNewSku('')
  }

  const addProduct = (start: Point, end: Point) => {
    if (!sourceImage.current || layers.length === 0) { setNotice('请先完成至少一个层位四点标定，再框选商品。'); return }
    if (!sku) { setNotice('请先选择或创建 SKU。'); return }
    const box = fitBox({ x: Math.min(start.x, end.x), y: Math.min(start.y, end.y), width: Math.abs(start.x - end.x), height: Math.abs(start.y - end.y) }, image!)
    if (box.width < 4 || box.height < 4) return
    const product = { id: crypto.randomUUID(), expected_sku: sku, actual_sku: emptyPosition ? null : sku, ...box, crop_png: cropPng(sourceImage.current, box) }
    setProducts((current) => [...current, product]); setSelectedProductId(product.id)
    setNotice('商品已加入审核表。')
  }

  const deleteDraftSkus = () => {
    if (!selectedDraftSkus.length) return
    const selected = new Set(selectedDraftSkus)
    setProducts((current) => current.map((product) => ({
      ...product,
      expected_sku: selected.has(product.expected_sku) ? UNKNOWN_SKU : product.expected_sku,
      actual_sku: product.actual_sku && selected.has(product.actual_sku) ? UNKNOWN_SKU : product.actual_sku,
    })))
    setCreatedSkus((current) => current.filter((name) => !selected.has(name)))
    if (selected.has(sku)) setSku(UNKNOWN_SKU)
    setSelectedDraftSkus([])
    setNotice(`已删除 ${selected.size} 个草稿 SKU；关联商品已改为 ${UNKNOWN_SKU}。`)
  }

  const runAiGrounding = async () => {
    if (!image || !sourceImage.current || !layers.length) { setNotice('请先上传照片并完成至少一个层位标定。'); return }
    setAiRunning(true); setAiStatus('正在将照片发送给 AI')
    try {
      window.setTimeout(() => setAiStatus('AI 正在定位前排商品'), 500)
      const response = await fetch('/api/grounding/products', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image_data: image.dataUrl }) })
      const body = await response.json() as GroundingResponse
      if (!response.ok) throw new Error(body.error ?? 'AI 框选失败')
      const existingByName = new Map(allSkus.map((name) => [name.trim().toLocaleLowerCase(), name]))
      const newDrafts = new Set<string>()
      const additions = body.boxes.filter((box) => box.width >= 4 && box.height >= 4).map((box) => {
        const detectedName = box.label.trim().slice(0, 80) || UNKNOWN_SKU
        const resolvedSku = existingByName.get(detectedName.toLocaleLowerCase()) ?? detectedName
        if (!existingByName.has(detectedName.toLocaleLowerCase()) && resolvedSku !== UNKNOWN_SKU) newDrafts.add(resolvedSku)
        const fitted = fitBox(box, image)
        return { id: crypto.randomUUID(), expected_sku: resolvedSku, actual_sku: resolvedSku, ...fitted, crop_png: cropPng(sourceImage.current!, fitted) }
      })
      setCreatedSkus((current) => Array.from(new Set([...current, ...newDrafts])))
      setProducts((current) => [...current, ...additions])
      setSelectedProductId(additions[0]?.id ?? null)
      setToolMode('edit'); setStep(3)
      setNotice(`AI 已生成 ${additions.length} 个待审核商品框。请逐个修正框和 SKU 后导入。`)
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'AI 框选失败')
    } finally {
      setAiRunning(false); setAiStatus('')
    }
  }

  const importItems = async () => {
    if (!image || !sourceImage.current || derivedItems.length === 0 || derivedItems.some((item) => item.error) || duplicateIds.size) { setNotice('请先修正审核表中的标定错误或重复实例 ID。'); return }
    setImporting(true)
    try {
      const requiredNewSkus = Array.from(new Set(products.flatMap((item) => [item.expected_sku, item.actual_sku].filter((name): name is string => Boolean(name))))).filter((name) => !state.skus.some((item) => item.sku === name))

      // --- NEW: Prepare calibration layers data ---
      // Format: { "0": { "0": [{x, y}, ...], "1": [...] } }
      // Each face has a dict of level -> array of 4 points
      const layersPayload: Record<string, Record<string, { x: number; y: number }[]>> = {
        [String(face)]: layers.reduce((acc, layer) => {
          acc[String(layer.level)] = layer.points.map(p => ({ x: Math.round(p.x), y: Math.round(p.y) }))
          return acc
        }, {} as Record<string, { x: number; y: number }[]>)
      }
      // ------------------------------------------

      const response = await fetch('/api/imports/manual', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          new_skus: requiredNewSkus.map((name) => ({ sku: name })),
          sku_prompts: promptRows.map((item) => ({ sku: item.sku, owlv2_prompt: promptValue(item.sku) })),
          shelf_image: { shelf_id: shelfId, face, image_data: image.dataUrl },
          layers: layersPayload, // <--- NEW: Add calibration layers
          items: derivedItems.map((item) => ({
            shelf_id: shelfId,
            face,
            level: item.level,
            y_cm: item.y_cm,
            expected_sku: item.expected_sku,
            actual_sku: item.actual_sku,
            width_cm: item.width_cm,
            height_cm: item.height_cm,
            image_png: item.crop_png,
            bbox: { x: Math.round(item.x), y: Math.round(item.y), width: Math.round(item.width), height: Math.round(item.height) }
          }))
        })
      })
      const body = await response.json() as { error?: string; state: WarehouseState }
      if (!response.ok) throw new Error(body.error ?? '导入失败')
      onImported(body.state); onBack()
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '导入失败')
    } finally { setImporting(false) }
  }

  const draftOwlPrompts = async () => {
    const requests = promptRows.filter((item) => !promptValue(item.sku).trim() && item.samples.length).map((item) => ({ sku: item.sku, images: item.samples }))
    if (!requests.length) { setNotice('没有可生成的空白提示词；每个 SKU 至少需要一个正常商品裁剪图。'); return }
    setPromptRunning(true)
    try {
      const response = await fetch('/api/sku-prompts/owlv2', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ requests }) })
      const body = await response.json() as { error?: string; drafts?: { sku: string; owlv2_prompt: string }[] }
      if (!response.ok) throw new Error(body.error ?? '提示词生成失败')
      setSkuPrompts((current) => ({ ...current, ...(body.drafts ?? []).reduce<Record<string, string>>((next, item) => ({ ...next, [item.sku]: item.owlv2_prompt }), {}) }))
      setNotice('已生成 OWLv2 提示词草稿，请逐项审核后再导入。')
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '提示词生成失败')
    } finally { setPromptRunning(false) }
  }

  const selectMode = (mode: ToolMode) => {
    if (mode === 'draw' && !layers.length) { setNotice('请先完成至少一个层位标定。'); return }
    setToolMode(mode); setStep(mode === 'calibrate' ? 2 : 3); setPendingPoints([])
  }

  return <main className="manual-import-page">
    <header className="manual-header"><button className="back-button" onClick={onBack}><ArrowLeft size={17} />返回仓库</button><div><span>人工批量录入</span><small>照片标定与审核导入</small></div><div className="manual-steps"><span className={step >= 1 ? 'active' : ''}>1 照片</span><span className={step >= 2 ? 'active' : ''}>2 层位标定</span><span className={step >= 3 ? 'active' : ''}>3 商品标注</span><span className={step >= 4 ? 'active' : ''}>4 审核导入</span></div></header>
    {notice && <div className="manual-notice">{notice}</div>}
    <section className="manual-workspace">
      <aside className="manual-tools">
        <label className="field"><span>货架</span><select value={shelfId} onChange={(event) => { setShelfId(Number(event.target.value)); setLayers([]); setProducts([]) }}>{state.shelves.map((item) => <option key={item.id} value={item.id}>{item.name || `${item.id}号货架`}</option>)}</select></label>
        <label className="field"><span>拍摄面</span><select value={face} onChange={(event) => setFace(Number(event.target.value))}><option value={0}>-X 侧</option><option value={1}>+X 侧</option></select></label>
        <div className="coordinate-hint"><strong>{face === 0 ? '照片右侧为 y=0' : '照片左侧为 y=0'}</strong><span>{face === 0 ? '向左为货架局部 +Y' : '向右为货架局部 +Y'}；向上为 +Z</span></div>
        <label className="upload-button"><ImagePlus size={17} /><span>{image ? '更换整面照片' : '上传整面照片'}</span><input type="file" accept="image/*" onChange={(event) => uploadPhoto(event.target.files?.[0])} /></label>
        {image && <>
          <div className="tool-divider" />
          <div className="annotation-mode-tabs" role="group" aria-label="标注模式"><button className={toolMode === 'calibrate' ? 'active' : ''} onClick={() => selectMode('calibrate')}>标定层位</button><button className={toolMode === 'draw' ? 'active' : ''} onClick={() => selectMode('draw')}>框选商品</button><button className={toolMode === 'edit' ? 'active' : ''} onClick={() => selectMode('edit')} disabled={!products.length}>编辑选框</button></div>
          {toolMode === 'calibrate' && <><div className="tool-title"><Layers3 size={15} />层位四点标定</div><label className="field"><span>当前层号</span><select value={activeLevel} onChange={(event) => { setActiveLevel(Number(event.target.value)); setPendingPoints([]) }}>{levels.map((level) => <option key={level} value={level}>第 {level} 层</option>)}</select></label><p className="tool-help">按照片左上、右上、右下、左下顺序点击该层位的四个角。</p><div className="point-progress">{[0, 1, 2, 3].map((index) => <span key={index} className={index < pendingPoints.length ? 'done' : ''}>{index + 1}</span>)}</div><button className="text-button" onClick={() => setPendingPoints([])}>清除当前四点</button><div className="layer-list">{[...layers].sort((a, b) => a.level - b.level).map((layer) => <button key={layer.id} className={layer.level === activeLevel ? 'active' : ''} onClick={() => setActiveLevel(layer.level)}>第 {layer.level} 层 <Check size={13} /></button>)}</div></>}
          {toolMode === 'draw' && <><div className="tool-title"><MousePointer2 size={15} />固定货位框选</div><label className="field"><span>应摆 SKU</span><select value={sku} onChange={(event) => setSku(event.target.value)}>{allSkus.map((name) => <option key={name} value={name}>{name}</option>)}</select></label><label className="empty-position-toggle"><input type="checkbox" checked={emptyPosition} onChange={(event) => setEmptyPosition(event.target.checked)} /><span>框选为空位（缺货）</span></label><div className="quick-sku"><input value={newSku} placeholder="新 SKU 名称" onChange={(event) => setNewSku(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') addSku() }} /><button title="创建 SKU" onClick={addSku}><Plus size={15} /></button></div><button className="ai-grounding-button" disabled={aiRunning || !layers.length} onClick={() => void runAiGrounding()}>{aiRunning ? <LoaderCircle size={16} className="spin" /> : <Sparkles size={16} />}{aiRunning ? aiStatus : 'AI 一键框选'}</button>{aiRunning && <div className="ai-progress"><i /></div>}<p className="tool-help">默认框选为正常状态；勾选空位后只登记应摆 SKU。框体四角必须完全落在同一个标定层位内。</p></>}
          {toolMode === 'edit' && <><div className="tool-title"><PencilRuler size={15} />编辑选框</div>{selectedProduct ? <><label className="field"><span>应摆 SKU</span><select value={selectedProduct.expected_sku} onChange={(event) => updateProduct(selectedProduct.id, { expected_sku: event.target.value })}>{allSkus.map((name) => <option key={name} value={name}>{name}</option>)}</select></label><label className="field"><span>实际 SKU</span><select value={selectedProduct.actual_sku ?? ''} onChange={(event) => updateProduct(selectedProduct.id, { actual_sku: event.target.value || null })}><option value="">空（缺货）</option>{allSkus.map((name) => <option key={name} value={name}>{name}</option>)}</select></label><p className="tool-help">拖动框体移动位置，拖动四角调整范围。</p><button className="danger-outline-button" onClick={() => { setProducts((current) => current.filter((item) => item.id !== selectedProduct.id)); setSelectedProductId(null) }}><Trash2 size={15} />删除选框</button></> : <p className="tool-help">点击照片上的任意货位框后即可编辑。</p>}</>}
          {createdSkus.length > 0 && <><div className="tool-divider" /><div className="tool-title">本次草稿 SKU</div><div className="draft-sku-list">{createdSkus.map((name) => <label key={name}><input type="checkbox" checked={selectedDraftSkus.includes(name)} onChange={() => setSelectedDraftSkus((current) => current.includes(name) ? current.filter((item) => item !== name) : [...current, name])} /><span>{name}</span></label>)}</div><button className="danger-outline-button" disabled={!selectedDraftSkus.length} onClick={deleteDraftSkus}><Trash2 size={15} />删除选中 SKU</button><p className="tool-help">删除后，关联商品会改为 {UNKNOWN_SKU}。</p></>}
          <div className="tool-divider" /><button className="primary-button" disabled={!products.length} onClick={() => setStep(4)}><PackagePlus size={16} />查看审核表</button>
        </>}
      </aside>
      <section className="annotation-stage">
        {!image ? <label className="empty-photo"><Upload size={30} /><strong>上传货架单面照片</strong><span>选择货架和拍摄面后开始人工标定</span><input type="file" accept="image/*" onChange={(event) => uploadPhoto(event.target.files?.[0])} /></label> : <svg className={`annotation-canvas mode-${toolMode}`} viewBox={`0 0 ${image.width} ${image.height}`} onPointerDown={(event) => { const point = valueFromEvent(event, image); if (toolMode === 'calibrate') addLayerPoint(point); else if (toolMode === 'draw') { event.currentTarget.setPointerCapture(event.pointerId); setDragStart(point); setDragCurrent(point) } else setSelectedProductId(null) }} onPointerMove={(event) => { const point = valueFromEvent(event, image); if (boxDrag) updateProduct(boxDrag.id, boxAfterDrag(boxDrag, point, image)); else if (dragStart) setDragCurrent(point) }} onPointerUp={(event) => { const point = valueFromEvent(event, image); if (boxDrag) { updateProductBox(boxDrag.id, boxAfterDrag(boxDrag, point, image)); setBoxDrag(null); return } if (dragStart && toolMode === 'draw') { addProduct(dragStart, point); setDragStart(null); setDragCurrent(null) } }}>
          <image href={image.url} width={image.width} height={image.height} />
          {layers.map((layer) => <g key={layer.id}><polygon points={layer.points.map((point) => `${point.x},${point.y}`).join(' ')} className={`layer-shape ${layer.level === activeLevel ? 'selected' : ''}`} /><text x={layer.points[0].x + 8} y={layer.points[0].y + 20} className="annotation-label">第 {layer.level} 层</text></g>)}
          {pendingPoints.map((point, index) => <g key={`${point.x}-${point.y}`}><circle cx={point.x} cy={point.y} r="7" className="pending-point" /><text x={point.x + 10} y={point.y - 8} className="annotation-label">{index + 1}</text></g>)}
          {products.map((product) => <g key={product.id} className={selectedProductId === product.id ? 'selected-product' : ''} onPointerDown={(event) => { if (toolMode !== 'edit') return; event.stopPropagation(); const svg = event.currentTarget.ownerSVGElement; if (!svg) return; svg.setPointerCapture(event.pointerId); setSelectedProductId(product.id); setBoxDrag({ id: product.id, handle: 'move', start: valueFromSvg(event, svg, image), original: product }) }}><rect x={product.x} y={product.y} width={product.width} height={product.height} className="product-shape" /><text x={product.x + 4} y={product.y - 5} className="product-label">{product.actual_sku ?? `${product.expected_sku}（空）`}</text>{toolMode === 'edit' && selectedProductId === product.id && <>{([['nw', product.x, product.y], ['ne', product.x + product.width, product.y], ['se', product.x + product.width, product.y + product.height], ['sw', product.x, product.y + product.height]] as Array<[Handle, number, number]>).map(([handle, x, y]) => <circle key={handle} cx={x} cy={y} r="7" className="box-handle" onPointerDown={(event) => { event.stopPropagation(); const svg = event.currentTarget.ownerSVGElement; if (!svg) return; svg.setPointerCapture(event.pointerId); setBoxDrag({ id: product.id, handle, start: valueFromSvg(event, svg, image), original: product }) }} />)}</>}</g>)}
          {dragStart && dragCurrent && <rect x={Math.min(dragStart.x, dragCurrent.x)} y={Math.min(dragStart.y, dragCurrent.y)} width={Math.abs(dragStart.x - dragCurrent.x)} height={Math.abs(dragStart.y - dragCurrent.y)} className="product-shape draft" />}
        </svg>}
      </section>
    </section>
    {step === 4 && <section className="review-band"><div className="review-heading"><div><span>导入审核表</span><strong>{derivedItems.length} 个固定货位</strong></div><button className="primary-button" disabled={importing || !derivedItems.length || Boolean(duplicateIds.size) || derivedItems.some((item) => item.error)} onClick={() => void importItems()}>{importing ? '正在导入' : '确认批量导入'}</button></div><div className="sku-prompt-review"><div><strong>OWLv2 商品提示词</strong><small>按 SKU 聚合；仅正常裁剪图会用于生成草稿。</small></div><button className="secondary-button" disabled={promptRunning || !promptRows.some((item) => !promptValue(item.sku).trim() && item.samples.length)} onClick={() => void draftOwlPrompts()}>{promptRunning ? '正在生成' : 'AI 一键编写'}</button>{promptRows.map((item) => <label key={item.sku}><span>{item.sku}</span><input value={promptValue(item.sku)} placeholder={item.samples.length ? '可生成英文提示词' : '无正常裁剪图，请手工填写'} onChange={(event) => setSkuPrompts((current) => ({ ...current, [item.sku]: event.target.value }))} /><small>{item.samples.length} 张正常样本</small></label>)}</div><div className="review-table-wrap"><table className="review-table"><thead><tr><th>0.png</th><th>位置 ID</th><th>应摆 SKU</th><th>实际 SKU</th><th>面 / 层</th><th>Y 中心</th><th>宽 / 高</th><th>状态</th><th></th></tr></thead><tbody>{derivedItems.map((item) => <tr key={item.id}><td><img src={item.crop_png} alt={`${item.expected_sku} 货位裁剪图`} /></td><td><code>{item.slot_id || '-'}</code></td><td><select value={item.expected_sku} onChange={(event) => updateProduct(item.id, { expected_sku: event.target.value })}>{allSkus.map((name) => <option key={name} value={name}>{name}</option>)}</select></td><td><select value={item.actual_sku ?? ''} onChange={(event) => updateProduct(item.id, { actual_sku: event.target.value || null })}><option value="">空</option>{allSkus.map((name) => <option key={name} value={name}>{name}</option>)}</select></td><td>{face === 0 ? '-X' : '+X'} / {item.level}</td><td><input value={item.y_cm} onChange={(event) => updateProduct(item.id, { yOverride: Number(event.target.value) })} /></td><td><span className="dimension-input"><input value={item.width_cm} onChange={(event) => updateProduct(item.id, { widthOverride: Number(event.target.value) })} /><input value={item.height_cm} onChange={(event) => updateProduct(item.id, { heightOverride: Number(event.target.value) })} /></span></td><td>{item.error ? <span className="row-error">{item.error}</span> : duplicateIds.has(item.slot_id) ? <span className="row-error">ID 重复</span> : <span className={`status-badge status-${slotStatus(item.expected_sku, item.actual_sku)}`}>{slotStatus(item.expected_sku, item.actual_sku)}</span>}</td><td><button className="icon-button" title="编辑货位" onClick={() => { setSelectedProductId(item.id); setToolMode('edit'); setStep(3) }}><PencilRuler size={15} /></button><button className="icon-button" title="移除货位" onClick={() => setProducts((current) => current.filter((product) => product.id !== item.id))}><Trash2 size={15} /></button></td></tr>)}</tbody></table></div></section>}
  </main>
}
