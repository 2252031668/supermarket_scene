import { useMemo, useRef, useState } from 'react'
import { ArrowLeft, Check, ImagePlus, Layers3, MousePointer2, PackagePlus, Plus, Trash2, Upload } from 'lucide-react'
import type { Shelf, ShelfType, WarehouseState } from './types'

type Point = { x: number; y: number }
type LayerCalibration = { id: string; level: number; points: [Point, Point, Point, Point] }
type ProductBox = { id: string; sku: string; x: number; y: number; width: number; height: number; crop_png: string; yOverride?: number; widthOverride?: number; heightOverride?: number }
type DerivedItem = ProductBox & { level: number; y_cm: number; width_cm: number; height_cm: number; slot_id: string; error?: string }

function valueFromEvent(event: React.PointerEvent<SVGSVGElement>, image: { width: number; height: number }): Point {
  const rect = event.currentTarget.getBoundingClientRect()
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
  // Points are photo top-left, top-right, bottom-right, bottom-left. Destination is (u, v), v up.
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
  return {
    x: (transform[0] * point.x + transform[1] * point.y + transform[2]) / denominator,
    y: (transform[3] * point.x + transform[4] * point.y + transform[5]) / denominator,
  }
}

function openingHeightCm(type: ShelfType, level: number) {
  void level
  return Math.max(0, (type.level_spacing - type.panel_thick) * 100)
}

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

export function ManualImportPage({ state, onBack, onImported }: { state: WarehouseState; onBack: () => void; onImported: (next: WarehouseState) => void }) {
  const [shelfId, setShelfId] = useState(state.shelves[0]?.id ?? 0)
  const [face, setFace] = useState(0)
  const [step, setStep] = useState(1)
  const [image, setImage] = useState<{ url: string; width: number; height: number } | null>(null)
  const sourceImage = useRef<HTMLImageElement | null>(null)
  const [activeLevel, setActiveLevel] = useState(0)
  const [layers, setLayers] = useState<LayerCalibration[]>([])
  const [pendingPoints, setPendingPoints] = useState<Point[]>([])
  const [sku, setSku] = useState(state.skus[0]?.sku ?? '')
  const [newSku, setNewSku] = useState('')
  const [createdSkus, setCreatedSkus] = useState<string[]>([])
  const [products, setProducts] = useState<ProductBox[]>([])
  const [dragStart, setDragStart] = useState<Point | null>(null)
  const [dragCurrent, setDragCurrent] = useState<Point | null>(null)
  const [notice, setNotice] = useState('')
  const [importing, setImporting] = useState(false)

  const shelf = state.shelves.find((item) => item.id === shelfId)
  const shelfType = state.shelf_types.find((item) => item.id === shelf?.shelf_type_id) ?? state.shelf_types[0]
  const levels = shelfType ? Array.from({ length: shelfType.num_levels }, (_, index) => index) : []
  const activeLayer = layers.find((item) => item.level === activeLevel)
  const allSkus = useMemo(() => [...state.skus.map((item) => item.sku), ...createdSkus], [state.skus, createdSkus])

  const derivedItems = useMemo<DerivedItem[]>(() => products.map((product) => {
    if (!shelfType || layers.length === 0) return { ...product, level: -1, y_cm: 0, width_cm: 0, height_cm: 0, slot_id: '', error: '缺少层位标定' }
    const corners = [
      { x: product.x, y: product.y },
      { x: product.x + product.width, y: product.y },
      { x: product.x + product.width, y: product.y + product.height },
      { x: product.x, y: product.y + product.height },
    ]
    const matches = layers.flatMap((candidate) => {
      const candidateTransform = homography(candidate.points)
      const mappedCorners = corners.map((corner) => project(candidateTransform, corner))
      const containsWholeBox = mappedCorners.every((corner) => corner && corner.x >= 0 && corner.x <= 1 && corner.y >= 0 && corner.y <= 1)
      return containsWholeBox && candidateTransform ? [{ layer: candidate, transform: candidateTransform }] : []
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
    const openCm = openingHeightCm(shelfType, layer.level)
    const mappedY = (face === 1 ? center.x : 1 - center.x) * lengthCm
    const mappedWidth = Math.abs(right.x - left.x) * lengthCm
    const mappedHeight = Math.abs(top.y - bottom.y) * openCm
    const y_cm = product.yOverride ?? Math.round(mappedY)
    const width_cm = product.widthOverride ?? Math.round(mappedWidth * 10) / 10
    const height_cm = product.heightOverride ?? Math.round(mappedHeight * 10) / 10
    const slot_id = `${shelfId}-${face}-${layer.level}-${y_cm}`
    return { ...product, level: layer.level, y_cm, width_cm, height_cm, slot_id }
  }), [products, layers, shelfType, shelfId, face])

  const duplicateIds = useMemo(() => new Set(derivedItems.filter((item, index, items) => item.slot_id && items.findIndex((other) => other.slot_id === item.slot_id) !== index).map((item) => item.slot_id)), [derivedItems])

  const uploadPhoto = (file?: File) => {
    if (!file) return
    const loaded = new Image()
    loaded.onload = () => {
      sourceImage.current = loaded
      setImage({ url: loaded.src, width: loaded.naturalWidth, height: loaded.naturalHeight })
      setLayers([])
      setProducts([])
      setPendingPoints([])
      setStep(2)
      setNotice('请先为每个需要录入的层位按左上、右上、右下、左下顺序点击四个角。')
    }
    loaded.src = URL.createObjectURL(file)
  }

  const addLayerPoint = (point: Point) => {
    if (pendingPoints.length >= 4) return
    const next = [...pendingPoints, point]
    if (next.length < 4) {
      setPendingPoints(next)
      return
    }
    const calibration: LayerCalibration = { id: `layer-${activeLevel}`, level: activeLevel, points: next as LayerCalibration['points'] }
    setLayers((current) => [...current.filter((item) => item.level !== activeLevel), calibration])
    setPendingPoints([])
    setNotice(`第 ${activeLevel} 层标定完成。可选择其他层号继续标定。`)
  }

  const addSku = () => {
    const name = newSku.trim()
    if (!name) return
    if (allSkus.includes(name)) {
      setSku(name)
      setNewSku('')
      return
    }
    setCreatedSkus((current) => [...current, name])
    setSku(name)
    setNewSku('')
  }

  const addProduct = (start: Point, end: Point) => {
    if (!sourceImage.current || layers.length === 0) {
      setNotice('请先完成至少一个层位四点标定，再框选商品。')
      return
    }
    if (!sku) {
      setNotice('请先选择或创建 SKU。')
      return
    }
    const x = Math.min(start.x, end.x)
    const y = Math.min(start.y, end.y)
    const width = Math.abs(start.x - end.x)
    const height = Math.abs(start.y - end.y)
    if (width < 4 || height < 4) return
    const box = { x, y, width, height }
    setProducts((current) => [...current, { id: crypto.randomUUID(), sku, ...box, crop_png: cropPng(sourceImage.current!, box) }])
    setNotice('商品已加入审核表。')
  }

  const importItems = async () => {
    if (!sourceImage.current || derivedItems.length === 0 || derivedItems.some((item) => item.error) || duplicateIds.size) {
      setNotice('请先修正审核表中的标定错误或重复实例 ID。')
      return
    }
    setImporting(true)
    try {
      const response = await fetch('/api/imports/manual', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          new_skus: createdSkus.map((name) => ({ sku: name })),
          items: derivedItems.map((item) => ({
            shelf_id: shelfId, face, level: item.level, y_cm: item.y_cm, sku: item.sku,
            width_cm: item.width_cm, height_cm: item.height_cm,
            image_png: item.crop_png,
          })),
        }),
      })
      const body = await response.json() as { error?: string; state: WarehouseState }
      if (!response.ok) throw new Error(body.error ?? '导入失败')
      onImported(body.state)
      onBack()
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '导入失败')
    } finally {
      setImporting(false)
    }
  }

  const updateProduct = (id: string, patch: Partial<ProductBox>) => setProducts((current) => current.map((item) => item.id === id ? { ...item, ...patch } : item))

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
          <div className="tool-title"><Layers3 size={15} />层位四点标定</div>
          <label className="field"><span>当前层号</span><select value={activeLevel} onChange={(event) => { setActiveLevel(Number(event.target.value)); setPendingPoints([]) }}>{levels.map((level) => <option key={level} value={level}>第 {level} 层</option>)}</select></label>
          <p className="tool-help">按照片左上、右上、右下、左下顺序点击该层位的四个角。</p>
          <div className="point-progress">{[0, 1, 2, 3].map((index) => <span key={index} className={index < pendingPoints.length ? 'done' : ''}>{index + 1}</span>)}</div>
          <button className="text-button" onClick={() => setPendingPoints([])}>清除当前四点</button>
          <div className="layer-list">{[...layers].sort((a, b) => a.level - b.level).map((layer) => <button key={layer.id} className={layer.level === activeLevel ? 'active' : ''} onClick={() => setActiveLevel(layer.level)}>第 {layer.level} 层 <Check size={13} /></button>)}</div>
          <div className="tool-divider" />
          <div className="tool-title"><MousePointer2 size={15} />商品框选</div>
          <label className="field"><span>商品 SKU</span><select value={sku} onChange={(event) => setSku(event.target.value)}>{allSkus.map((name) => <option key={name} value={name}>{name}</option>)}</select></label>
          <div className="quick-sku"><input value={newSku} placeholder="新 SKU 名称" onChange={(event) => setNewSku(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') addSku() }} /><button title="创建 SKU" onClick={addSku}><Plus size={15} /></button></div>
          <p className="tool-help">完成层位标定后可连续框选商品。系统要求商品框四角完全落在同一个标定层位内。</p>
          <button className="primary-button" disabled={!products.length} onClick={() => setStep(4)}><PackagePlus size={16} />查看审核表</button>
        </>}
      </aside>
      <section className="annotation-stage">
        {!image ? <label className="empty-photo"><Upload size={30} /><strong>上传货架单面照片</strong><span>选择货架和拍摄面后开始人工标定</span><input type="file" accept="image/*" onChange={(event) => uploadPhoto(event.target.files?.[0])} /></label> : <svg className="annotation-canvas" viewBox={`0 0 ${image.width} ${image.height}`} onPointerDown={(event) => {
          const point = valueFromEvent(event, image)
          if (step === 2) addLayerPoint(point)
          else if (step === 3) { event.currentTarget.setPointerCapture(event.pointerId); setDragStart(point); setDragCurrent(point) }
        }} onPointerMove={(event) => { if (dragStart) setDragCurrent(valueFromEvent(event, image)) }} onPointerUp={(event) => {
          if (!dragStart || step !== 3) return
          addProduct(dragStart, valueFromEvent(event, image))
          setDragStart(null); setDragCurrent(null)
        }}>
          <image href={image.url} width={image.width} height={image.height} />
          {layers.map((layer) => <g key={layer.id}><polygon points={layer.points.map((point) => `${point.x},${point.y}`).join(' ')} className={`layer-shape ${layer.level === activeLevel ? 'selected' : ''}`} /><text x={layer.points[0].x + 8} y={layer.points[0].y + 20} className="annotation-label">第 {layer.level} 层</text></g>)}
          {pendingPoints.map((point, index) => <g key={`${point.x}-${point.y}`}><circle cx={point.x} cy={point.y} r="7" className="pending-point" /><text x={point.x + 10} y={point.y - 8} className="annotation-label">{index + 1}</text></g>)}
          {products.map((product) => <g key={product.id}><rect x={product.x} y={product.y} width={product.width} height={product.height} className="product-shape" /><text x={product.x + 4} y={product.y - 5} className="product-label">{product.sku}</text></g>)}
          {dragStart && dragCurrent && <rect x={Math.min(dragStart.x, dragCurrent.x)} y={Math.min(dragStart.y, dragCurrent.y)} width={Math.abs(dragStart.x - dragCurrent.x)} height={Math.abs(dragStart.y - dragCurrent.y)} className="product-shape draft" />}
        </svg>}
        {image && <div className="stage-controls"><button className={step === 2 ? 'active' : ''} onClick={() => { setStep(2); setDragStart(null) }}>标定层位</button><button className={step === 3 ? 'active' : ''} onClick={() => { setStep(3); setPendingPoints([]) }} disabled={!layers.length}>框选商品</button><span>{products.length} 个待审核商品</span></div>}
      </section>
    </section>
    {step === 4 && <section className="review-band"><div className="review-heading"><div><span>导入审核表</span><strong>{derivedItems.length} 个商品实例</strong></div><button className="primary-button" disabled={importing || !derivedItems.length || Boolean(duplicateIds.size) || derivedItems.some((item) => item.error)} onClick={() => void importItems()}>{importing ? '正在导入' : '确认批量导入'}</button></div><div className="review-table-wrap"><table className="review-table"><thead><tr><th>0.png</th><th>实例 ID</th><th>SKU</th><th>面 / 层</th><th>Y 中心 (cm)</th><th>宽 / 高 (cm)</th><th>状态</th><th></th></tr></thead><tbody>{derivedItems.map((item) => <tr key={item.id}><td><img src={item.crop_png} alt={`${item.sku} 裁剪图`} /></td><td><code>{item.slot_id || '-'}</code></td><td><select value={item.sku} onChange={(event) => updateProduct(item.id, { sku: event.target.value })}>{allSkus.map((name) => <option key={name} value={name}>{name}</option>)}</select></td><td>{face === 0 ? '-X' : '+X'} / {item.level}</td><td><input value={item.y_cm} onChange={(event) => updateProduct(item.id, { yOverride: Number(event.target.value) })} /></td><td><span className="dimension-input"><input value={item.width_cm} onChange={(event) => updateProduct(item.id, { widthOverride: Number(event.target.value) })} /><input value={item.height_cm} onChange={(event) => updateProduct(item.id, { heightOverride: Number(event.target.value) })} /></span></td><td>{item.error ? <span className="row-error">{item.error}</span> : duplicateIds.has(item.slot_id) ? <span className="row-error">ID 重复</span> : <span className="row-ok">可导入</span>}</td><td><button className="icon-button" title="移除商品" onClick={() => setProducts((current) => current.filter((product) => product.id !== item.id))}><Trash2 size={15} /></button></td></tr>)}</tbody></table></div></section>}
  </main>
}
