import { useState } from 'react'
import { ArrowLeft, Download, ImagePlus, LoaderCircle, Play, ScanLine, Upload, X } from 'lucide-react'
import type { ImageStitchReport } from './types'

type Point = { x: number; y: number }

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, { ...options, headers: { 'Content-Type': 'application/json', ...(options?.headers ?? {}) } })
  const text = await response.text()
  let body: (T & { error?: string }) | undefined
  try { body = text ? JSON.parse(text) as T & { error?: string } : undefined } catch { throw new Error('服务器返回了无法解析的响应') }
  if (!response.ok) throw new Error(body?.error ?? '请求失败 (' + response.status + ')')
  if (!body) throw new Error('服务器返回空响应')
  return body
}

function readFile(file: File) {
  return new Promise<{ name: string; data: string }>((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve({ name: file.name, data: String(reader.result) })
    reader.onerror = () => reject(new Error('无法读取图片'))
    reader.readAsDataURL(file)
  })
}

export function ImageStitchPage({ onBack, onUseForImport }: { onBack: () => void; onUseForImport: (url: string) => void }) {
  const [images, setImages] = useState<Array<{ name: string; data: string }>>([])
  const [mainIndex, setMainIndex] = useState(0)
  const [report, setReport] = useState<ImageStitchReport | null>(null)
  const [showDebug, setShowDebug] = useState(false)
  const [points, setPoints] = useState<Point[]>([])
  const [notice, setNotice] = useState('')
  const [running, setRunning] = useState(false)
  const [rectifying, setRectifying] = useState(false)

  const artifactUrl = (name: string) => '/api/image-stitch/runs/' + encodeURIComponent(report!.run_id) + '/artifact/' + encodeURIComponent(name)
  const imageUrl = report ? artifactUrl(report.artifacts.final_image) : ''

  const run = async () => {
    if (images.length < 2) { setNotice('请至少上传两张同一货架面的重叠照片'); return }
    setRunning(true); setNotice('正在配准并拼接照片')
    try {
      const result = await request<{ report: ImageStitchReport }>('/api/image-stitch', { method: 'POST', body: JSON.stringify({ images: images.map((image) => image.data), main_index: mainIndex }) })
      const skipped = result.report.skipped.map((item) => '#' + (item.index + 1)).join('、')
      setReport(result.report); setPoints([]); setShowDebug(false); setNotice(skipped ? '已生成可拼接结果，未接入图片：' + skipped : '拼接完成，请框选货架四个外角以校正正面视图')
    } catch (error) { setNotice(error instanceof Error ? error.message : '图片拼接失败') } finally { setRunning(false) }
  }

  const rectify = async () => {
    if (!report || points.length !== 4) { setNotice('请按左上、右上、右下、左下顺序点击四个货架外角'); return }
    setRectifying(true)
    try {
      const result = await request<{ report: ImageStitchReport }>('/api/image-stitch/runs/' + encodeURIComponent(report.run_id) + '/rectify', { method: 'POST', body: JSON.stringify({ points: points.map((point) => [point.x, point.y]) }) })
      setReport(result.report); setPoints([]); setNotice('正面校正完成')
    } catch (error) { setNotice(error instanceof Error ? error.message : '正面校正失败') } finally { setRectifying(false) }
  }

  const addPoint = (event: React.PointerEvent<HTMLImageElement>) => {
    if (!report || points.length >= 4) return
    const bounds = event.currentTarget.getBoundingClientRect()
    setPoints((current) => [...current, { x: (event.clientX - bounds.left) * report.width / bounds.width, y: (event.clientY - bounds.top) * report.height / bounds.height }])
  }

  return <main className="inspection-page">
    <header className="manual-header"><button className="back-button" onClick={onBack}><ArrowLeft size={16} />返回管理</button><div><span>图片拼接</span><small>多张局部货架照片配准、融合与正面校正</small></div></header>
    {notice && <div className="manual-notice">{notice}</div>}
    <section className="inspection-workspace">
      <aside className="inspection-tools">
        <div className="tool-title"><ImagePlus size={16} />拼接照片</div>
        <label className="upload-button"><Upload size={16} /><span>添加局部货架图片</span><input type="file" multiple accept="image/png,image/jpeg,image/webp" onChange={(event) => { const files = Array.from(event.target.files ?? []); if (files.length) void Promise.all(files.map(readFile)).then((items) => { setImages((current) => [...current, ...items].slice(0, 8)); setReport(null); setPoints([]) }).catch((error) => setNotice(error.message)) }} /></label>
        <div className="stitch-input-list">{images.map((image, index) => <div key={image.name + index}><img src={image.data} alt={image.name} /><span>{index + 1}. {image.name}</span><label className="stitch-main-plane"><input type="radio" name="stitch-main-plane" checked={mainIndex === index} onChange={() => { setMainIndex(index); setReport(null); setPoints([]) }} />主平面</label><button title="移除图片" onClick={() => { setImages((current) => current.filter((_, itemIndex) => itemIndex !== index)); setMainIndex((current) => current === index ? 0 : current > index ? current - 1 : current); setReport(null); setPoints([]) }}><X size={14} /></button></div>)}</div>
        <button className="ai-grounding-button" disabled={running || images.length < 2} onClick={() => void run()}>{running ? <LoaderCircle size={16} className="spin" /> : <Play size={16} />}{running ? '正在拼接' : '运行拼接'}</button>
        {report && <>{report.artifacts.warped && <button className="secondary-button" onClick={() => setShowDebug((current) => !current)}><ImagePlus size={15} />{showDebug ? '隐藏调试图' : '显示调试图'}</button>}<button className="secondary-button" disabled={rectifying || points.length !== 4} onClick={() => void rectify()}>{rectifying ? <LoaderCircle size={15} className="spin" /> : <ScanLine size={15} />}正面校正</button><a className="secondary-button stitch-download" href={imageUrl} download="shelf-stitch.png"><Download size={15} />保存图片</a><button className="primary-button" onClick={() => onUseForImport(imageUrl)}>用于人工批量录入</button></>}
      </aside>
      <section className="inspection-results">{report ? <><div className="stitch-summary"><span>主平面：#{report.main_index + 1}</span><span>已拼入：{report.used_indices.map((index) => '#' + (index + 1)).join('、')}</span><span>连接：{report.pairs.map((pair) => '#' + (pair.source_index + 1) + ' -> #' + (pair.destination_index + 1)).join('，')}</span><span>接缝：GraphCut</span><span>融合：{report.rendering.blend_bands} 层</span>{report.skipped.length > 0 && <span>未接入：{report.skipped.map((item) => '#' + (item.index + 1)).join('、')}</span>}</div>{showDebug && <div className="stitch-debug-grid"><figure><img src={artifactUrl(report.artifacts.warped)} alt="变换预览" /><figcaption>变换预览</figcaption></figure><figure><img src={artifactUrl(report.artifacts.seams)} alt="GraphCut 接缝蒙版" /><figcaption>GraphCut 接缝蒙版</figcaption></figure></div>}<div className="stitch-image-frame"><img className="inspection-source-image" src={imageUrl} alt="货架拼接结果" onPointerDown={addPoint} />{points.length > 0 && <svg viewBox={'0 0 ' + report.width + ' ' + report.height} aria-hidden="true">{points.map((point, index) => <g key={index}><circle cx={point.x} cy={point.y} r={Math.max(report.width, report.height) * 0.008} /><text x={point.x} y={point.y}>{index + 1}</text></g>)}{points.length === 4 && <polygon points={points.map((point) => point.x + ',' + point.y).join(' ')} />}</svg>}</div></> : <div className="inspection-empty"><ImagePlus size={28} /><strong>添加照片后运行拼接</strong></div>}</section>
    </section>
  </main>
}
