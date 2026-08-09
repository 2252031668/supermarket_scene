import { useEffect, useState } from 'react'
import { ArrowLeft, CheckSquare, Images, LoaderCircle, Play, Save, ScanLine, Square, Upload } from 'lucide-react'
import type { VisionConfig, VisionInspectionReport, WarehouseState } from './types'

const defaultConfig: VisionConfig = { min_current_coverage: 0.05, analysis_center_ratio: 0.8, lab_distance_threshold: 12, slot_change_ratio_threshold: 0.15, dino_confidence_threshold: 0.72, ambiguity_margin: 0.05, vlm_fallback: false, vlm_top_k: 4 }

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
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(new Error('无法读取图片'))
    reader.readAsDataURL(file)
  })
}

export function VisionInspectionPage({ onBack, onApplied }: { onBack: () => void; onApplied: (state: WarehouseState) => void }) {
  const [config, setConfig] = useState<VisionConfig>(defaultConfig)
  const [imageData, setImageData] = useState('')
  const [imageName, setImageName] = useState('')
  const [report, setReport] = useState<VisionInspectionReport | null>(null)
  const [notice, setNotice] = useState('')
  const [running, setRunning] = useState(false)
  const [applying, setApplying] = useState(false)
  const [showDebug, setShowDebug] = useState(false)

  useEffect(() => {
    void request<VisionConfig>('/api/vision/config').then(setConfig).catch((error) => setNotice(error instanceof Error ? error.message : '无法读取巡检配置'))
  }, [])

  const saveConfig = async () => {
    try {
      const result = await request<{ inspection: VisionConfig }>('/api/vision/config', { method: 'PUT', body: JSON.stringify(config) })
      setConfig(result.inspection); setNotice('巡检配置已保存')
    } catch (error) { setNotice(error instanceof Error ? error.message : '保存配置失败') }
  }

  const run = async () => {
    if (!imageData) { setNotice('请先选择局部货架图片'); return }
    setRunning(true); setNotice('正在配准图片并识别异常货位')
    try {
      const result = await request<{ report: VisionInspectionReport }>('/api/vision/inspect', { method: 'POST', body: JSON.stringify({ image_data: imageData, config, debug: true }) })
      setReport(result.report)
      setShowDebug(false)
      setNotice('识别完成，共有 ' + result.report.slots.filter((row) => row.selected).length + ' 个待应用结果')
    } catch (error) { setNotice(error instanceof Error ? error.message : '巡检失败') } finally { setRunning(false) }
  }

  const setAll = (selected: boolean) => setReport((current) => current && ({ ...current, slots: current.slots.map((row) => ({ ...row, selected })) }))

  const apply = async () => {
    if (!report) return
    const slotIds = report.slots.filter((row) => row.selected).map((row) => row.slot_id)
    if (!slotIds.length) { setNotice('请先勾选需要应用的识别结果'); return }
    setApplying(true)
    try {
      const result = await request<{ slots: VisionInspectionReport['slots']; state: WarehouseState }>('/api/vision/runs/' + encodeURIComponent(report.run_id) + '/apply', { method: 'POST', body: JSON.stringify({ slot_ids: slotIds }) })
      onApplied(result.state)
      setReport((current) => current && ({ ...current, slots: current.slots.map((row) => slotIds.includes(row.slot_id) ? { ...row, selected: false } : row) }))
      setNotice('已应用 ' + result.slots.length + ' 个货位修改')
    } catch (error) { setNotice(error instanceof Error ? error.message : '应用修改失败') } finally { setApplying(false) }
  }

  const debugArtifacts = report ? [
    { label: '对齐基准图', name: report.artifacts.aligned_reference },
    { label: '当前重叠图', name: report.artifacts.current_overlap },
    { label: '差分图', name: report.artifacts.difference },
    { label: '异常候选框', name: report.artifacts.candidate_boxes },
  ].filter((artifact): artifact is { label: string; name: string } => Boolean(artifact.name)) : []

  const artifactUrl = (name: string) => '/api/vision/runs/' + encodeURIComponent(report!.run_id) + '/artifact/' + encodeURIComponent(name)
  const resultOverlay = report?.artifacts.result_overlay
  const previewInset = ((1 - config.analysis_center_ratio) * 50).toFixed(2) + '%'

  return <main className="inspection-page">
    <header className="manual-header"><button className="back-button" onClick={onBack}><ArrowLeft size={16} />返回管理</button><div><span>巡检识别</span><small>局部照片差分、DINOv2 SKU 比对与 Ark 保底</small></div></header>
    {notice && <div className="manual-notice">{notice}</div>}
    <section className="inspection-workspace">
      <aside className="inspection-tools">
        <div className="tool-title"><ScanLine size={16} />巡检图片</div>
        <label className="upload-button"><Upload size={16} /><span>{imageName || '上传局部货架图片'}</span><input type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => { const file = event.target.files?.[0]; if (file) void readFile(file).then((data) => { setImageData(data); setImageName(file.name); setReport(null); setShowDebug(false) }).catch((error) => setNotice(error.message)) }} /></label>
        <div className="tool-divider" />
        <label className="inspection-field"><span>最低配准覆盖率 {(config.min_current_coverage * 100).toFixed(0)}%</span><input type="range" min="0.01" max="0.3" step="0.01" value={config.min_current_coverage} onChange={(event) => setConfig({ ...config, min_current_coverage: Number(event.target.value) })} /></label>
        <label className="inspection-field"><span>中心分析范围 {(config.analysis_center_ratio * 100).toFixed(0)}%</span><input type="range" min="0.5" max="1" step="0.01" value={config.analysis_center_ratio} onChange={(event) => setConfig({ ...config, analysis_center_ratio: Number(event.target.value) })} /></label>
        <label className="inspection-field"><span>Lab 异常阈值 {config.lab_distance_threshold.toFixed(0)}</span><input type="range" min="1" max="60" step="1" value={config.lab_distance_threshold} onChange={(event) => setConfig({ ...config, lab_distance_threshold: Number(event.target.value) })} /></label>
        <label className="inspection-field"><span>Slot 异常占比 {(config.slot_change_ratio_threshold * 100).toFixed(0)}%</span><input type="range" min="0.01" max="0.8" step="0.01" value={config.slot_change_ratio_threshold} onChange={(event) => setConfig({ ...config, slot_change_ratio_threshold: Number(event.target.value) })} /></label>
        <label className="inspection-field"><span>DINO 置信度 {config.dino_confidence_threshold.toFixed(2)}</span><input type="range" min="0.4" max="0.99" step="0.01" value={config.dino_confidence_threshold} onChange={(event) => setConfig({ ...config, dino_confidence_threshold: Number(event.target.value) })} /></label>
        <label className="inspection-field"><span>并列差值 {config.ambiguity_margin.toFixed(2)}</span><input type="range" min="0" max="0.3" step="0.01" value={config.ambiguity_margin} onChange={(event) => setConfig({ ...config, ambiguity_margin: Number(event.target.value) })} /></label>
        <label className="empty-position-toggle"><input type="checkbox" checked={config.vlm_fallback} onChange={(event) => setConfig({ ...config, vlm_fallback: event.target.checked })} />Ark VLM 保底</label>
        <button className="secondary-button" onClick={() => void saveConfig()}><Save size={15} />保存配置</button>
        <button className="ai-grounding-button" disabled={running || !imageData} onClick={() => void run()}>{running ? <LoaderCircle size={16} className="spin" /> : <Play size={16} />}运行识别</button>
      </aside>
      <section className="inspection-results">{resultOverlay ? <div className="inspection-image-frame"><img className="inspection-source-image" src={artifactUrl(resultOverlay)} alt="巡检最终结果" /></div> : imageData ? <div className="inspection-image-frame"><img className="inspection-source-image" src={imageData} alt={imageName || '待巡检货架图片'} /><span className="analysis-roi-preview" style={{ inset: previewInset }} /></div> : <div className="inspection-empty"><ScanLine size={28} /><strong>上传图片后运行巡检</strong></div>}</section>
    </section>
    {report && debugArtifacts.length > 0 && showDebug && <section className="debug-gallery" aria-label="巡检调试图">
      {debugArtifacts.map((artifact) => <figure key={artifact.name}><img loading="lazy" src={artifactUrl(artifact.name)} alt={artifact.label} /><figcaption>{artifact.label}</figcaption></figure>)}
    </section>}
    {report && <section className="review-band inspection-review">
      <div className="review-heading"><div><span>识别结果</span><strong>{report.shelf_id} 号货架 / {report.face === 0 ? '-X 面' : '+X 面'}</strong>{report.analysis && <small>中心分析 {(report.analysis.center_ratio * 100).toFixed(0)}% / 跳过边缘 {report.analysis.skipped_edge_slots} 个位置</small>}</div><div className="inspection-actions">{debugArtifacts.length > 0 && <button className="secondary-button" onClick={() => setShowDebug((visible) => !visible)}><Images size={15} />{showDebug ? '隐藏调试图' : '显示调试图'}</button>}<button className="secondary-button" onClick={() => setAll(true)}><CheckSquare size={15} />全选</button><button className="secondary-button" onClick={() => setAll(false)}><Square size={15} />取消全选</button><button className="primary-button" disabled={applying} onClick={() => void apply()}>{applying ? <LoaderCircle size={16} className="spin" /> : <Save size={16} />}应用修改</button></div></div>
      <div className="review-table-wrap"><table className="review-table inspection-table"><thead><tr><th>应用</th><th>固定位置 ID</th><th>应摆 SKU</th><th>识别 SKU</th><th>状态</th><th>异常占比</th><th>置信度</th><th>来源</th><th>结论</th></tr></thead><tbody>{report.slots.map((row) => <tr key={row.slot_id} className={row.selected ? 'inspection-selected' : ''}><td><input aria-label={'应用 ' + row.slot_id} type="checkbox" checked={row.selected} onChange={() => setReport((current) => current && ({ ...current, slots: current.slots.map((item) => item.slot_id === row.slot_id ? { ...item, selected: !item.selected } : item) }))} /></td><td><code>{row.slot_id}</code></td><td>{row.expected_sku}</td><td>{row.actual_sku ?? '缺货'}</td><td><span className={'status-badge status-' + row.status}>{row.status}</span></td><td>{(row.difference_ratio * 100).toFixed(1)}%</td><td>{row.confidence === null ? '-' : row.confidence.toFixed(3)}</td><td>{row.source}</td><td>{row.reason}</td></tr>)}</tbody></table></div>
    </section>}
  </main>
}
