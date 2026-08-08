import { useEffect, useState } from 'react'
import { ArrowLeft, LoaderCircle, Play, Save, Search, Upload } from 'lucide-react'
import type { SkuQueryConfig, SkuQueryReport, WarehouseState } from './types'

const defaultModels = {
  ark: 'doubao-seed-2-1-turbo-260628',
  siliconflow: 'Qwen/Qwen3.6-35B-A3B',
  dashscope: 'qwen3-vl-plus',
} as const

const defaultConfig: SkuQueryConfig = { max_boxes: 1, dino_fallback: false, dino_confidence_threshold: 0.72 }

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

export function SkuQueryPage({ state, onBack }: { state: WarehouseState; onBack: () => void }) {
  const [imageData, setImageData] = useState('')
  const [imageName, setImageName] = useState('')
  const [query, setQuery] = useState(state.skus[0]?.sku ?? '')
  const [provider, setProvider] = useState<keyof typeof defaultModels>('ark')
  const [model, setModel] = useState<string>(defaultModels.ark)
  const [config, setConfig] = useState<SkuQueryConfig>(defaultConfig)
  const [report, setReport] = useState<SkuQueryReport | null>(null)
  const [notice, setNotice] = useState('')
  const [running, setRunning] = useState(false)

  useEffect(() => {
    void request<SkuQueryConfig>('/api/sku-query/config').then(setConfig).catch((error) => setNotice(error instanceof Error ? error.message : '无法读取查询配置'))
  }, [])

  const saveConfig = async () => {
    try {
      const result = await request<{ sku_query: SkuQueryConfig }>('/api/sku-query/config', { method: 'PUT', body: JSON.stringify(config) })
      setConfig(result.sku_query); setNotice('货物查询配置已保存')
    } catch (error) { setNotice(error instanceof Error ? error.message : '保存配置失败') }
  }

  const run = async () => {
    if (!imageData) { setNotice('请先选择局部货架图片'); return }
    if (!query.trim()) { setNotice('请输入固定位置 ID 或 SKU'); return }
    setRunning(true); setNotice('正在调用 VLM 查询目标商品')
    try {
      const result = await request<{ report: SkuQueryReport }>('/api/sku-query', { method: 'POST', body: JSON.stringify({ image_data: imageData, query: query.trim(), provider, model, config }) })
      setReport(result.report)
      setNotice('查询完成，识别到 ' + result.report.detected_boxes.length + ' 个目标商品')
    } catch (error) { setNotice(error instanceof Error ? error.message : '货物查询失败') } finally { setRunning(false) }
  }

  const artifactUrl = (name: string) => '/api/sku-query/runs/' + encodeURIComponent(report!.run_id) + '/artifact/' + encodeURIComponent(name)
  const resultImage = report?.artifacts.annotated_matches

  return <main className="inspection-page">
    <header className="manual-header"><button className="back-button" onClick={onBack}><ArrowLeft size={16} />返回管理</button><div><span>货物查询</span><small>根据 SKU 参考图，在局部货架照片中定位商品</small></div></header>
    {notice && <div className="manual-notice">{notice}</div>}
    <section className="inspection-workspace">
      <aside className="inspection-tools">
        <div className="tool-title"><Search size={16} />查询条件</div>
        <label className="upload-button"><Upload size={16} /><span>{imageName || '上传局部货架图片'}</span><input type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => { const file = event.target.files?.[0]; if (file) void readFile(file).then((data) => { setImageData(data); setImageName(file.name); setReport(null) }).catch((error) => setNotice(error.message)) }} /></label>
        <div className="tool-divider" />
        <label className="inspection-field"><span>固定位置 ID 或 SKU</span><input list="sku-query-options" value={query} placeholder="例如 1-0-2-43 或 雪碧" onChange={(event) => setQuery(event.target.value)} /></label>
        <datalist id="sku-query-options">{state.slots.map((slot) => <option key={slot.slot_id} value={slot.slot_id}>{slot.expected_sku}</option>)}{state.skus.map((item) => <option key={'sku-' + item.sku} value={item.sku} />)}</datalist>
        <label className="inspection-field"><span>来源</span><select value={provider} onChange={(event) => { const next = event.target.value as keyof typeof defaultModels; setProvider(next); setModel(defaultModels[next]) }}><option value="ark">Ark</option><option value="siliconflow">SiliconFlow</option><option value="dashscope">DashScope</option></select></label>
        <label className="inspection-field"><span>模型</span><input value={model} onChange={(event) => setModel(event.target.value)} /></label>
        <label className="inspection-field"><span>最多输出框 {config.max_boxes}</span><input type="range" min="1" max="20" step="1" value={config.max_boxes} onChange={(event) => setConfig({ ...config, max_boxes: Number(event.target.value) })} /></label>
        <label className="empty-position-toggle"><input type="checkbox" checked={config.dino_fallback} onChange={(event) => setConfig({ ...config, dino_fallback: event.target.checked })} />DINO 保底验证</label>
        {config.dino_fallback && <label className="inspection-field"><span>DINO 置信度 {config.dino_confidence_threshold.toFixed(2)}</span><input type="range" min="0" max="1" step="0.01" value={config.dino_confidence_threshold} onChange={(event) => setConfig({ ...config, dino_confidence_threshold: Number(event.target.value) })} /></label>}
        <button className="secondary-button" onClick={() => void saveConfig()}><Save size={15} />保存配置</button>
        <button className="ai-grounding-button" disabled={running || !imageData || !query.trim() || !model.trim()} onClick={() => void run()}>{running ? <LoaderCircle size={16} className="spin" /> : <Play size={16} />}{running ? '正在查询' : '运行查询'}</button>
      </aside>
      <section className="inspection-results">{resultImage ? <div className="inspection-image-frame"><img className="inspection-source-image" src={artifactUrl(resultImage)} alt="货物查询标注结果" /></div> : imageData ? <div className="inspection-image-frame"><img className="inspection-source-image" src={imageData} alt={imageName || '待查询货架图片'} /></div> : <div className="inspection-empty"><Search size={28} /><strong>上传图片后查询货物</strong></div>}</section>
    </section>
    {report && <section className="review-band sku-query-review">
      <div className="sku-query-summary">
        <img src={'/api/item-images/' + encodeURIComponent(report.reference_slot_id) + '/0.png'} alt={report.sku + ' 参考图'} />
        <div><span>查询结果</span><strong>{report.sku}</strong><small>参考位置 {report.reference_slot_id} / VLM {report.vlm_detected_boxes.length} 个 / 最终 {report.detected_boxes.length} 个</small></div>
        <div className="sku-query-metrics"><span>来源 {report.provider}</span><span>模型 {report.model}</span><span>{report.config.dino_fallback ? 'DINO 已验证' : '仅 VLM'}</span><span>请求 {report.request_seconds.toFixed(2)} 秒</span><span>总计 {report.total_seconds.toFixed(2)} 秒</span></div>
      </div>
      <pre className="sku-query-raw" aria-label="VLM 原始输出">{report.raw_response || 'NONE'}</pre>
    </section>}
  </main>
}
