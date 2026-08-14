import { useEffect, useState } from 'react'
import { ArrowLeft, Camera, ImageUp, LoaderCircle, Play, Trophy } from 'lucide-react'
import type { RgbdStockoutReport, RgbMisplacementReport } from './types'

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

export function RgbdStockoutPage({ onBack }: { onBack: () => void }) {
  const [samples, setSamples] = useState<string[]>([])
  const [sample, setSample] = useState('')
  const [report, setReport] = useState<RgbdStockoutReport | null>(null)
  const [notice, setNotice] = useState('')
  const [running, setRunning] = useState(false)
  const [round, setRound] = useState<'stockout' | 'misplacement'>('stockout')
  const [misplacementImage, setMisplacementImage] = useState('')
  const [misplacementName, setMisplacementName] = useState('')
  const [misplacementReport, setMisplacementReport] = useState<RgbMisplacementReport | null>(null)

  useEffect(() => {
    void request<{ samples: string[] }>('/api/rgbd-stockout/samples')
      .then((result) => { setSamples(result.samples); setSample(result.samples[0] ?? '') })
      .catch((error) => setNotice(error instanceof Error ? error.message : '无法读取 RGB-D 测试样本'))
  }, [])

  const run = async () => {
    if (!sample) { setNotice('请选择一个 RGB-D 测试样本'); return }
    setRunning(true); setNotice('正在通过深度定位后排商品并识别补货 SKU')
    try {
      const result = await request<{ report: RgbdStockoutReport }>('/api/rgbd-stockout/runs', { method: 'POST', body: JSON.stringify({ sample, debug: true }) })
      setReport(result.report); setNotice('巡检完成，检测到 ' + result.report.candidates.length + ' 个补货候选')
    } catch (error) { setReport(null); setNotice(error instanceof Error ? error.message : 'RGB-D 巡检失败') } finally { setRunning(false) }
  }

  const runMisplacement = async () => {
    if (!misplacementImage) { setNotice('请先上传局部货架 RGB 图片'); return }
    setRunning(true); setNotice('正在识别连续商品序列中的异常摆放')
    try {
      const result = await request<{ report: RgbMisplacementReport }>('/api/rgb-misplacement/runs', { method: 'POST', body: JSON.stringify({ image_data: misplacementImage, debug: true }) })
      setMisplacementReport(result.report); setNotice('识别完成，发现 ' + result.report.candidates.length + ' 个异常摆放')
    } catch (error) { setMisplacementReport(null); setNotice(error instanceof Error ? error.message : '异常摆放识别失败') } finally { setRunning(false) }
  }

  const overlay = report?.run_id && report.artifacts?.result_overlay
    ? '/api/rgbd-stockout/runs/' + encodeURIComponent(report.run_id) + '/artifact/' + encodeURIComponent(report.artifacts.result_overlay)
    : ''
  const sourceLabel = (source: 'dino' | 'qwen' | 'unknown') => source === 'dino' ? 'DINO' : source === 'qwen' ? 'Qwen' : '未确认'
  const misplacementOverlay = misplacementReport?.run_id && misplacementReport.artifacts?.result_overlay
    ? '/api/rgb-misplacement/runs/' + encodeURIComponent(misplacementReport.run_id) + '/artifact/' + encodeURIComponent(misplacementReport.artifacts.result_overlay)
    : ''

  return <main className="rgbd-page">
    <header className="manual-header"><button className="back-button" onClick={onBack}><ArrowLeft size={16} />返回管理</button><div><span>比赛巡检</span><small>{round === 'stockout' ? 'RGB-D 后排缺货识别（测试样本）' : '局部 RGB 异常摆放识别'}</small></div></header>
    {notice && <div className="manual-notice">{notice}</div>}
    <nav className="rgbd-tabs" aria-label="比赛巡检回合"><button className={round === 'stockout' ? 'active' : ''} onClick={() => { setRound('stockout'); setNotice('') }}>RGB-D 缺货</button><button className={round === 'misplacement' ? 'active' : ''} onClick={() => { setRound('misplacement'); setNotice('') }}>异常摆放</button></nav>
    {round === 'stockout' && <>
    <section className="inspection-workspace">
      <aside className="inspection-tools">
        <div className="tool-title"><Camera size={16} />RGB-D 测试样本</div>
        <p className="tool-help">样本由服务器从头部深度相机采集目录读取。</p>
        <label className="inspection-field"><span>样本目录</span><select value={sample} onChange={(event) => { setSample(event.target.value); setReport(null) }}><option value="">请选择样本</option>{samples.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
        <div className="tool-divider" />
        <div className="rgbd-explain"><span>深度定位后排商品</span><span>DINO 排名，Qwen 复核不确定结果</span></div>
        <button className="ai-grounding-button" disabled={running || !sample} onClick={() => void run()}>{running ? <LoaderCircle size={16} className="spin" /> : <Play size={16} />}{running ? '正在识别' : '运行 RGB-D 巡检'}</button>
      </aside>
      <section className="inspection-results">{overlay ? <div className="inspection-image-frame"><img className="inspection-source-image" src={overlay} alt="RGB-D 缺货识别结果" /></div> : <div className="inspection-empty"><Trophy size={28} /><strong>选择样本后运行巡检</strong></div>}</section>
    </section>
    {report && <section className="review-band rgbd-review"><div className="review-heading"><div><span>补货候选</span><strong>{report.sample}</strong></div></div><div className="review-table-wrap"><table className="review-table rgbd-table"><thead><tr><th>层 / 商品组</th><th>补货 SKU</th><th>后退距离</th><th>判定来源</th><th>DINO 候选</th></tr></thead><tbody>{report.candidates.map((item) => <tr key={item.shelf_index + '-' + item.group_index}><td><code>S{item.shelf_index} G{item.group_index}</code></td><td>{item.sku ?? '未确认'}</td><td>+{item.setback_mm.toFixed(1)} mm</td><td><span className={'rgbd-source rgbd-source-' + item.source}>{sourceLabel(item.source)}</span></td><td>{item.dino_matches.map((match) => match.sku).join(' / ') || '-'}</td></tr>)}</tbody></table></div>{report.skipped_shelves.length > 0 && <div className="rgbd-skipped">跳过层位：{report.skipped_shelves.map((item) => item.reason).join('；')}</div>}</section>}
    </>}
    {round === 'misplacement' && <>
    <section className="inspection-workspace">
      <aside className="inspection-tools">
        <div className="tool-title"><ImageUp size={16} />局部 RGB 图片</div>
        <label className="upload-button"><ImageUp size={16} /><span>{misplacementName || '上传局部货架图片'}</span><input type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => { const file = event.target.files?.[0]; if (file) void readFile(file).then((data) => { setMisplacementImage(data); setMisplacementName(file.name); setMisplacementReport(null) }).catch((error) => setNotice(error.message)) }} /></label>
        <div className="tool-divider" />
        <div className="rgbd-explain"><span>完整图片检测连续序列中的明显异类</span><span>多数图片没有异常，最多输出两个</span></div>
        <button className="ai-grounding-button" disabled={running || !misplacementImage} onClick={() => void runMisplacement()}>{running ? <LoaderCircle size={16} className="spin" /> : <Play size={16} />}{running ? '正在识别' : '运行异常摆放识别'}</button>
      </aside>
      <section className="inspection-results">{misplacementOverlay ? <div className="inspection-image-frame"><img className="inspection-source-image" src={misplacementOverlay} alt="异常摆放识别结果" /></div> : misplacementImage ? <div className="inspection-image-frame"><img className="inspection-source-image" src={misplacementImage} alt={misplacementName || '待识别货架图片'} /></div> : <div className="inspection-empty"><Trophy size={28} /><strong>上传图片后运行识别</strong></div>}</section>
    </section>
    {misplacementReport && <section className="review-band rgbd-review"><div className="review-heading"><div><span>异常摆放</span><strong>{misplacementName || '局部 RGB 图片'}</strong></div></div><div className="review-table-wrap"><table className="review-table rgbd-table"><thead><tr><th>异常商品</th><th>当前 SKU</th><th>置信度</th><th>判定来源</th><th>原因</th><th>DINO 候选</th></tr></thead><tbody>{misplacementReport.candidates.map((item, index) => <tr key={index}><td><code>x {item.box.x}, y {item.box.y}</code></td><td>{item.current_sku ?? '未确认'}</td><td>{item.confidence.toFixed(2)}</td><td><span className={'rgbd-source rgbd-source-' + item.source}>{sourceLabel(item.source)}</span></td><td>{item.reason}</td><td>{item.dino_matches.map((match) => match.sku).join(' / ') || '-'}</td></tr>)}</tbody></table></div>{misplacementReport.candidates.length === 0 && <div className="rgbd-skipped">未发现明显异常摆放。</div>}</section>}
    </>}
  </main>
}
