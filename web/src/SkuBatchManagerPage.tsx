import { useMemo, useState } from 'react'
import { ArrowLeft, Grid2X2, ImagePlus, List, LoaderCircle, Plus, Save, Sparkles, Trash2, Upload } from 'lucide-react'
import type { Sku, WarehouseState } from './types'

type Draft = Sku & { original_sku: string; reference_image_data?: string }

async function request<T>(path: string, payload: unknown): Promise<T> {
  const response = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
  const body = await response.json() as T & { error?: string }
  if (!response.ok) throw new Error(body.error ?? '操作失败')
  return body
}

export function SkuBatchManagerPage({ state, onBack, onSaved }: { state: WarehouseState; onBack: () => void; onSaved: (state: WarehouseState) => void }) {
  const [drafts, setDrafts] = useState<Draft[]>(() => state.skus.map((sku) => ({ ...sku, original_sku: sku.sku })))
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [view, setView] = useState<'list' | 'grid'>('list')
  const [newName, setNewName] = useState('')
  const [folder, setFolder] = useState('/home/wxx/桌面/商品裁剪_png')
  const [notice, setNotice] = useState('')
  const [running, setRunning] = useState(false)
  const dirty = useMemo(() => JSON.stringify(drafts.map(({ reference_image_data, ...row }) => row)) !== JSON.stringify(state.skus.map((sku) => ({ ...sku, original_sku: sku.sku }))) || drafts.some((row) => Boolean(row.reference_image_data)), [drafts, state.skus])
  const selectedRows = drafts.filter((row) => selected.has(row.original_sku))

  const update = (original_sku: string, patch: Partial<Draft>) => setDrafts((rows) => rows.map((row) => row.original_sku === original_sku ? { ...row, ...patch } : row))
  const toggle = (original_sku: string) => setSelected((current) => { const next = new Set(current); next.has(original_sku) ? next.delete(original_sku) : next.add(original_sku); return next })
  const upload = (original_sku: string, file?: File) => { if (!file) return; const reader = new FileReader(); reader.onload = () => update(original_sku, { reference_image_data: String(reader.result) }); reader.readAsDataURL(file) }
  const reset = (next: WarehouseState) => { setDrafts(next.skus.map((sku) => ({ ...sku, original_sku: sku.sku }))); setSelected(new Set()); onSaved(next) }

  const create = () => {
    const sku = newName.trim()
    if (!sku || drafts.some((row) => row.sku === sku)) { setNotice('请输入未重复的 SKU 名称'); return }
    setDrafts((rows) => [...rows, { sku, original_sku: sku, category: '', mesh_file: '', tex_file: '', owlv2_prompt: '', qwen_grounding_prompt: '', reference_image_path: '', grasp_method: '夹爪' }])
    setNewName(''); setNotice(`已加入草稿 SKU：${sku}`)
  }
  const save = async () => {
    setRunning(true); setNotice('')
    try { reset((await request<{ state: WarehouseState }>('/api/skus/batch', { skus: drafts })).state); setNotice('已保存全部 SKU 修改') }
    catch (error) { setNotice(error instanceof Error ? error.message : '保存失败') } finally { setRunning(false) }
  }
  const generate = async () => {
    if (selectedRows.length < 2 || selectedRows.length > 8) return
    if (selectedRows.some((row) => row.original_sku !== row.sku || row.reference_image_data)) {
      setNotice('请先保存所选 SKU 的名称和主图修改，再生成对比提示词')
      return
    }
    setRunning(true); setNotice('')
    try {
      const result = await request<{ drafts: Array<{ sku: string; owlv2_prompt: string }>; skipped: Array<{ sku: string; reason: string }> }>('/api/skus/prompts/generate', { skus: selectedRows.map((row) => row.sku) })
      const generated = new Map(result.drafts.map((row) => [row.sku, row.owlv2_prompt]))
      setDrafts((rows) => rows.map((row) => generated.has(row.sku) ? { ...row, owlv2_prompt: generated.get(row.sku)! } : row))
      const skippedText = result.skipped.map((row) => `${row.sku}（${row.reason}）`).join('、')
      setNotice(`已覆盖 ${result.drafts.length} 项提示词${skippedText ? `；跳过 ${skippedText}` : ''}`)
    } catch (error) { setNotice(error instanceof Error ? error.message : '生成失败') } finally { setRunning(false) }
  }
  const importFolder = async () => {
    if (dirty) { setNotice('请先保存或刷新放弃当前草稿，再导入图片文件夹'); return }
    setRunning(true); setNotice('')
    try { const result = await request<{ state: WarehouseState; imported: string[]; skipped: Array<{ file: string; reason: string }> }>('/api/skus/import-image-directory', { directory: folder }); reset(result.state); setNotice(`已导入 ${result.imported.length} 张主图${result.skipped.length ? `；跳过 ${result.skipped.length} 个文件` : ''}`) }
    catch (error) { setNotice(error instanceof Error ? error.message : '导入失败') } finally { setRunning(false) }
  }
  const remove = async () => {
    if (dirty) { setNotice('请先保存或刷新放弃当前草稿，再删除'); return }
    const affected = state.slots.filter((slot) => selectedRows.some((row) => row.sku === slot.expected_sku || row.sku === slot.actual_sku)).length
    if (!window.confirm(`删除 ${selectedRows.length} 个 SKU？${affected} 个固定货位中的引用会替换为 unknown。`)) return
    setRunning(true); setNotice('')
    try { reset((await request<{ state: WarehouseState }>('/api/skus/delete', { skus: selectedRows.map((row) => row.sku) })).state); setNotice('已删除所选 SKU，关联货位已替换为 unknown') }
    catch (error) { setNotice(error instanceof Error ? error.message : '删除失败') } finally { setRunning(false) }
  }
  const fields = (row: Draft) => <><label><span>SKU 名称</span><input value={row.sku} disabled={row.original_sku === 'unknown'} onChange={(event) => update(row.original_sku, { sku: event.target.value })} /></label><label><span>抓取方式</span><select value={row.grasp_method} onChange={(event) => update(row.original_sku, { grasp_method: event.target.value as Draft['grasp_method'] })}><option value="夹爪">夹爪</option><option value="吸盘">吸盘</option></select></label><label className="sku-batch-prompt"><span>OWLv2 提示词</span><textarea value={row.owlv2_prompt} onChange={(event) => update(row.original_sku, { owlv2_prompt: event.target.value })} /></label><label className="sku-batch-prompt"><span>Qwen 专用提示词</span><textarea value={row.qwen_grounding_prompt} onChange={(event) => update(row.original_sku, { qwen_grounding_prompt: event.target.value })} /></label></>
  const image = (row: Draft) => <label className="sku-batch-image">{row.reference_image_data || row.reference_image_path ? <img src={row.reference_image_data || `/api/sku-images/${encodeURIComponent(row.original_sku)}`} alt={`${row.sku} 主图`} /> : <span>暂无主图</span>}<b><ImagePlus size={14} />更换主图</b><input type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => upload(row.original_sku, event.target.files?.[0])} /></label>
  return <main className="sku-batch-page"><header className="sku-batch-header"><button className="back-button" onClick={onBack}><ArrowLeft size={17} />返回仓库</button><div><strong>SKU 批量管理</strong><small>全量 SKU 主图、抓取方式、OWLv2 与 Qwen 提示词</small></div></header>{notice && <div className="manual-notice">{notice}</div>}<section className="sku-batch-toolbar"><span>{drafts.length} 个 SKU{dirty ? ' · 有未保存修改' : ''}</span><small className="sku-prompt-group-hint">选择 2-8 个相近 SKU 生成对比提示词</small><button className="icon-button" title="列表模式" onClick={() => setView('list')}><List size={17} /></button><button className="icon-button" title="大图模式" onClick={() => setView('grid')}><Grid2X2 size={17} /></button><input value={newName} placeholder="新 SKU 名称" onChange={(event) => setNewName(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') create() }} /><button className="secondary-button" onClick={create}><Plus size={15} />新建 SKU</button><input className="sku-folder-input" value={folder} onChange={(event) => setFolder(event.target.value)} /><button className="secondary-button" disabled={running} onClick={() => void importFolder()}><Upload size={15} />导入图片文件夹</button><button className="secondary-button" disabled={running || selectedRows.length < 2 || selectedRows.length > 8} onClick={() => void generate()}><Sparkles size={15} />生成并覆盖提示词</button><button className="danger-outline-button" disabled={running || !selectedRows.length} onClick={() => void remove()}><Trash2 size={15} />删除勾选</button><button className="primary-button" disabled={running || !dirty} onClick={() => void save()}>{running ? <LoaderCircle size={16} className="spin" /> : <Save size={16} />}保存全部修改</button></section>{view === 'list' ? <div className="sku-batch-table-wrap"><table className="sku-batch-table"><thead><tr><th><input type="checkbox" checked={selected.size > 0 && selected.size === drafts.filter((row) => row.original_sku !== 'unknown').length} onChange={() => setSelected(selected.size ? new Set() : new Set(drafts.filter((row) => row.original_sku !== 'unknown').map((row) => row.original_sku)))} /></th><th>主图</th><th>SKU 名称</th><th>抓取方式</th><th>OWLv2 提示词</th><th>Qwen 专用提示词</th></tr></thead><tbody>{drafts.map((row) => <tr key={row.original_sku}><td><input type="checkbox" disabled={row.original_sku === 'unknown'} checked={selected.has(row.original_sku)} onChange={() => toggle(row.original_sku)} /></td><td>{image(row)}</td><td>{fields(row).props.children[0]}</td><td>{fields(row).props.children[1]}</td><td>{fields(row).props.children[2]}</td><td>{fields(row).props.children[3]}</td></tr>)}</tbody></table></div> : <div className="sku-batch-grid">{drafts.map((row) => <article className="sku-batch-card" key={row.original_sku}><label><input type="checkbox" disabled={row.original_sku === 'unknown'} checked={selected.has(row.original_sku)} onChange={() => toggle(row.original_sku)} />勾选</label>{image(row)}{fields(row)}</article>)}</div>}</main>
}
