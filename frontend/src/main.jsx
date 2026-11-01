import React, { useEffect, useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './style.css'

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
const QUICK_ITEMS = ['clean water', 'clean cloth', 'gauze', 'bandage', 'ice pack', 'soap']
const STAGES = ['Reading the image', 'Running the EfficientNet-B0 classifier', 'Retrieving first-aid facts', 'Writing your guidance']

function Section({ id, title, intro, children }) {
  return <section id={id} className="block"><h2>{title}</h2>{intro && <p className="intro">{intro}</p>}{children}</section>
}

function GuideList({ title, tone, mark, items, citations = [] }) {
  if (!items?.length) return null
  return <div className={`gl ${tone}`}><h3>{title}</h3><ul>{items.map((t, i) => <li key={i}><span className="mark" aria-hidden>{mark}</span><span className="guide-copy">{t}{citations?.[i]?.length > 0 && <span className="claim-sources"><SourceLinks sources={citations[i]} label="Source" /></span>}</span></li>)}</ul></div>
}

function SourceLinks({ sources, label = 'Source' }) {
  const valid = (sources || []).filter(source => source?.url && /^https?:\/\//i.test(source.url))
  if (!valid.length) return <span className="source-missing">No item-specific source link was returned by the backend.</span>
  return <span className="inline-sources"><span className="source-label">{label}{valid.length > 1 ? 's' : ''}: </span>{valid.map((source, i) => <React.Fragment key={`${source.url}-${i}`}><a href={source.url} target="_blank" rel="noopener noreferrer">{source.name || source.title || source.url}</a>{i < valid.length - 1 ? <span className="source-separator"> · </span> : null}</React.Fragment>)}</span>
}

function Rejected({ result }) {
  return <div className="result">
    <div className="alert amber"><strong>{result.reason === 'not_injury' ? 'No supported injury found in this photo.' : 'The model could not recognise an injury clearly.'}</strong> {result.message}</div>
    <GuideList title="For a better photo" tone="do" mark="✓" items={[
      'Fill the frame with the injured skin, about 15–30 cm away.',
      'Use bright, even light and keep the camera steady.',
      'Show one injury only, with no filters, text or other objects.',
      'For severe, worsening or uncertain injuries, seek professional care.'
    ]} />
    <p className="small muted">WoundGuide provides guidance only for cuts, burns and bruises on the skin.</p>
  </div>
}

function Result({ result }) {
  if (result.rejected) return <Rejected result={result} />
  const { prediction: p, structured: s } = result
  const probs = Object.entries(p.probabilities).sort((a, b) => b[1] - a[1])
  return <div className="result">
    <div className="verdict">
      <div className="ring" style={{ '--v': p.confidence }}><span>{Math.round(p.confidence)}%</span></div>
      <div><p className="muted">The model's best match</p><h3 className="injury">{p.injury.replaceAll('_', ' ')}</h3><p className="muted small">Not a diagnosis. Confidence reflects the model's certainty, not medical accuracy.</p></div>
      <div className="probs">{probs.map(([name, v]) => <div key={name} className="prob"><span>{name.replaceAll('_', ' ')}</span><div className="track"><i style={{ width: `${v}%` }} /></div><b>{v}%</b></div>)}</div>
    </div>
    {result.low_confidence && <div className="alert amber"><strong>Low confidence.</strong> Retake the photo in good light and closer to the injury, or seek professional advice when concerned.</div>}
    {s?.see_doctor_now && <div className="alert red"><strong>Seek professional care.</strong> The guidance suggests this injury may need urgent medical attention.</div>}
    {s && <>
      {s.summary && <div className="summary-block"><p className="summary">{s.summary}</p>{s.citations?.summary?.length > 0 && <div className="claim-sources summary-sources"><SourceLinks sources={s.citations.summary} label="Source" /></div>}</div>}
      <div className="gl-grid"><GuideList title="What to do" tone="do" mark="✓" items={s.do} citations={s.citations?.do} /><GuideList title="What not to do" tone="dont" mark="✕" items={s.dont} citations={s.citations?.dont} /></div>
      {s.items?.length > 0 && <div className="gl items"><h3>Using the items you have</h3><div className="itemrows">{s.items.map((it, i) => <div className="itemrow" key={`${it.item}-${i}`}><strong>{it.item}</strong><div className="item-detail"><span>{it.how_to_use}</span><div className="item-source"><SourceLinks sources={it.sources} label="Source" /></div></div></div>)}</div></div>}
      <div className="gl-grid"><GuideList title="Good signs" tone="good" mark="+" items={s.good_signs} citations={s.citations?.good_signs} /><GuideList title="Warning signs – see a doctor" tone="warn" mark="!" items={s.warning_signs} citations={s.citations?.warning_signs} /></div>
    </>}
    {result.sources?.length > 0 && <div className="references"><h3>Other references used for the overall guidance</h3><p className="small muted">These are additional retrieved references. When the model supplies a valid source ID for an individual instruction, its clickable link appears directly beneath that instruction.</p><ul>{result.sources.filter(source => source?.url && /^https?:\/\//i.test(source.url)).map((source, i) => <li key={`${source.url}-${i}`}><a href={source.url} target="_blank" rel="noopener noreferrer">{source.name || source.title || source.url}</a></li>)}</ul>{result.guidance_source === 'knowledge_base' ? <p className="small muted">Guidance shown directly from the knowledge base.</p> : <p className="small muted">Guidance phrased by {result.provider || 'the configured language model'} using retrieved evidence.</p>}</div>}
  </div>
}

function App() {
  const [file, setFile] = useState(null)
  const [items, setItems] = useState('')
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [stage, setStage] = useState(0)
  const [error, setError] = useState('')
  const [drag, setDrag] = useState(false)
  const resultRef = useRef(null)
  const preview = useMemo(() => file ? URL.createObjectURL(file) : null, [file])

  useEffect(() => { if (!busy) return; const t = setInterval(() => setStage(s => Math.min(s + 1, STAGES.length - 1)), 1800); return () => clearInterval(t) }, [busy])

  const chosen = items.split(',').map(x => x.trim().toLowerCase()).filter(Boolean)
  function toggleItem(name) { const next = chosen.includes(name) ? chosen.filter(x => x !== name) : [...chosen, name]; setItems(next.join(', ')) }
  function pick(f) { if (f && f.type.startsWith('image/')) { setFile(f); setError(''); setResult(null) } else if (f) setError('Please choose an image file (JPG, PNG or WEBP).') }

  async function analyze() {
    if (!file) { setError('Please add a photo first.'); return }
    setBusy(true); setStage(0); setError(''); setResult(null)
    const form = new FormData(); form.append('image', file); form.append('items', items)
    try {
      const response = await fetch(`${API}/api/analyze`, { method: 'POST', body: form })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Analysis failed')
      setResult(data)
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100)
    } catch (e) {
      setError(e.message === 'Failed to fetch' ? `Cannot reach the backend. Check that it is running on ${API}` : e.message)
    } finally { setBusy(false) }
  }

  return <>
    <nav className="nav"><a href="#top" className="brand">WoundGuide</a><div><a href="#analyze">Analyze</a><a href="#how">How it works</a><a href="#model">The model</a><a href="#safety">Safety</a></div></nav>
    <header id="top" className="hero"><h1>Know what to do in the first minutes after an injury.</h1><p>Upload a photo of a cut, burn or bruise, tell us what you have at hand, and get clear first-aid steps drawn from a knowledge base.</p><a className="cta" href="#analyze">Start an analysis</a></header>
    <main className="wrap">
      <Section id="analyze" title="Analyze an injury" intro="Add a photo and optionally list the items you have.">
        <div className="panel">
          <div className="col"><h3>1. Add a photo</h3><label className={`drop ${drag ? 'drag' : ''}`} onDragOver={e => { e.preventDefault(); setDrag(true) }} onDragLeave={() => setDrag(false)} onDrop={e => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files?.[0]) }}><input type="file" accept="image/*" capture="environment" onChange={e => pick(e.target.files?.[0])} />{preview ? <img src={preview} alt="Selected image" /> : <><strong>Drop a photo here, or tap to choose</strong><span>JPG, PNG or WEBP, up to 10 MB. On a phone this opens the camera.</span></>}</label>{file && <p className="small muted">{file.name} · <button className="link" onClick={() => { setFile(null); setResult(null) }}>remove</button></p>}</div>
          <div className="col"><h3>2. What do you have?</h3><div className="chips">{QUICK_ITEMS.map(n => <button key={n} type="button" className={chosen.includes(n) ? 'on' : ''} onClick={() => toggleItem(n)}>{n}</button>)}</div><textarea value={items} onChange={e => setItems(e.target.value)} placeholder="Or type your own, separated by commas" /><button className="primary" onClick={analyze} disabled={busy}>{busy ? 'Analyzing…' : 'Get first-aid guidance'}</button>{busy && <ol className="stages">{STAGES.map((t, i) => <li key={t} className={i < stage ? 'done' : i === stage ? 'now' : ''}>{t}</li>)}</ol>}{error && <div className="alert red" role="alert">{error}</div>}</div>
        </div>
        <div ref={resultRef} className="anchor">{result && <Result result={result} />}</div>
      </Section>
      <Section id="how" title="How it works" intro="Three stages, each doing one job."><div className="steps"><div><h4>Classify</h4><p>EfficientNet-B0 predicts bruise, burn, cut, no injury, or an unrelated/unknown image.</p></div><div><h4>Retrieve</h4><p>The injury category and available items are matched to the first-aid knowledge base.</p></div><div><h4>Explain</h4><p>The language model phrases only the retrieved facts into short steps. If it fails, the knowledge-base fallback is used.</p></div></div></Section>
      <Section id="model" title="The model" intro="Transfer learning from ImageNet followed by fine-tuning the last feature blocks."><div className="stats"><div><b>409</b><span>images across five classes in the training run</span></div><div><b>93.55%</b><span>test accuracy in the recorded run (62 test images)</span></div><div><b>93.44%</b><span>validation accuracy in the recorded run</span></div><div><b>5</b><span>model classes; three supported injury types</span></div></div><p className="small muted">Metrics come from one training split and are not a guarantee on new images.</p></Section>
      <Section id="safety" title="Safety"><div className="alert amber">This is an educational college project, not a diagnostic tool. It provides guidance only for bruises, burns and cuts. Normal or unrelated images are rejected. Seek professional care for heavy bleeding, large or deep burns, head injuries, or if you are unsure.</div></Section>
    </main>
    <footer>WoundGuide · College ML project · Guidance provider: {result?.provider || 'configured on backend'}</footer>
  </>
}

createRoot(document.getElementById('root')).render(<App />)
