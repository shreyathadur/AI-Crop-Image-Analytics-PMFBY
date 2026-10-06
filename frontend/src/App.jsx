import { lazy, Suspense, useEffect, useMemo, useState } from 'react'
import FieldManager from './FieldManager.jsx'
import WeatherCard from './WeatherCard.jsx'
import HistoryPage from './HistoryPage.jsx'
import GradCamPanel from './GradCamPanel.jsx'

const HealthChart = lazy(() => import('./HealthChart.jsx'))

const API = import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
const SEVERITY_NOTE = 'Severity is an AI-assisted visual estimate based on visible affected regions in the uploaded image. It is not an official PMFBY loss percentage.'
const DISCLAIMER = 'AI-assisted preliminary assessment — not an official PMFBY claim settlement or government decision.'

async function request(path, token, options = {}) {
  const headers = { ...(options.headers || {}) }
  if (token) headers.Authorization = `Bearer ${token}`
  const response = await fetch(`${API}${path}`, { ...options, headers })
  if (!response.ok) {
    let message = 'Request failed. Please try again.'
    try { message = (await response.json()).detail || message } catch {}
    throw new Error(message)
  }
  return response.status === 204 ? null : response
}

export default function App() {
  const [token, setToken] = useState(localStorage.getItem('crop_token') || '')
  const [user, setUser] = useState(JSON.parse(localStorage.getItem('crop_user') || 'null'))
  const [page, setPage] = useState('dashboard')
  const [analyses, setAnalyses] = useState([])
  const [fields, setFields] = useState([])
  const [selectedField, setSelectedField] = useState('')
  const [historyField, setHistoryField] = useState('all')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [result, setResult] = useState(null)
  const [reportItem, setReportItem] = useState(null)
  const initialAuthMode = location.pathname === '/forgot-password' ? 'forgot' : location.pathname === '/reset-password' ? 'reset' : 'login'
  const [authMode, setAuthMode] = useState(initialAuthMode)

  async function refresh() {
    if (!token) return
    try { setAnalyses(await (await request('/api/analyses', token)).json()) }
    catch (e) { setError(e.message) }
  }
  async function refreshFields() {
    if (!token) return
    try { setFields(await (await request('/api/fields', token)).json()) }
    catch (e) { setError(e.message) }
  }
  useEffect(() => { refresh(); refreshFields() }, [token])
  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview])

  async function authenticate(event) {
    event.preventDefault(); setError(''); setBusy(true)
    const form = new FormData(event.currentTarget)
    const body = { email: form.get('email'), password: form.get('password') }
    try {
      if (authMode === 'register') await request('/api/auth/register', '', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) })
      const response = await request('/api/auth/login', '', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) })
      const data = await response.json(); localStorage.setItem('crop_token', data.access_token); localStorage.setItem('crop_user', JSON.stringify(data.user)); setToken(data.access_token); setUser(data.user)
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }
  async function requestPasswordReset(event) {
    event.preventDefault(); setError(''); setBusy(true)
    try {
      const form = new FormData(event.currentTarget)
      const response = await request('/api/auth/forgot-password', '', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({email:form.get('email')})})
      const data = await response.json()
      setError('')
      setResetMessage(data.message + (data.development_only ? ' DEVELOPMENT ONLY: use the reset link shown by the local API.' : ''))
      if (data.reset_url) setDevelopmentResetUrl(data.reset_url)
    } catch(e) { setError(e.message) } finally { setBusy(false) }
  }
  async function submitNewPassword(event) {
    event.preventDefault(); setError(''); setBusy(true)
    const form = new FormData(event.currentTarget)
    if (form.get('password') !== form.get('confirmPassword')) { setError('Passwords do not match.'); setBusy(false); return }
    try {
      const response = await request('/api/auth/reset-password', '', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token:new URLSearchParams(location.search).get('token') || '',new_password:form.get('password')})})
      setResetMessage((await response.json()).message); setAuthMode('resetDone')
    } catch(e) { setError(e.message) } finally { setBusy(false) }
  }
  const [resetMessage, setResetMessage] = useState('')
  const [developmentResetUrl, setDevelopmentResetUrl] = useState('')
  function chooseImage(next) {
    setError(''); setResult(null); setFile(next || null)
    if (preview) URL.revokeObjectURL(preview)
    setPreview(next ? URL.createObjectURL(next) : '')
  }
  async function analyze(event) {
    event.preventDefault(); if (!file) return
    setBusy(true); setError('')
    try {
      const body = new FormData(); body.append('image', file)
      if (selectedField) body.append('field_id', selectedField)
      const data = await (await request('/api/analyze', token, { method:'POST', body })).json()
      setResult(data); setPage('result'); await refresh()
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }
  async function downloadReport(id) {
    try {
      const response = await request(`/api/report/${id}`, token)
      const blob = await response.blob(); const url = URL.createObjectURL(blob); const link = document.createElement('a'); link.href = url; link.download = `crop-assessment-${id}.pdf`; link.click(); URL.revokeObjectURL(url)
    } catch (e) { setError(e.message) }
  }
  function logout() { localStorage.removeItem('crop_token'); localStorage.removeItem('crop_user'); setToken(''); setUser(null); setAnalyses([]); setFields([]); setPage('dashboard') }
  function openReport(item) { setReportItem(item); setPage('report') }

  const chartData = useMemo(() => ['Healthy', 'Potential disease class detected'].map(status => ({name: status === 'Healthy' ? 'Healthy' : 'Affected', count: analyses.filter(a => a.health_status === status).length})), [analyses])
  if (!token) return <main className="auth-shell"><div className="auth-art"><div className="brand light"><span className="brand-mark">✳</span> CropLens</div><div><p className="eyebrow">FIELD NOTES, MADE CLEAR</p><h1>See the signal<br/>in every leaf.</h1><p className="muted-light">A careful first look at crop health, built for the people closest to the field.</p></div><span className="art-caption">AI assisted · Human reviewed</span></div><form className="auth-card" onSubmit={authMode === 'forgot' ? requestPasswordReset : authMode === 'reset' ? submitNewPassword : authMode === 'resetDone' ? e => e.preventDefault() : authenticate}><div className="brand"><span className="brand-mark">✳</span> CropLens</div><p className="eyebrow">YOUR FIELD WORKSPACE</p><h2>{authMode === 'forgot' ? 'Forgot password' : authMode === 'reset' ? 'Reset password' : authMode === 'resetDone' ? 'Password updated' : authMode === 'login' ? 'Welcome back' : 'Create your account'}</h2><p className="subtle">{authMode === 'forgot' ? 'Enter your registered email address.' : authMode === 'reset' ? 'Choose a new password for your account.' : 'Sign in to review your crop observations.'}</p>{error && <div className="error">{error}</div>}{resetMessage && <div className="notice"><p>{resetMessage}</p></div>}{developmentResetUrl && <p className="development-only"><b>DEVELOPMENT ONLY</b><br/><a href={developmentResetUrl}>Open local reset link</a></p>}{['login','register','forgot'].includes(authMode) && <label>Email address<input name="email" type="email" autoComplete="email" placeholder="you@example.com" required/></label>}{['login','register'].includes(authMode) && <label>Password<input name="password" type="password" minLength={authMode === 'register' ? 10 : 1} autoComplete={authMode === 'login' ? 'current-password' : 'new-password'} placeholder="At least 10 characters" required/></label>}{authMode === 'reset' && <><label>New password<input name="password" type="password" minLength="10" autoComplete="new-password" required/></label><label>Confirm new password<input name="confirmPassword" type="password" minLength="10" autoComplete="new-password" required/></label></>}{authMode !== 'resetDone' && <button className="primary wide" disabled={busy}>{busy ? 'Please wait…' : authMode === 'forgot' ? 'Send reset link' : authMode === 'reset' ? 'Reset password' : authMode === 'login' ? 'Sign in' : 'Create account'} <span>→</span></button>}{authMode === 'login' && <p className="switch-auth"><button type="button" onClick={() => {setAuthMode('forgot');setError('');setResetMessage('')}}>Forgot Password?</button></p>}{['login','register','forgot','resetDone'].includes(authMode) && <p className="switch-auth">{authMode === 'login' || authMode === 'forgot' ? 'New to CropLens?' : 'Already have an account?'} <button type="button" onClick={() => {setAuthMode(authMode === 'login' || authMode === 'forgot' || authMode === 'resetDone' ? 'register' : 'login');setError('');setResetMessage('');setDevelopmentResetUrl('')}}>{authMode === 'login' || authMode === 'forgot' || authMode === 'resetDone' ? 'Create an account' : 'Sign in'}</button></p>}{authMode === 'forgot' && <p className="switch-auth"><button type="button" onClick={() => {setAuthMode('login');setError('');setResetMessage('')}}>Back to sign in</button></p>}{authMode === 'reset' && <p className="switch-auth"><button type="button" onClick={() => {setAuthMode('login');setError('')}}>Back to sign in</button></p>}<div className="disclaimer">For preliminary decision support only. Not an official PMFBY claim settlement.</div></form></main>

  return <div className="app-shell"><aside className="sidebar"><div className="brand light"><span className="brand-mark">✳</span> CropLens</div><div className="workspace-label">WORKSPACE</div><nav><button className={page==='dashboard'?'active':''} onClick={()=>setPage('dashboard')}><span>◫</span> Overview</button><button className={page==='analysis'?'active':''} onClick={()=>setPage('analysis')}><span>⊕</span> New analysis</button><button className={page==='history'?'active':''} onClick={()=>setPage('history')}><span>◷</span> Analysis history <small>{analyses.length}</small></button><button className={page==='fields'?'active':''} onClick={()=>setPage('fields')}><span>⌖</span> My Fields <small>{fields.length}</small></button></nav><div className="side-bottom"><div className="profile"><div className="avatar">{user?.email?.[0]?.toUpperCase() || 'F'}</div><div className="profile-copy"><b>{user?.email?.split('@')[0] || 'Field user'}</b><span>{user?.email}</span></div></div><button className="logout" onClick={logout}>↪ &nbsp; Sign out</button></div></aside><main className="main"><header className="topbar"><div><span className="top-kicker">CROP HEALTH WORKSPACE</span><span className="crumb"> / {page === 'dashboard' ? 'Overview' : page === 'analysis' ? 'New analysis' : page === 'history' ? 'History' : page === 'fields' ? 'My Fields' : 'Assessment'}</span></div><div className="online"><i/> Local workspace</div></header><div className="content">{error && <div className="error banner">{error}<button onClick={()=>setError('')}>×</button></div>}
    {page==='fields' && <FieldManager request={request} token={token} fields={fields} analyses={analyses} onRefresh={refreshFields} onError={setError}/>}
    {page==='dashboard' && <><div className="welcome"><div><p className="eyebrow">FIELD OVERVIEW</p><h1>Good work starts with a closer look.</h1><p className="subtle">Review your crop observations and start a new leaf analysis.</p></div><button className="primary" onClick={()=>{setPage('analysis');setError('')}}>＋ &nbsp; New analysis</button></div><section className="stat-grid"><Stat label="Total analyses" value={analyses.length} note="Your recorded observations" icon="◷"/><Stat label="Healthy" value={analyses.filter(a=>a.health_status==='Healthy').length} note="Predicted healthy classes" icon="↗"/><Stat label="Affected" value={analyses.filter(a=>a.health_status!=='Healthy').length} note="Potential disease classes" icon="⌁"/></section><section className="panel-grid"><div className="panel chart-panel"><div className="panel-head"><div><h3>Health observations</h3><p>Based on your saved analyses</p></div><span className="tag">ALL TIME</span></div>{analyses.length ? <div className="chart"><Suspense fallback={<div className="chart-loading">Preparing chart…</div>}><HealthChart data={chartData}/></Suspense></div> : <Empty title="Your chart will grow with your field notes" text="Run your first analysis to see the observations you've saved."/>}</div><div className="panel recent-panel"><div className="panel-head"><div><h3>Recent analyses</h3><p>Your latest field observations</p></div><button className="text-button" onClick={()=>setPage('history')}>View all →</button></div>{analyses.slice(0,4).map(a=><AnalysisRow key={a.id} item={a} onReport={downloadReport} onView={openReport}/>)}{!analyses.length&&<Empty title="Nothing here yet" text="Your saved analyses will appear here."/>}</div></section><div className="notice"><span>ⓘ</span><p><b>Decision support, with care.</b> {DISCLAIMER} Results need review by a qualified agricultural professional.</p></div></>}
    {page==='analysis' && <><div className="page-heading"><p className="eyebrow">NEW OBSERVATION</p><h1>Analyze a crop image</h1><p className="subtle">Upload a clear photo of a single crop leaf for a preliminary classification.</p></div><div className="analysis-layout"><form className="panel upload-panel" onSubmit={analyze}><label className="field-select">Select Field<select value={selectedField} onChange={e=>setSelectedField(e.target.value)}><option value="">No Field / Unassigned</option>{fields.map(field=><option key={field.id} value={field.id}>{field.field_name} · {field.crop_name}</option>)}</select></label><label className="upload-zone" onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();chooseImage(e.dataTransfer.files[0])}}>{preview ? <img className="preview" src={preview} alt="Selected crop leaf preview"/> : <><span className="upload-icon">↑</span><b>Drop a leaf image here</b><span className="subtle">or browse files from your device</span><span className="file-types">JPG, PNG or WEBP · up to 10 MB</span></>}<input type="file" accept="image/jpeg,image/png,image/webp" onChange={e=>chooseImage(e.target.files?.[0])}/></label>{file&&<div className="selected-file"><span>▧ &nbsp; {file.name}</span><button type="button" onClick={()=>chooseImage(null)}>Remove</button></div>}<button className="primary wide" disabled={!file||busy}>{busy?'Analyzing image…':'Run preliminary analysis'} <span>→</span></button><p className="form-note">Images are stored privately with your analysis history.</p></form><aside className="panel guide-panel"><p className="eyebrow">FOR A CLEARER RESULT</p><h3>Before you upload</h3><ul><li>Focus on one leaf and keep it in frame.</li><li>Use natural light and avoid strong shadows.</li><li>Keep the leaf sharp and visible.</li><li>Try to show the affected area clearly.</li></ul><div className="notice compact"><span>ⓘ</span><p>Model predictions can be wrong, especially on field images that differ from its training photos.</p></div><p className="disclaimer">{DISCLAIMER}</p></aside></div></>}
    {page==='result' && result && <><div className="page-heading"><p className="eyebrow">ANALYSIS COMPLETE</p><h1>Preliminary assessment</h1><p className="subtle">A first look at the uploaded leaf image.</p></div><div className="result-layout"><section className="panel result-card"><div className="result-top"><div><span className={`status-pill ${result.health_status==='Healthy'?'healthy':''}`}>{result.health_status}</span><h2>{result.prediction}</h2><p className="subtle">Crop identified: {result.crop}</p></div><div className="confidence"><strong>{(result.confidence*100).toFixed(1)}<small>%</small></strong><span>Model score</span></div></div><div className="result-details"><Detail label="Crop" value={result.crop}/><Detail label="Prediction" value={result.prediction}/><Detail label="Detected condition" value={result.detected_condition || result.prediction}/><Detail label="Health status" value={result.health_status}/><Detail label="Severity" value={result.severity}/><Detail label="Estimated Visual Damage Indicator" value={(result.estimated_visual_damage_indicator ?? result.estimated_damage_indicator)==null?'Not provided':`${result.estimated_visual_damage_indicator ?? result.estimated_damage_indicator}%`}/><Detail label="Severity Method" value={result.severity_method}/><Detail label="Analyzed at" value={new Date(result.created_at).toLocaleString()}/></div><div className="explanation"><b>AI explanation</b><p>{result.explanation}</p><p>{result.severity_explanation || SEVERITY_NOTE}</p></div><div className="button-row"><button className="primary" onClick={()=>downloadReport(result.id)}>↓ &nbsp; Download report</button><button className="secondary" onClick={()=>{setPage('analysis');chooseImage(null)}}>Analyze another</button></div></section><div className="panel image-panel">{preview&&<img src={preview} alt="Analyzed crop leaf"/>}<p className="subtle">{file?.name || result.image_name}</p></div></div><GradCamPanel analysisId={result.id} token={token} originalImage={preview}/><WeatherCard weather={result.weather}/><div className="notice"><span>ⓘ</span><p><b>Preliminary decision support only.</b> {DISCLAIMER} The model does not assess policy eligibility, event cause, or insured loss.</p></div></>}
    {page==='history' && <HistoryPage analyses={analyses} fields={fields} historyField={historyField} setHistoryField={setHistoryField} onReport={downloadReport} onView={openReport}/>}
    {page==='report' && reportItem && <><div className="page-heading"><p className="eyebrow">REPORT VIEW</p><h1>Analysis report</h1><p className="subtle">Review the saved assessment and download its PDF copy.</p></div><section className="panel report-view"><div className="panel-head"><div><h3>{reportItem.crop} · {reportItem.prediction}</h3><p>Analysis ID: {reportItem.id}</p></div><span className="tag">{new Date(reportItem.created_at).toLocaleString()}</span></div><div className="result-details"><Detail label="Health status" value={reportItem.health_status}/><Detail label="Model score" value={`${(reportItem.confidence*100).toFixed(1)}%`}/><Detail label="Severity" value={reportItem.severity}/><Detail label="Estimated Visual Damage Indicator" value={(reportItem.estimated_visual_damage_indicator ?? reportItem.estimated_damage_indicator)==null?'Not provided':`${reportItem.estimated_visual_damage_indicator ?? reportItem.estimated_damage_indicator}%`}/><Detail label="Severity Method" value={reportItem.severity_method}/><Detail label="Image" value={reportItem.image_name}/></div><div className="explanation"><b>AI explanation</b><p>{reportItem.explanation}</p><p>{reportItem.severity_explanation || SEVERITY_NOTE}</p></div><p className="disclaimer">{DISCLAIMER}</p><div className="button-row"><button className="primary" onClick={()=>downloadReport(reportItem.id)}>↓ &nbsp; Download PDF</button><button className="secondary" onClick={()=>setPage('history')}>Back to history</button></div></section></>}
  </div><footer className="footer">CropLens <span>·</span> Preliminary crop image decision support <span>·</span> Not an official PMFBY claim assessment</footer></main></div>
}

function Stat({label,value,note,icon}){return <div className="stat"><div className="stat-head"><span>{label}</span><i>{icon}</i></div><strong>{value}</strong><small>{note}</small></div>}
function Empty({title,text}){return <div className="empty"><span>⌁</span><b>{title}</b><p>{text}</p></div>}
function Detail({label,value}){return <div className="detail"><span>{label}</span><b>{value ?? 'Not available'}</b></div>}
function AnalysisRow({item,onReport,onView}){return <div className="analysis-row"><div className="row-icon">{item.health_status==='Healthy'?'✓':'⌁'}</div><div className="row-main"><b>{item.crop || 'Crop image'}</b><span>{item.prediction}</span></div><span className={`row-status ${item.health_status==='Healthy'?'ok':''}`}>{item.health_status==='Healthy'?'Healthy':'Affected'}</span><span className="row-date">{new Date(item.created_at).toLocaleDateString()}</span>{item.field?.field_name && <span className="row-field">{item.field.field_name}</span>}{item.weather?.available && <span className="row-weather">Weather available</span>}<button className="text-button" onClick={()=>onView?.(item)}>View</button><button className="icon-button" title="Download report" onClick={()=>onReport(item.id)}>↓</button></div>}
