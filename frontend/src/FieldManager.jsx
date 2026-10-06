import { useState } from 'react'
import './field.css'

const EMPTY = {field_name:'',crop_name:'',variety:'',state:'',district:'',village:'',area:'',area_unit:'acre',latitude:'',longitude:'',sowing_date:'',notes:''}
const FORM_FIELDS = [
  ['field_name','Field name *','text'],['crop_name','Crop *','text'],['variety','Variety','text'],
  ['state','State','text'],['district','District','text'],['village','Village','text'],
  ['area','Area','number'],['latitude','Latitude','number'],['longitude','Longitude','number'],
  ['sowing_date','Sowing date','date'],
]

export default function FieldManager({request, token, fields, analyses, onRefresh, onError}) {
  const [mode, setMode] = useState('list')
  const [selected, setSelected] = useState(null)
  const [form, setForm] = useState(EMPTY)
  const [busy, setBusy] = useState(false)

  function startCreate() { setSelected(null); setForm(EMPTY); setMode('form') }
  function startEdit(field) {
    setSelected(field)
    setForm(Object.fromEntries(Object.keys(EMPTY).map(key => [key, field[key] ?? (key === 'area_unit' ? 'acre' : '')])))
    setMode('form')
  }
  async function showDetails(field) {
    setBusy(true)
    try { setSelected(await (await request(`/api/fields/${field.id}`, token)).json()); setMode('detail') }
    catch (error) { onError(error.message) }
    finally { setBusy(false) }
  }
  async function save(event) {
    event.preventDefault(); setBusy(true)
    const data = Object.fromEntries(new FormData(event.currentTarget))
    for (const key of ['area','latitude','longitude']) data[key] = data[key] === '' ? null : Number(data[key])
    for (const key of ['variety','state','district','village','sowing_date','notes']) data[key] ||= null
    if (data.area === null) data.area_unit = null
    try {
      await request(selected ? `/api/fields/${selected.id}` : '/api/fields', token, {
        method:selected ? 'PUT' : 'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data),
      })
      await onRefresh(); setMode('list')
    } catch (error) { onError(error.message) }
    finally { setBusy(false) }
  }
  async function archive(field) {
    if (!window.confirm(`Archive “${field.field_name}”? Existing analyses and reports will be kept.`)) return
    setBusy(true)
    try { await request(`/api/fields/${field.id}`, token, {method:'DELETE'}); await onRefresh(); setMode('list') }
    catch (error) { onError(error.message) }
    finally { setBusy(false) }
  }

  if (mode === 'form') return <>
    <div className="page-heading"><p className="eyebrow">FIELD MANAGEMENT</p><h1>{selected ? 'Edit field' : 'Add new field'}</h1><p className="subtle">Location details are optional. Only you can view and manage your fields.</p></div>
    <form className="panel field-form" onSubmit={save}>
      <div className="field-form-grid">{FORM_FIELDS.map(([name,label,type]) => <label key={name}>{label}<input name={name} type={type} value={form[name]} required={name.endsWith('_name')} min={type==='number' ? (name==='area' ? '0.000001' : name==='latitude' ? '-90' : '-180') : undefined} max={type==='number' && name==='latitude' ? '90' : type==='number' && name==='longitude' ? '180' : undefined} step={type==='number' ? 'any' : undefined} onChange={e=>setForm({...form,[name]:e.target.value})}/></label>)}
        <label>Area unit<select name="area_unit" value={form.area_unit} onChange={e=>setForm({...form,area_unit:e.target.value})}><option value="acre">Acre</option><option value="hectare">Hectare</option><option value="sq_meter">Square meter</option></select></label>
        <label className="field-notes">Notes<textarea name="notes" maxLength="2000" rows="4" value={form.notes} onChange={e=>setForm({...form,notes:e.target.value})}/></label>
      </div>
      <div className="button-row"><button className="secondary" type="button" onClick={()=>setMode('list')}>Cancel</button><button className="primary" disabled={busy}>{busy?'Saving…':'Save field'}</button></div>
    </form>
  </>

  if (mode === 'detail' && selected) {
    const items = analyses.filter(item=>item.field_id===selected.id)
    const latest = items[0]
    const location = [selected.village,selected.district,selected.state].filter(Boolean).join(', ')
    return <>
      <div className="page-heading"><p className="eyebrow">FIELD DETAILS</p><h1>{selected.field_name}</h1><p className="subtle">{selected.crop_name}{location ? ` · ${location}` : ''}</p></div>
      <section className="panel field-detail"><div className="field-detail-grid">
        <FieldValue label="Crop" value={selected.crop_name}/><FieldValue label="Variety" value={selected.variety}/><FieldValue label="Location" value={location}/>
        <FieldValue label="Area" value={selected.area == null ? null : `${selected.area} ${selected.area_unit || ''}`}/>
        <FieldValue label="Coordinates" value={selected.latitude == null ? null : `${selected.latitude}, ${selected.longitude}`}/>
        <FieldValue label="Sowing date" value={selected.sowing_date}/><FieldValue label="Notes" value={selected.notes}/>
        <FieldValue label="Number of analyses" value={selected.analysis_count ?? items.length}/>
        <FieldValue label="Latest analysis" value={latest ? new Date(latest.created_at).toLocaleString() : null}/>
        <FieldValue label="Latest prediction" value={latest?.prediction}/><FieldValue label="Latest severity" value={latest?.severity}/>
        <FieldValue label="Latest visual damage indicator" value={latest?.estimated_visual_damage_indicator == null ? null : `${latest.estimated_visual_damage_indicator}%`}/>
      </div><div className="button-row"><button className="secondary" onClick={()=>setMode('list')}>Back to fields</button><button className="secondary" onClick={()=>startEdit(selected)}>Edit</button><button className="secondary archive-button" disabled={busy} onClick={()=>archive(selected)}>Archive field</button></div></section>
    </>
  }

  return <>
    <div className="page-heading field-page-heading"><div><p className="eyebrow">YOUR FIELD RECORD</p><h1>My Fields</h1><p className="subtle">Organize crop observations by agricultural field.</p></div><button className="primary" onClick={startCreate}>＋ &nbsp; Add New Field</button></div>
    {!fields.length ? <section className="panel"><EmptyFields onAdd={startCreate}/></section> : <section className="field-list">{fields.map(field=>{
      const count = analyses.filter(item=>item.field_id===field.id).length
      const location = [field.state,field.district].filter(Boolean).join(' · ')
      return <article className="panel field-card" key={field.id}><div className="field-card-copy"><h2>{field.field_name}</h2><p>{field.crop_name}{location ? ` · ${location}` : ''}</p><span>{count} {count===1?'analysis':'analyses'}</span></div><div className="field-card-actions"><button className="text-button" disabled={busy} onClick={()=>showDetails(field)}>View</button><button className="text-button" onClick={()=>startEdit(field)}>Edit</button></div></article>
    })}</section>}
  </>
}

function FieldValue({label,value}) { return <div className="detail"><span>{label}</span><b>{value || 'Not provided'}</b></div> }
function EmptyFields({onAdd}) { return <div className="empty"><b>No fields added</b><p>Create a field to organize future crop analyses.</p><button className="primary" onClick={onAdd}>Add New Field</button></div> }
