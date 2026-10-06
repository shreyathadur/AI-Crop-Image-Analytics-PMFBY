export default function HistoryPage({analyses, fields, historyField, setHistoryField, onReport, onView}) {
  const visible = analyses.filter(item => historyField === 'all'
    || (historyField === 'unassigned' ? !item.field_id : item.field_id === historyField))
  return <>
    <div className="page-heading"><p className="eyebrow">YOUR FIELD RECORD</p><h1>Analysis history</h1><p className="subtle">Every observation you've saved in one place.</p></div>
    <section className="panel history-panel">
      <div className="panel-head"><div><h3>All analyses</h3><p>{visible.length} saved {visible.length === 1 ? 'observation' : 'observations'}</p></div>
        <label className="history-filter">Filter by field<select value={historyField} onChange={event=>setHistoryField(event.target.value)}>
          <option value="all">All Fields</option><option value="unassigned">No Field / Unassigned</option>
          {fields.map(field=><option key={field.id} value={field.id}>{field.field_name}</option>)}
        </select></label>
      </div>
      {visible.map(item=><AnalysisRow key={item.id} item={item} onReport={onReport} onView={onView}/>)}
      {!visible.length && <div className="empty"><b>{analyses.length ? 'No analyses for this field' : 'No analyses recorded'}</b><p>{analyses.length ? 'Choose another field filter to see more observations.' : 'Upload a crop image to start your field record.'}</p></div>}
    </section>
  </>
}

function AnalysisRow({item,onReport,onView}) {
  return <div className="analysis-row"><div className="row-icon">{item.health_status==='Healthy'?'✓':'⌁'}</div>
    <div className="row-main"><b>{item.crop || 'Crop image'}</b><span>{item.prediction}</span></div>
    <span className={`row-status ${item.health_status==='Healthy'?'ok':''}`}>{item.health_status==='Healthy'?'Healthy':'Affected'}</span>
    <span className="row-date">{new Date(item.created_at).toLocaleDateString()}</span>
    {item.field?.field_name && <span className="row-field">{item.field.field_name}</span>}
    {item.weather?.available && <span className="row-weather">Weather available</span>}
    <button className="text-button" onClick={()=>onView(item)}>View</button>
    <button className="icon-button" title="Download report" onClick={()=>onReport(item.id)}>↓</button>
  </div>
}
