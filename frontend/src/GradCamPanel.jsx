import { useEffect, useState } from 'react'

const API = import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

export default function GradCamPanel({ analysisId, token, originalImage }) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [images, setImages] = useState(null)
  useEffect(() => () => { if (images) URL.revokeObjectURL(images) }, [images])
  async function explain() {
    setLoading(true); setError('')
    try {
      const response = await fetch(`${API}/api/analyses/${analysisId}/gradcam`, {headers:{Authorization:`Bearer ${token}`}})
      if (!response.ok) throw new Error()
      setImages(URL.createObjectURL(await response.blob()))
    } catch { setError('AI model explanation is currently unavailable.') }
    finally { setLoading(false) }
  }
  return <section className="panel gradcam-panel">
    <div className="panel-head"><div><p className="eyebrow">AI MODEL EXPLANATION</p><h3>Grad-CAM Visualization</h3></div></div>
    {!images && <button className="secondary" disabled={loading} onClick={explain}>{loading ? 'Generating explanation…' : 'Show AI Model Explanation'}</button>}
    {error && <p className="gradcam-error" role="status">{error}</p>}
    {loading && <p className="subtle" role="status">Generating a visualization from the saved image…</p>}
    {images && <div className="gradcam-images"><figure><img src={originalImage} alt="Original analyzed crop image"/><figcaption>Original Image</figcaption></figure><figure><img src={images} alt="Grad-CAM visualization over the crop image"/><figcaption>Grad-CAM</figcaption></figure></div>}
    <p className="gradcam-note">Grad-CAM highlights image regions that contributed to this model prediction. This is a qualitative explanation, not an exact disease boundary or damage measurement. For healthy predictions, highlighted areas relate to the healthy-class prediction.</p>
  </section>
}
