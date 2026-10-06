import './weather.css'

export default function WeatherCard({weather}) {
  if (!weather?.available) return <section className="panel weather-card"><p className="eyebrow">ENVIRONMENTAL CONTEXT</p><p className="weather-unavailable">Weather information currently unavailable.</p><p className="subtle">This does not affect crop image analysis.</p></section>
  const observed = weather.observed_at ? new Date(weather.observed_at).toLocaleString() : 'Not provided'
  return <section className="panel weather-card"><p className="eyebrow">ENVIRONMENTAL CONTEXT</p><div className="weather-grid">
    <WeatherValue label="Location" value={weather.location}/>
    <WeatherValue label="Temperature" value={`${weather.temperature_c}°C`}/>
    <WeatherValue label="Humidity" value={`${weather.humidity_percent}%`}/>
    <WeatherValue label="Precipitation" value={`${weather.precipitation_mm} mm`}/>
    <WeatherValue label="Wind" value={`${weather.wind_speed_kmh} km/h`}/>
    <WeatherValue label="Condition" value={weather.condition}/>
    <WeatherValue label="Observed" value={observed}/>
  </div><p className="weather-attribution">Weather data: <a href="https://open-meteo.com/" target="_blank" rel="noreferrer">Open-Meteo</a></p><p className="weather-disclaimer">Context only. This data does not establish disease cause, crop loss, eligibility, or compensation.</p></section>
}

function WeatherValue({label,value}) { return <div className="weather-value"><span>{label}</span><b>{value ?? 'Not provided'}</b></div> }
