"""Optional current-weather context from Open-Meteo, independent of image inference."""
import hashlib
import logging
import math
from urllib.parse import urlencode

import httpx

from backend import database
from backend.config import WEATHER_CACHE_MINUTES, WEATHER_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
PROVIDER = "Open-Meteo"
WEATHER_UNAVAILABLE = {
    "available": False,
    "message": "Weather information currently unavailable.",
}
WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
    55: "Dense drizzle", 56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain", 66: "Light freezing rain",
    67: "Heavy freezing rain", 71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",
    82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


def _request_json(url, params):
    response = httpx.get(url, params=params, timeout=WEATHER_TIMEOUT_SECONDS)
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise ValueError("Provider returned an invalid response")
    return result


def _field_location(field):
    latitude, longitude = field.get("latitude"), field.get("longitude")
    locality = ", ".join(str(value).strip() for value in
                          (field.get("village"), field.get("district"), field.get("state")) if value and str(value).strip())
    if latitude is not None or longitude is not None:
        if latitude is None or longitude is None:
            raise ValueError("Incomplete field coordinates")
        latitude, longitude = float(latitude), float(longitude)
        if not math.isfinite(latitude) or not math.isfinite(longitude) or not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError("Invalid field coordinates")
        # The provider's forecast is grid-based; rounding also avoids sharing
        # unnecessarily precise user coordinates with the external provider.
        return round(latitude, 2), round(longitude, 2), locality or "Approximate field location"
    administrative = ", ".join(str(value).strip() for value in
                                 (field.get("village"), field.get("district"), field.get("state")) if value and str(value).strip())
    if not administrative:
        raise ValueError("Field has no usable location")
    result = _request_json(GEOCODING_URL, {"name": administrative, "count": 1, "language": "en", "format": "json"})
    places = result.get("results")
    if not isinstance(places, list) or not places:
        raise ValueError("Administrative location could not be resolved")
    place = places[0]
    latitude, longitude = float(place["latitude"]), float(place["longitude"])
    if not math.isfinite(latitude) or not math.isfinite(longitude) or not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("Geocoder returned invalid coordinates")
    return round(latitude, 2), round(longitude, 2), locality or str(place.get("name") or "Approximate field location")


def _numeric(values, key, low=None, high=None):
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Provider returned invalid weather measurements")
    if low is not None and value < low or high is not None and value > high:
        raise ValueError("Provider returned out-of-range weather measurements")
    return float(value)


def _normalize(response):
    current = response.get("current")
    if not isinstance(current, dict):
        raise ValueError("Provider omitted current conditions")
    observed_at = current.get("time")
    if not isinstance(observed_at, str) or not observed_at:
        raise ValueError("Provider omitted observation time")
    temperature = _numeric(current, "temperature_2m", -100, 70)
    humidity = _numeric(current, "relative_humidity_2m", 0, 100)
    precipitation = _numeric(current, "precipitation", 0)
    wind = _numeric(current, "wind_speed_10m", 0)
    code = current.get("weather_code")
    if isinstance(code, bool) or not isinstance(code, int) or code not in WEATHER_CODES:
        raise ValueError("Provider returned invalid weather condition")
    if observed_at[-1:] != "Z" and "+" not in observed_at[10:] and "-" not in observed_at[10:]:
        observed_at += "+00:00"  # Request is made in UTC.
    return {"provider": PROVIDER, "temperature_c": temperature, "humidity_percent": humidity,
            "precipitation_mm": precipitation, "wind_speed_kmh": wind,
            "condition": WEATHER_CODES[code], "observed_at": observed_at}


def get_weather(field):
    """Return normalized current conditions, or an explicit unavailable result."""
    try:
        latitude, longitude, display_location = _field_location(field)
        cache_key = hashlib.sha256(f"{latitude:.2f},{longitude:.2f}".encode("ascii")).hexdigest()
        cached = database.get_cached_weather(cache_key, max(0, WEATHER_CACHE_MINUTES) * 60)
        if cached:
            return {"available": True, "location": display_location, **cached}
        response = _request_json(FORECAST_URL, {
            "latitude": latitude, "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
            "temperature_unit": "celsius", "wind_speed_unit": "kmh",
            "precipitation_unit": "mm", "timezone": "UTC",
        })
        normalized = _normalize(response)
        database.save_weather_cache(cache_key, normalized)
        return {"available": True, "location": display_location, **normalized}
    except Exception as exc:
        # Provider internals and response bodies are deliberately not sent to callers.
        logger.info("Weather context unavailable (%s)", type(exc).__name__)
        return dict(WEATHER_UNAVAILABLE)
