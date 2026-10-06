# API

The API is served by FastAPI under `/api`; interactive OpenAPI documentation is available at `/docs` during development.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness and model availability |
| POST | `/api/auth/register` | Create account |
| POST | `/api/auth/login` | Authenticate and issue token |
| POST | `/api/fields` | Create an authenticated user's field |
| GET | `/api/fields` | List the current user's active fields |
| GET | `/api/fields/{field_id}` | Read one owned active field and its latest analysis summary |
| PUT | `/api/fields/{field_id}` | Update one owned active field |
| DELETE | `/api/fields/{field_id}` | Archive one owned field; analyses are retained |
| GET | `/api/fields/{field_id}/weather` | Get cached/current environmental context for an owned field |
| POST | `/api/analyze` | Authenticated multipart image analysis |
| GET | `/api/analyses` | Current user's analysis history |
| GET | `/api/analyses/{id}` | Read one owned analysis |
| GET | `/api/analyses/{id}/gradcam` | Generate an owned analysis's Grad-CAM overlay on demand |
| DELETE | `/api/analyses/{id}` | Delete one owned analysis |
| GET | `/api/report/{id}` | Download an owned analysis report |

Analysis responses include prediction, detected condition, crop, health status, model score, timestamp, and a separate `severity`, `estimated_visual_damage_indicator`, `severity_method`, and `severity_explanation`. Healthy classifier outputs receive `Healthy` and 0.0. Other classes use the documented visual color heuristic in [ML_MODEL.md](ML_MODEL.md). These fields are not calibrated to severity labels and do not represent crop loss, a PMFBY percentage, or an official assessment.

`POST /api/fields` and `PUT /api/fields/{field_id}` accept JSON fields `field_name` and `crop_name` (required), plus optional `variety`, `state`, `district`, `village`, paired `latitude`/`longitude`, positive `area`, `area_unit` (`acre`, `hectare`, or `sq_meter`), ISO `sowing_date` (`YYYY-MM-DD`), and `notes`. Every operation requires the existing bearer token and is scoped to its user. Foreign or archived fields return 404. `DELETE` soft-archives a field so its past analyses and reports remain intact.

`POST /api/analyze` remains compatible with existing multipart requests. It accepts an optional `field_id`; when omitted, the analysis remains unassigned. A supplied ID must refer to an active field owned by the authenticated user. Associated analysis responses and history include additive `field_id` and `field` metadata; old unassigned analysis records remain unchanged.

`GET /api/fields/{field_id}/weather` requires authentication and the active field must belong to the caller. It returns normalized current conditions with `provider`, approximate `location`, `temperature_c`, `humidity_percent`, `precipitation_mm`, `wind_speed_kmh`, `condition`, and provider `observed_at`. If location or provider data is unavailable, it returns HTTP 200 with `{"available": false, "message": "Weather information currently unavailable."}`. A weather API failure does not fail an analysis. New analysis responses include a `weather` result; successful snapshots are stored in the analysis payload, while unavailable status is returned only to the analysis caller. Historical analyses without weather remain valid.

The backend uses Open-Meteo Forecast and Geocoding APIs; no API key is required or sent to the frontend. Backend settings are `WEATHER_TIMEOUT_SECONDS` (default 5) and `WEATHER_CACHE_MINUTES` (default 30). Recent observations are reused per rounded location for the cache period. Weather is contextual information only and does not establish disease cause, crop loss, insurance eligibility, or compensation.

`GET /api/analyses/{id}/gradcam` requires the existing bearer token and returns the generated overlay as `image/png` with `Cache-Control: private, no-store`. The result page displays the original upload from its local preview beside the overlay; this avoids base64 copies of large images in a JSON response. The route looks up the analysis scoped to the caller and verifies its private source image is under the upload directory before generating output. Missing/foreign analyses and missing source images return 404; explanation failures return a generic 503. Grad-CAM is generated on demand and not stored. It is a qualitative model explanation, not a validated segmentation map or crop-loss measure. The ordinary analysis and history responses are unchanged.
