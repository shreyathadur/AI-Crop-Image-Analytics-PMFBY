import io
import hashlib
import sqlite3
import json
from urllib.parse import urlparse, parse_qs
from unittest.mock import MagicMock
from datetime import datetime, timezone

import pytest
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend import auth, database
from backend.config import DISCLAIMER
from backend.main import app, _validate_production_configuration
from backend.services.reports import make_pdf
from backend import config
from backend.services.visual_severity import estimate_visual_damage
from backend.services import weather as weather_service
from backend.services import gradcam as gradcam_service
import httpx


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_URL", f"sqlite:///{tmp_path / 'test.sqlite3'}")
    monkeypatch.setattr(auth, "SECRET_KEY", "test-only-secret-that-is-not-used-outside-tests")
    monkeypatch.setattr("backend.main.APP_ENV", "test")
    monkeypatch.setattr("backend.main.UPLOAD_DIR", tmp_path / "uploads")
    with TestClient(app) as test_client:
        yield test_client


def create_account(client, email="field@example.org"):
    response = client.post("/api/auth/register", json={"email": email, "password": "correct horse battery"})
    assert response.status_code == 201
    response = client.post("/api/auth/login", json={"email": email, "password": "correct horse battery"})
    return response.json()["access_token"]


def png_bytes():
    image = Image.new("RGB", (32, 32), (45, 120, 50)); content = io.BytesIO(); image.save(content, format="PNG")
    return content.getvalue()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_gradcam_service_generates_finite_png_for_saved_rust_image():
    rust_dir = next(path for path in (config.MODEL_PATH.parents[1] / "data" / "test").iterdir()
                    if path.name.startswith("Corn_(maize)___Common_rust"))
    image_path = next(rust_dir.glob("*.jpg"))
    from backend.services.inference import predict_image
    prediction = predict_image(image_path)
    explanation = gradcam_service.generate_gradcam(image_path, prediction["prediction"])
    from backend.services.inference import predict_scores
    scores, _ = predict_scores(image_path)
    label_index = int(np.argmax(scores))
    with Image.open(io.BytesIO(explanation["overlay_png"])) as overlay:
        assert overlay.format == "PNG"
        assert overlay.size == Image.open(image_path).size
        assert np.isfinite(np.asarray(overlay)).all()
    with Image.open(io.BytesIO(explanation["heatmap_png"])) as heatmap:
        assert heatmap.format == "PNG"
        assert heatmap.size == overlay.size
        assert np.isfinite(np.asarray(heatmap)).all()
    with Image.open(io.BytesIO(explanation["original_png"])) as original:
        assert original.format == "PNG"
        assert original.size == overlay.size
    assert explanation["prediction"] == prediction["prediction"]
    assert explanation["target_class_index"] == label_index
    _, target_name = gradcam_service._explanation_model()
    assert explanation["target_layer"] == target_name


def test_gradcam_healthy_corn_uses_inference_class_index():
    healthy_dir = next(path for path in (config.MODEL_PATH.parents[1] / "data" / "test").iterdir()
                       if path.name.startswith("Corn_(maize)___healthy"))
    image_path = next(healthy_dir.glob("*.jpg"))
    from backend.services.inference import predict_image, predict_scores
    prediction = predict_image(image_path)
    scores, _ = predict_scores(image_path)
    explanation = gradcam_service.generate_gradcam(image_path, prediction["prediction"])
    assert explanation["target_class_index"] == int(np.argmax(scores))
    assert explanation["prediction"] == prediction["prediction"]


def test_gradcam_zero_heatmap_is_finite_and_missing_file_fails_cleanly(tmp_path):
    overlay = gradcam_service._render_overlay(Image.new("RGB", (24, 18), "green"), np.zeros((7, 7), dtype=np.float32))
    with Image.open(io.BytesIO(overlay)) as image:
        assert image.size == (24, 18)
        assert np.isfinite(np.asarray(image)).all()
    with pytest.raises(FileNotFoundError):
        gradcam_service.generate_gradcam(tmp_path / "missing.png", "Corn (maize) — Common rust")


def test_gradcam_withholds_overlay_when_graph_argmax_differs_from_inference(monkeypatch):
    rust_dir = next(path for path in (config.MODEL_PATH.parents[1] / "data" / "test").iterdir()
                    if path.name.startswith("Corn_(maize)___Common_rust"))
    image_path = next(rust_dir.glob("*.jpg"))
    from backend.services.inference import predict_image, predict_scores
    prediction = predict_image(image_path)
    inference_scores, model_config = predict_scores(image_path)
    inference_index = int(np.argmax(inference_scores))
    wrong_index = (inference_index + 1) % len(model_config["labels"])
    import tensorflow as tf

    class MismatchingExplanationGraph:
        def __call__(self, inputs, training=False):
            feature_maps = tf.ones((1, 7, 7, 2), dtype=tf.float32)
            scores = tf.one_hot([wrong_index], len(model_config["labels"]), dtype=tf.float32)
            return feature_maps, scores

    monkeypatch.setattr(gradcam_service, "_explanation_model", lambda: (MismatchingExplanationGraph(), "Conv_1"))
    with pytest.raises(gradcam_service.GradCAMConsistencyError):
        gradcam_service.generate_gradcam(image_path, prediction["prediction"])


def test_gradcam_endpoint_is_owner_scoped_and_hides_internal_paths(client, monkeypatch, tmp_path):
    owner_token = create_account(client, "gradcam-owner@example.org")
    other_token = create_account(client, "gradcam-other@example.org")
    image_path = tmp_path / "uploads" / "owned.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), "green").save(image_path)
    analysis = database.save_analysis({"user_id": auth.verify_token(owner_token), "prediction": "Corn (maize) — Common rust",
                                       "image_path": str(image_path), "image_name": "owned.png"})
    monkeypatch.setattr("backend.main.generate_gradcam", lambda path, prediction, **kwargs: {
        "overlay_png": b"overlay", "prediction": prediction, "target_layer": "Conv_1"})
    own = client.get(f"/api/analyses/{analysis['id']}/gradcam", headers={"Authorization": f"Bearer {owner_token}"})
    assert own.status_code == 200
    assert own.content == b"overlay" and own.headers["content-type"] == "image/png"
    assert own.headers["cache-control"] == "private, no-store"
    assert str(image_path) not in own.text
    denied = client.get(f"/api/analyses/{analysis['id']}/gradcam", headers={"Authorization": f"Bearer {other_token}"})
    assert denied.status_code == 404
    assert client.get(f"/api/analyses/{analysis['id']}/gradcam").status_code == 401


def test_gradcam_api_missing_analysis_and_deleted_source_are_404(client):
    token = create_account(client)
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/api/analyses/missing/gradcam", headers=headers).status_code == 404
    analysis = database.save_analysis({"user_id": auth.verify_token(token), "prediction": "Corn (maize) — healthy",
                                       "image_path": str(config.UPLOAD_DIR / "deleted-image.png")})
    assert client.get(f"/api/analyses/{analysis['id']}/gradcam", headers=headers).status_code == 404


def test_pdf_still_renders_when_gradcam_fails(monkeypatch, tmp_path):
    image_path = tmp_path / "crop.png"
    Image.new("RGB", (32, 32), "green").save(image_path)
    monkeypatch.setattr(gradcam_service, "generate_gradcam", lambda *_: (_ for _ in ()).throw(
        gradcam_service.GradCAMConsistencyError("inconsistent explanation")))
    pdf = make_pdf({"id": "pdf-test", "image_path": str(image_path), "image_name": "crop.png", "confidence": .8,
                    "prediction": "Corn (maize) — Common rust", "crop": "Corn", "health_status": "Affected"})
    assert pdf.startswith(b"%PDF")


def test_register_new_user(client):
    response = client.post("/api/auth/register", json={
        "email": "field@example.org", "password": "correct horse battery",
    })
    assert response.status_code == 201
    assert response.json()["email"] == "field@example.org"


def test_register_existing_email(client):
    create_account(client)
    response = client.post("/api/auth/register", json={
        "email": "field@example.org", "password": "correct horse battery",
    })
    assert response.status_code == 409
    assert response.json()["detail"] == "An account with that email already exists"


def test_login_with_valid_credentials_returns_usable_token(client):
    create_account(client)
    response = client.post("/api/auth/login", json={
        "email": "field@example.org", "password": "correct horse battery",
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["access_token"]
    assert payload["token_type"] == "bearer"
    assert payload["user"]["email"] == "field@example.org"
    assert auth.verify_token(payload["access_token"]) == payload["user"]["id"]
    protected = client.get("/api/analyses", headers={
        "Authorization": f"Bearer {payload['access_token']}",
    })
    assert protected.status_code == 200
    assert protected.json() == []


def test_login_with_incorrect_password(client):
    create_account(client)
    response = client.post("/api/auth/login", json={
        "email": "field@example.org", "password": "incorrect password",
    })
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password"


def test_login_with_nonexistent_email(client):
    response = client.post("/api/auth/login", json={
        "email": "missing@example.org", "password": "correct horse battery",
    })
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password"


def test_local_password_reset_flow_is_hashed_single_use_and_generic(client, monkeypatch, tmp_path):
    create_account(client)
    monkeypatch.setattr("backend.main.APP_ENV", "development")
    monkeypatch.setattr("backend.main.SMTP_HOST", "")
    monkeypatch.setattr("backend.main.SMTP_USERNAME", "")
    monkeypatch.setattr("backend.main.SMTP_PASSWORD", "")
    monkeypatch.setattr("backend.main.SMTP_FROM_EMAIL", "")
    existing = client.post("/api/auth/forgot-password", json={"email": "field@example.org"})
    missing = client.post("/api/auth/forgot-password", json={"email": "missing@example.org"})
    assert existing.status_code == missing.status_code == 200
    assert existing.json()["message"] == missing.json()["message"]
    assert existing.json()["development_only"] is True
    assert missing.json().get("development_only") is None
    link = existing.json()["reset_url"]
    reset_token = parse_qs(urlparse(link).query)["token"][0]
    with sqlite3.connect(database._sqlite_path()) as conn:
        stored_hash = conn.execute("SELECT token_hash FROM password_reset_tokens").fetchone()[0]
    assert stored_hash == hashlib.sha256(reset_token.encode()).hexdigest()
    assert reset_token not in stored_hash
    changed = client.post("/api/auth/reset-password", json={"token": reset_token, "new_password": "brand new secure passphrase"})
    assert changed.status_code == 200
    assert reset_token not in changed.text and "brand new secure passphrase" not in changed.text
    assert client.post("/api/auth/reset-password", json={"token": reset_token, "new_password": "another secure passphrase"}).status_code == 400
    assert client.post("/api/auth/login", json={"email": "field@example.org", "password": "correct horse battery"}).status_code == 401
    assert client.post("/api/auth/login", json={"email": "field@example.org", "password": "brand new secure passphrase"}).status_code == 200


def test_production_password_reset_never_returns_token(client, monkeypatch):
    create_account(client)
    monkeypatch.setattr("backend.main.APP_ENV", "production")
    response = client.post("/api/auth/forgot-password", json={"email": "field@example.org"})
    assert response.status_code == 200
    assert "reset_url" not in response.json() and "token" not in response.text


def test_password_reset_delivery_configuration_does_not_enumerate_accounts(client, monkeypatch):
    create_account(client)
    monkeypatch.setattr("backend.main.APP_ENV", "production")
    monkeypatch.setattr("backend.main.SMTP_HOST", "smtp.example.com")
    monkeypatch.setattr("backend.main.SMTP_USERNAME", "")
    monkeypatch.setattr("backend.main.SMTP_PASSWORD", "")
    monkeypatch.setattr("backend.main.SMTP_FROM_EMAIL", "")
    existing = client.post("/api/auth/forgot-password", json={"email": "field@example.org"})
    missing = client.post("/api/auth/forgot-password", json={"email": "missing@example.org"})
    assert existing.status_code == missing.status_code == 200
    assert existing.json() == missing.json()
    assert "reset_url" not in existing.json()


def test_smtp_delivery_failure_does_not_enumerate_accounts(client, monkeypatch):
    create_account(client)
    monkeypatch.setattr("backend.main.APP_ENV", "production")
    for name, value in (("SMTP_HOST", "smtp.example.com"), ("SMTP_USERNAME", "user@example.com"),
                        ("SMTP_PASSWORD", "test-only-mail-password"), ("SMTP_FROM_EMAIL", "user@example.com")):
        monkeypatch.setattr(f"backend.main.{name}", value)
    monkeypatch.setattr("backend.main._send_reset_email", lambda *_: (_ for _ in ()).throw(OSError("private error")))
    existing = client.post("/api/auth/forgot-password", json={"email": "field@example.org"})
    missing = client.post("/api/auth/forgot-password", json={"email": "missing@example.org"})
    assert existing.status_code == missing.status_code == 200
    assert existing.json() == missing.json()
    assert "reset_url" not in existing.json()


def test_production_requires_strong_secret_and_https_frontend_origin(monkeypatch):
    monkeypatch.setattr("backend.main.APP_ENV", "production")
    monkeypatch.setattr("backend.main.SECRET_KEY", "short")
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        _validate_production_configuration()
    monkeypatch.setattr("backend.main.SECRET_KEY", "x" * 40)
    monkeypatch.setattr("backend.main.FRONTEND_ORIGIN", "http://localhost:5173")
    with pytest.raises(RuntimeError, match="HTTPS origin"):
        _validate_production_configuration()
    monkeypatch.setattr("backend.main.FRONTEND_ORIGIN", "https://crop-ui.example")
    monkeypatch.setattr("backend.main.MONGODB_URI", "mongodb://db.example/crop")
    monkeypatch.setenv("UPLOAD_DIR", "/var/data/uploads")
    _validate_production_configuration()


def test_production_rejects_ephemeral_default_database_and_uploads(monkeypatch):
    monkeypatch.setattr("backend.main.APP_ENV", "production")
    monkeypatch.setattr("backend.main.SECRET_KEY", "x" * 40)
    monkeypatch.setattr("backend.main.FRONTEND_ORIGIN", "https://crop-ui.example")
    monkeypatch.setattr("backend.main.MONGODB_URI", "")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("UPLOAD_DIR", raising=False)
    with pytest.raises(RuntimeError, match="UPLOAD_DIR"):
        _validate_production_configuration()
    monkeypatch.setenv("UPLOAD_DIR", "/var/data/uploads")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./crop_analytics.db")
    with pytest.raises(RuntimeError, match="absolute persistent path"):
        _validate_production_configuration()
    monkeypatch.setenv("DATABASE_URL", "sqlite:////var/data/crop_analytics.db")
    _validate_production_configuration()


def field_body(**overrides):
    return {"field_name": "North Plot", "crop_name": "Corn", "variety": "Hybrid A",
            "state": "Telangana", "district": "Hyderabad", "village": "Village 1",
            "latitude": 17.385, "longitude": 78.486, "area": 2.5,
            "area_unit": "acre", "sowing_date": "2026-06-01", "notes": "North irrigation line", **overrides}


def test_field_crud_and_archive_preserves_associations(client):
    token = create_account(client)
    headers = {"Authorization": f"Bearer {token}"}
    created = client.post("/api/fields", headers=headers, json=field_body())
    assert created.status_code == 201, created.text
    field = created.json()
    assert field["field_name"] == "North Plot" and "user_id" not in field
    assert client.get("/api/fields", headers=headers).json() == [field]
    detail = client.get(f"/api/fields/{field['id']}", headers=headers)
    assert detail.status_code == 200 and detail.json()["analysis_count"] == 0
    updated = client.put(f"/api/fields/{field['id']}", headers=headers, json=field_body(field_name="East Plot", area=3))
    assert updated.status_code == 200 and updated.json()["field_name"] == "East Plot"
    assert client.delete(f"/api/fields/{field['id']}", headers=headers).status_code == 204
    assert client.get("/api/fields", headers=headers).json() == []
    assert database.get_field(field["id"], database.get_user_by_email("field@example.org")["id"], include_archived=True)["is_active"] is False
    assert client.get(f"/api/fields/{field['id']}", headers=headers).status_code == 404


@pytest.mark.parametrize("overrides", [
    {"field_name": ""}, {"field_name": "   "}, {"crop_name": ""},
    {"latitude": 90.1}, {"latitude": -90.1}, {"longitude": 180.1}, {"longitude": -180.1},
    {"latitude": 10, "longitude": None}, {"area": 0}, {"area": -2},
    {"area_unit": "sq_km"}, {"sowing_date": "not-a-date"}, {"notes": "x" * 2001},
])
def test_invalid_field_data_is_rejected(client, overrides):
    token = create_account(client)
    response = client.post("/api/fields", headers={"Authorization": f"Bearer {token}"}, json=field_body(**overrides))
    assert response.status_code == 422


def test_field_routes_require_authentication(client):
    assert client.get("/api/fields").status_code == 401
    assert client.post("/api/fields", json=field_body()).status_code == 401


def test_user_cannot_read_update_or_delete_another_users_field(client):
    owner_token = create_account(client, "owner@example.org")
    other_token = create_account(client, "other@example.org")
    field = client.post("/api/fields", headers={"Authorization": f"Bearer {owner_token}"}, json=field_body()).json()
    other_headers = {"Authorization": f"Bearer {other_token}"}
    assert client.get(f"/api/fields/{field['id']}", headers=other_headers).status_code == 404
    assert client.put(f"/api/fields/{field['id']}", headers=other_headers, json=field_body(field_name="Stolen")).status_code == 404
    assert client.delete(f"/api/fields/{field['id']}", headers=other_headers).status_code == 404
    assert client.get(f"/api/fields/{field['id']}", headers={"Authorization": f"Bearer {owner_token}"}).status_code == 200


def test_analysis_can_be_associated_with_owned_field_and_history_exposes_field_metadata(client, monkeypatch):
    token = create_account(client)
    headers = {"Authorization": f"Bearer {token}"}
    field = client.post("/api/fields", headers=headers, json=field_body()).json()
    monkeypatch.setattr("backend.main.predict_image", lambda path: {
        "prediction": "Corn — healthy", "crop": "Corn", "detected_condition": "healthy",
        "health_status": "Healthy", "confidence": .9, "severity": "Healthy",
        "estimated_visual_damage_indicator": 0.0, "severity_method": "image_based_visual_estimation",
        "severity_explanation": "heuristic", "explanation": "test"})
    weather_snapshot = {"available": True, "provider": "Open-Meteo", "location": "Village 1, Hyderabad, Telangana",
                        "temperature_c": 28.0, "humidity_percent": 72.0, "precipitation_mm": 0.0,
                        "wind_speed_kmh": 12.0, "condition": "Partly cloudy", "observed_at": "2026-10-06T10:00:00+00:00"}
    monkeypatch.setattr("backend.main.get_weather", lambda _: weather_snapshot)
    response = client.post("/api/analyze", headers=headers, data={"field_id": field["id"]},
                           files={"image": ("leaf.png", png_bytes(), "image/png")})
    assert response.status_code == 201, response.text
    item = response.json()
    assert item["field_id"] == field["id"] and item["field"] == {"id": field["id"], "field_name": "North Plot"}
    assert item["weather"] == weather_snapshot
    assert "image_path" not in item and "user_id" not in item
    assert client.get("/api/analyses", headers=headers).json()[0]["field"]["field_name"] == "North Plot"
    assert client.get(f"/api/fields/{field['id']}", headers=headers).json()["analysis_count"] == 1
    assert client.get("/api/analyses", headers=headers).json()[0]["weather"]["available"] is True
    report = client.get(f"/api/report/{item['id']}", headers=headers)
    assert report.status_code == 200 and b"ENVIRONMENTAL CONTEXT" in report.content
    assert b"Partly cloudy" in report.content and all(word in report.content for word in (b"Environmental", b"disclaimer", b"establish", b"cause", b"compensation"))


def test_analysis_succeeds_when_weather_is_unavailable_and_does_not_persist_fake_snapshot(client, monkeypatch):
    token = create_account(client)
    headers = {"Authorization": f"Bearer {token}"}
    field = client.post("/api/fields", headers=headers, json=field_body()).json()
    monkeypatch.setattr("backend.main.predict_image", lambda _: {"prediction":"Corn", "crop":"Corn", "health_status":"Healthy", "confidence":.9})
    monkeypatch.setattr("backend.main.get_weather", lambda _: dict(weather_service.WEATHER_UNAVAILABLE))
    response = client.post("/api/analyze", headers=headers, data={"field_id":field["id"]},
                           files={"image":("leaf.png",png_bytes(),"image/png")})
    assert response.status_code == 201
    assert response.json()["prediction"] == "Corn"
    assert response.json()["weather"]["available"] is False
    history_item = client.get("/api/analyses", headers=headers).json()[0]
    assert "weather" not in history_item
    pdf = client.get(f"/api/report/{response.json()['id']}", headers=headers)
    assert pdf.status_code == 200 and b"ENVIRONMENTAL CONTEXT" not in pdf.content


def test_weather_service_normalizes_and_caches_open_meteo_response(client, monkeypatch):
    calls = []
    provider = {"current": {"time":"2026-10-06T10:30", "temperature_2m":28.0, "relative_humidity_2m":72,
                             "precipitation":0.0, "wind_speed_10m":12.0, "weather_code":2}}
    def fake_request(url, params):
        calls.append((url, params))
        return provider
    monkeypatch.setattr(weather_service, "_request_json", fake_request)
    field = {"latitude":17.385, "longitude":78.486, "village":"Village 1", "district":"Hyderabad", "state":"Telangana"}
    result = weather_service.get_weather(field)
    assert result == {"available":True, "location":"Village 1, Hyderabad, Telangana", "provider":"Open-Meteo",
                      "temperature_c":28.0, "humidity_percent":72.0, "precipitation_mm":0.0,
                      "wind_speed_kmh":12.0, "condition":"Partly cloudy", "observed_at":"2026-10-06T10:30+00:00"}
    assert calls[0][0] == weather_service.FORECAST_URL
    assert calls[0][1]["latitude"] == 17.39 and "api_key" not in calls[0][1]
    monkeypatch.setattr(weather_service, "_request_json", lambda *_: pytest.fail("A fresh cache entry should be reused"))
    assert weather_service.get_weather(field)["temperature_c"] == 28.0


def test_weather_service_does_not_require_api_key_and_supports_admin_location(client, monkeypatch):
    urls = []
    def fake_request(url, params):
        urls.append(url)
        if url == weather_service.GEOCODING_URL:
            return {"results":[{"latitude":17.4,"longitude":78.5,"name":"Hyderabad"}]}
        return {"current":{"time":"2026-10-06T10:00","temperature_2m":25,"relative_humidity_2m":60,
                            "precipitation":0,"wind_speed_10m":3,"weather_code":1}}
    monkeypatch.setattr(weather_service, "_request_json", fake_request)
    result = weather_service.get_weather({"state":"Telangana","district":"Hyderabad"})
    assert result["available"] is True and result["location"] == "Hyderabad, Telangana"
    assert urls == [weather_service.GEOCODING_URL, weather_service.FORECAST_URL]
    assert "WEATHER_API_KEY" not in result


@pytest.mark.parametrize("failure", [
    httpx.ReadTimeout("timeout"), httpx.ConnectError("offline"),
    httpx.HTTPStatusError("provider error", request=httpx.Request("GET", weather_service.FORECAST_URL), response=httpx.Response(503)),
])
def test_weather_service_handles_timeout_and_provider_http_errors(client, monkeypatch, failure):
    monkeypatch.setattr(weather_service, "_request_json", lambda *_: (_ for _ in ()).throw(failure))
    result = weather_service.get_weather({"latitude":10,"longitude":20})
    assert result == weather_service.WEATHER_UNAVAILABLE
    assert "timeout" not in str(result).lower() and "offline" not in str(result).lower()


def test_weather_service_rejects_invalid_provider_response_and_missing_location(client, monkeypatch):
    monkeypatch.setattr(weather_service, "_request_json", lambda *_: {"current":{}})
    assert weather_service.get_weather({"latitude":11,"longitude":22}) == weather_service.WEATHER_UNAVAILABLE
    monkeypatch.setattr(weather_service, "_request_json", lambda *_: pytest.fail("No location means no request"))
    assert weather_service.get_weather({"field_name":"No location"}) == weather_service.WEATHER_UNAVAILABLE


def test_weather_endpoint_requires_auth_and_enforces_field_ownership(client, monkeypatch):
    owner = create_account(client, "weather-owner@example.org")
    other = create_account(client, "weather-other@example.org")
    owner_headers = {"Authorization":f"Bearer {owner}"}
    field = client.post("/api/fields", headers=owner_headers, json=field_body()).json()
    result = {"available":True,"provider":"Open-Meteo","location":"Approximate location","temperature_c":22.0,
              "humidity_percent":55.0,"precipitation_mm":0.0,"wind_speed_kmh":5.0,"condition":"Clear sky","observed_at":"2026-10-06T10:00:00+00:00"}
    monkeypatch.setattr("backend.main.get_weather", lambda _: result)
    assert client.get(f"/api/fields/{field['id']}/weather").status_code == 401
    response = client.get(f"/api/fields/{field['id']}/weather", headers=owner_headers)
    assert response.status_code == 200 and response.json() == result
    assert "WEATHER_API_KEY" not in response.text
    assert client.get(f"/api/fields/{field['id']}/weather", headers={"Authorization":f"Bearer {other}"}).status_code == 404


def test_mongodb_weather_cache_adapter_uses_upsert_and_reads_fresh_entry(monkeypatch):
    db = MagicMock()
    payload = {"provider":"Open-Meteo","temperature_c":20.0}
    db.weather_cache.find_one.return_value = {"payload":payload,"fetched_at":datetime.now(timezone.utc).isoformat()}
    monkeypatch.setattr(database, "_mongo", lambda: db)
    database.save_weather_cache("cache-hash", payload)
    assert database.get_cached_weather("cache-hash", 1800) == payload
    assert db.weather_cache.update_one.call_args.kwargs["upsert"] is True


def test_analysis_cannot_be_associated_with_another_users_field(client, monkeypatch):
    owner = create_account(client, "owner@example.org")
    other = create_account(client, "other@example.org")
    field = client.post("/api/fields", headers={"Authorization": f"Bearer {owner}"}, json=field_body()).json()
    monkeypatch.setattr("backend.main.predict_image", lambda _: pytest.fail("Inference must not run for a foreign field"))
    response = client.post("/api/analyze", headers={"Authorization": f"Bearer {other}"}, data={"field_id": field["id"]},
                           files={"image": ("leaf.png", png_bytes(), "image/png")})
    assert response.status_code == 404


def test_existing_sqlite_schema_is_migrated_additively(tmp_path, monkeypatch):
    legacy_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(legacy_path) as conn:
        conn.execute("CREATE TABLE users (id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, created_at TEXT NOT NULL)")
        conn.execute("CREATE TABLE analyses (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL)")
        conn.execute("INSERT INTO users VALUES (?, ?, ?, ?)", ("legacy-user", "legacy@example.org", "old-hash", "2025-01-01T00:00:00+00:00"))
        conn.execute("INSERT INTO analyses VALUES (?, ?, ?, ?)", ("legacy-analysis", "legacy-user", json.dumps({"id":"legacy-analysis","user_id":"legacy-user","prediction":"Corn — healthy"}), "2025-01-02T00:00:00+00:00"))
    monkeypatch.setattr(database, "DATABASE_URL", f"sqlite:///{legacy_path}")
    database.initialize()
    assert database.get_user_by_email("legacy@example.org")["id"] == "legacy-user"
    assert database.list_analyses("legacy-user")[0]["prediction"] == "Corn — healthy"
    with sqlite3.connect(legacy_path) as conn:
        assert "field_id" in {row[1] for row in conn.execute("PRAGMA table_info(analyses)")}
        assert conn.execute("SELECT COUNT(*) FROM fields").fetchone()[0] == 0


def test_mongodb_field_adapter_uses_owner_scoped_collection(monkeypatch):
    db = MagicMock()
    db.fields.find.return_value.sort.return_value = [{"_id":"mongo-field", "user_id":"mongo-user", "field_name":"Plot", "crop_name":"Rice", "is_active":True}]
    db.fields.find_one.return_value = {"_id":"mongo-field", "user_id":"mongo-user", "field_name":"Plot", "crop_name":"Rice", "is_active":True}
    db.fields.update_one.return_value.matched_count = 1
    db.fields.update_one.return_value.modified_count = 1
    monkeypatch.setattr(database, "_mongo", lambda: db)
    created = database.create_field("mongo-user", {"field_name":"Plot", "crop_name":"Rice"})
    assert created["field_name"] == "Plot"
    assert database.list_fields("mongo-user")[0]["id"] == "mongo-field"
    assert database.get_field("mongo-field", "mongo-user")["id"] == "mongo-field"
    assert database.update_field("mongo-field", "mongo-user", {"field_name":"Plot", "crop_name":"Rice"})["id"] == "mongo-field"
    assert database.archive_field("mongo-field", "mongo-user")
    assert db.fields.insert_one.call_args.args[0]["user_id"] == "mongo-user"
    assert db.fields.update_one.call_args.args[0]["user_id"] == "mongo-user"


@pytest.mark.parametrize("origin", ["http://localhost:5173", "http://127.0.0.1:5173"])
@pytest.mark.parametrize("path", ["/api/auth/register", "/api/auth/login"])
def test_auth_cors_preflight(client, origin, path):
    response = client.options(path, headers={
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization,content-type",
    })
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    assert "POST" in response.headers["access-control-allow-methods"]
    assert response.headers["access-control-allow-credentials"] == "true"


def test_passwords_are_hashed():
    stored = auth.hash_password("a long secure passphrase")
    assert "a long secure passphrase" not in stored
    assert auth.verify_password("a long secure passphrase", stored)
    assert not auth.verify_password("incorrect", stored)


def test_secret_key_falls_back_to_dotenv_when_process_value_is_blank(monkeypatch, tmp_path):
    secret = "test-only-dotenv-signing-key"
    env_file = tmp_path / ".env"
    env_file.write_text(f"SECRET_KEY={secret}\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_FILE", env_file)
    monkeypatch.setenv("SECRET_KEY", "")
    assert config._load_secret_key() == secret


def test_visual_severity_is_conservative_and_healthy_override(tmp_path):
    from PIL import ImageDraw
    image = Image.new("RGB", (128, 128), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((14, 9, 114, 119), fill=(38, 125, 48))
    draw.ellipse((45, 38, 70, 61), fill=(135, 75, 35))
    image_path = tmp_path / "leaf.png"
    image.save(image_path)

    diseased = estimate_visual_damage(image_path, healthy=False)
    healthy = estimate_visual_damage(image_path, healthy=True)
    assert 0 <= diseased["estimated_visual_damage_indicator"] <= 100
    assert diseased["severity"] in {"Low", "Moderate", "Severe"}
    assert diseased["severity_method"] == "image_based_visual_estimation"
    assert healthy["severity"] == "Healthy"
    assert healthy["estimated_visual_damage_indicator"] == 0.0


def test_image_validation_and_analysis_storage(client, monkeypatch):
    token = create_account(client)
    headers = {"Authorization": f"Bearer {token}"}
    invalid = client.post("/api/analyze", headers=headers, files={"image": ("bad.txt", b"x", "text/plain")})
    assert invalid.status_code == 415
    corrupt = client.post("/api/analyze", headers=headers, files={"image": ("bad.png", b"not an image", "image/png")})
    assert corrupt.status_code == 400
    monkeypatch.setattr("backend.main.predict_image", lambda path: {
        "prediction": "Tomato — healthy", "crop": "Tomato", "health_status": "Healthy", "confidence": .91,
        "detected_condition": "healthy", "severity": "Healthy",
        "estimated_visual_damage_indicator": 0.0,
        "severity_method": "image_based_visual_estimation",
        "severity_explanation": "Severity is an AI-assisted visual estimate based on visible affected regions in the uploaded image. It is not an official PMFBY loss percentage.",
        "explanation": "Test inference"})
    response = client.post("/api/analyze", headers=headers, files={"image": ("leaf.png", png_bytes(), "image/png")})
    assert response.status_code == 201, response.text
    item = response.json()
    assert item["crop"] == "Tomato" and item["disclaimer"] == DISCLAIMER
    assert item["severity"] == "Healthy"
    assert item["estimated_visual_damage_indicator"] == 0.0
    assert item["severity_method"] == "image_based_visual_estimation"
    assert "image_path" not in item and "user_id" not in item
    assert len(client.get("/api/analyses", headers=headers).json()) == 1
    history_item = client.get("/api/analyses", headers=headers).json()[0]
    assert history_item["estimated_visual_damage_indicator"] == 0.0
    assert client.get(f"/api/analyses/{item['id']}", headers=headers).status_code == 200
    report = client.get(f"/api/report/{item['id']}", headers=headers)
    assert report.status_code == 200 and report.headers["content-type"] == "application/pdf"
    assert report.content.startswith(b"%PDF")
    assert all(word in report.content for word in (b"Estimated", b"Visual", b"Damage", b"Indicator"))
    assert b"Severity Category" in report.content
    assert b"Severity Method" in report.content
    assert b"official PMFBY claim percentage" in report.content
    assert client.delete(f"/api/analyses/{item['id']}", headers=headers).status_code == 204


def test_authentication_required(client):
    assert client.get("/api/analyses").status_code == 401


def test_report_generation():
    report = make_pdf({"id": "abc", "created_at": "today", "confidence": .5, "explanation": "Preliminary"})
    assert report.startswith(b"%PDF")


def test_missing_model_is_not_a_fake_prediction(client, monkeypatch):
    token = create_account(client)
    monkeypatch.setattr("backend.main.predict_image", lambda _: (_ for _ in ()).throw(FileNotFoundError()))
    response = client.post("/api/analyze", headers={"Authorization": f"Bearer {token}"},
                           files={"image": ("leaf.png", png_bytes(), "image/png")})
    assert response.status_code == 503
    assert "model is not available" in response.json()["detail"]
