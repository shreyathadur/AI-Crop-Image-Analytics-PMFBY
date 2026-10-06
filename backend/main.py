"""FastAPI application for preliminary crop leaf image analysis."""
import asyncio
import logging
import hashlib
import secrets
import smtplib
from email.message import EmailMessage
import re
import os
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator
from typing import Literal

from backend import database
from backend.auth import hash_password, issue_token, verify_password, verify_token
from backend.config import (APP_ENV, DISCLAIMER, FRONTEND_ORIGIN, MAX_UPLOAD_MB, MONGODB_URI, SECRET_KEY,
                            MODEL_PATH, UPLOAD_DIR, RESET_TOKEN_EXPIRE_MINUTES, SMTP_HOST,
                            SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SMTP_FROM_EMAIL, SMTP_FROM_NAME)
from backend.services.inference import predict_image
from backend.services.reports import make_pdf
from backend.services.weather import WEATHER_UNAVAILABLE, get_weather
from backend.services.gradcam import generate_gradcam

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ALLOWED_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
ALLOWED_FRONTEND_ORIGINS = {FRONTEND_ORIGIN.rstrip("/")}
if APP_ENV in {"development", "dev", "local", "test"}:
    ALLOWED_FRONTEND_ORIGINS.update({"http://localhost:5173", "http://127.0.0.1:5173"})
ALLOWED_FRONTEND_ORIGINS = sorted(ALLOWED_FRONTEND_ORIGINS)
bearer = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _validate_production_configuration()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    database.initialize()
    yield


def _validate_production_configuration():
    if APP_ENV != "production":
        return
    if len(SECRET_KEY) < 32:
        raise RuntimeError("Production requires SECRET_KEY with at least 32 characters")
    frontend = urlsplit(FRONTEND_ORIGIN)
    if frontend.scheme != "https" or not frontend.netloc or frontend.path not in {"", "/"} or frontend.query or frontend.fragment:
        raise RuntimeError("Production requires FRONTEND_ORIGIN to be the HTTPS origin of the deployed frontend")
    if not MONGODB_URI:
        database_setting = os.getenv("DATABASE_URL", "").strip()
        if not database_setting.startswith("sqlite:///"):
            raise RuntimeError("Production must configure MongoDB or SQLite on persistent storage")
        sqlite_path = database_setting.removeprefix("sqlite:///")
        if not (sqlite_path.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", sqlite_path)):
            raise RuntimeError("Production SQLite DATABASE_URL must use an absolute persistent path")


app = FastAPI(title="Crop Image Analytics API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_FRONTEND_ORIGINS, allow_credentials=True,
                   allow_methods=["GET", "POST", "PUT", "DELETE"], allow_headers=["Authorization", "Content-Type"])


@app.exception_handler(Exception)
async def unexpected_error_handler(request, exc):
    logger.exception("Unhandled API error", exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "The service encountered an unexpected error"})


class RegisterInput(BaseModel):
    email: EmailStr
    password: str = Field(min_length=10, max_length=128)


class LoginInput(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class ForgotPasswordInput(BaseModel):
    email: EmailStr


class ResetPasswordInput(BaseModel):
    token: str = Field(min_length=32, max_length=256)
    new_password: str = Field(min_length=10, max_length=128)


class FieldInput(BaseModel):
    field_name: str = Field(min_length=1, max_length=100)
    crop_name: str = Field(min_length=1, max_length=100)
    variety: str | None = Field(default=None, max_length=100)
    state: str | None = Field(default=None, max_length=100)
    district: str | None = Field(default=None, max_length=100)
    village: str | None = Field(default=None, max_length=100)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    area: float | None = Field(default=None, gt=0)
    area_unit: Literal["acre", "hectare", "sq_meter"] | None = None
    sowing_date: date | None = None
    notes: str | None = Field(default=None, max_length=2000)

    @field_validator("field_name", "crop_name")
    @classmethod
    def required_text_not_blank(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("This field cannot be blank")
        return value

    @model_validator(mode="after")
    def coordinates_are_paired(self):
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("Latitude and longitude must be provided together")
        return self


def _send_reset_email(email, link):
    message = EmailMessage()
    message["Subject"] = "Reset your CropLens password"
    message["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>"
    message["To"] = email
    message.set_content(f"Use this link to reset your CropLens password. It expires in {RESET_TOKEN_EXPIRE_MINUTES} minutes:\n\n{link}\n")
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(message)


def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
    user_id = verify_token(credentials.credentials) if credentials else None
    if not user_id:
        raise HTTPException(status_code=401, detail="Sign in to continue")
    return user_id


@app.get("/health")
def health():
    return {"status": "ok", "model_available": MODEL_PATH.is_file() and MODEL_PATH.with_name("model_config.json").is_file(),
            "storage": "mongodb" if MONGODB_URI else "sqlite"}


@app.post("/api/auth/register", status_code=201)
def register(body: RegisterInput):
    email = body.email.lower()
    try:
        user = database.create_user(email, hash_password(body.password))
    except Exception as exc:
        if "unique" in str(exc).lower() or "duplicate" in str(exc).lower():
            raise HTTPException(status_code=409, detail="An account with that email already exists") from exc
        logger.exception("Registration persistence failure")
        raise HTTPException(status_code=503, detail="Account service is temporarily unavailable") from exc
    return {"id": user["id"], "email": email}


@app.post("/api/auth/login")
def login(body: LoginInput):
    user = database.get_user_by_email(body.email.lower())
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    try:
        token = issue_token(user["id"])
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Authentication is not configured on this server") from exc
    return {"access_token": token, "token_type": "bearer", "user": {"id": user["id"], "email": user["email"]}}


@app.post("/api/auth/forgot-password")
def forgot_password(body: ForgotPasswordInput):
    generic = {"message": "If an account exists for this email, a password reset link has been sent."}
    user = database.get_user_by_email(body.email.lower())
    if not user:
        return generic
    raw_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=RESET_TOKEN_EXPIRE_MINUTES)
    database.create_password_reset(user["id"], token_hash, expires_at.isoformat())
    link = f"{FRONTEND_ORIGIN.rstrip('/')}/reset-password?token={raw_token}"
    smtp_values = [SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, SMTP_FROM_EMAIL]
    smtp_configured = all(smtp_values)
    if any(smtp_values) and not smtp_configured:
        logger.error("Password reset email not sent because SMTP configuration is incomplete")
        return generic
    if smtp_configured:
        try:
            _send_reset_email(user["email"], link)
        except Exception as exc:
            logger.exception("Password reset email delivery failed")
            return generic
    elif APP_ENV.lower() in {"development", "dev", "local"}:
        logger.warning("DEVELOPMENT ONLY password reset link (do not use outside local development): %s", link)
        return generic | {"development_only": True, "reset_url": link}
    else:
        logger.error("SMTP is not configured; reset request accepted without email delivery")
    return generic


@app.post("/api/auth/reset-password")
def reset_password(body: ResetPasswordInput):
    token_hash = hashlib.sha256(body.token.encode()).hexdigest()
    user_id = database.consume_password_reset(token_hash, datetime.now(timezone.utc).isoformat())
    if not user_id:
        raise HTTPException(status_code=400, detail="This password reset link is invalid or expired")
    database.update_password(user_id, hash_password(body.new_password))
    return {"message": "Password reset successfully. You can now sign in with your new password."}


def field_summary(field, user_id):
    analyses = [row for row in database.list_analyses(user_id) if row.get("field_id") == field["id"]]
    latest = analyses[0] if analyses else None
    return field | {
        "analysis_count": len(analyses),
        "latest_analysis": ({"id": latest.get("id"), "created_at": latest.get("created_at"),
                             "prediction": latest.get("prediction"), "severity": latest.get("severity"),
                             "estimated_visual_damage_indicator": latest.get("estimated_visual_damage_indicator")}
                            if latest else None),
    }


@app.post("/api/fields", status_code=201)
def create_field(body: FieldInput, user_id: str = Depends(current_user)):
    return database.create_field(user_id, body.model_dump(mode="json"))


@app.get("/api/fields")
def list_fields(user_id: str = Depends(current_user)):
    return database.list_fields(user_id)


@app.get("/api/fields/{field_id}")
def get_field(field_id: str, user_id: str = Depends(current_user)):
    field = database.get_field(field_id, user_id)
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")
    return field_summary(field, user_id)


@app.get("/api/fields/{field_id}/weather")
async def field_weather(field_id: str, user_id: str = Depends(current_user)):
    field = database.get_field(field_id, user_id)
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")
    return await asyncio.to_thread(get_weather, field)


@app.put("/api/fields/{field_id}")
def update_field(field_id: str, body: FieldInput, user_id: str = Depends(current_user)):
    field = database.update_field(field_id, user_id, body.model_dump(mode="json"))
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")
    return field


@app.delete("/api/fields/{field_id}", status_code=204)
def delete_field(field_id: str, user_id: str = Depends(current_user)):
    if not database.archive_field(field_id, user_id):
        raise HTTPException(status_code=404, detail="Field not found")
    return Response(status_code=204)


def public_analysis(item):
    public = {key: value for key, value in item.items() if key not in {"user_id", "image_path"}} | {"disclaimer": DISCLAIMER}
    if item.get("field_id") and item.get("field_name"):
        public["field"] = {"id": item["field_id"], "field_name": item["field_name"]}
    return public


@app.post("/api/analyze", status_code=201)
async def analyze(image: UploadFile = File(...), field_id: str | None = Form(default=None),
                  user_id: str = Depends(current_user)):
    associated_field = None
    if field_id:
        associated_field = database.get_field(field_id, user_id)
        if not associated_field:
            raise HTTPException(status_code=404, detail="Field not found")
    if not image.filename:
        raise HTTPException(status_code=400, detail="Choose an image to analyze")
    suffix = ALLOWED_TYPES.get(image.content_type or "")
    if not suffix:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image")
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    contents = await image.read(max_bytes + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    if len(contents) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Image must be {MAX_UPLOAD_MB} MB or smaller")
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(image.filename).name)[:120] or "crop-image"
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    try:
        from io import BytesIO
        with Image.open(BytesIO(contents)) as decoded:
            if decoded.width < 16 or decoded.height < 16 or decoded.width * decoded.height > 25_000_000:
                raise ValueError("Image dimensions are outside the accepted range")
            expected_format = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}[image.content_type]
            if decoded.format != expected_format:
                raise ValueError("Image content does not match its declared type")
            decoded.verify()
        path.write_bytes(contents)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image")
    try:
        prediction = await asyncio.to_thread(predict_image, path)
    except (FileNotFoundError, ImportError):
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail="The trained analysis model is not available yet")
    except Exception as exc:
        logger.exception("Inference failed")
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail="Image analysis is temporarily unavailable") from exc
    weather_context = (await asyncio.to_thread(get_weather, associated_field)
                       if associated_field else dict(WEATHER_UNAVAILABLE))
    record = {"user_id": user_id, "image_name": safe_name, **prediction,
              "created_at": datetime.now(timezone.utc).isoformat(), "image_path": str(path)}
    if associated_field:
        record["field_id"] = associated_field["id"]
        record["field_name"] = associated_field["field_name"]
    if weather_context.get("available"):
        record["weather"] = weather_context
    try:
        result = database.save_analysis(record)
    except Exception as exc:
        logger.exception("Analysis persistence failed")
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail="Could not save the analysis; please try again") from exc
    return public_analysis(result) | {"weather": weather_context}


@app.get("/api/analyses")
def history(user_id: str = Depends(current_user)):
    return [public_analysis(row) for row in database.list_analyses(user_id)]


@app.get("/api/analyses/{analysis_id}")
def get_analysis(analysis_id: str, user_id: str = Depends(current_user)):
    row = database.get_analysis(analysis_id, user_id)
    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return public_analysis(row)


@app.get("/api/analyses/{analysis_id}/gradcam")
async def analysis_gradcam(analysis_id: str, user_id: str = Depends(current_user)):
    row = database.get_analysis(analysis_id, user_id)
    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found")
    try:
        image_path = Path(row.get("image_path", "")).resolve(strict=True)
        image_path.relative_to(UPLOAD_DIR.resolve())
    except (OSError, ValueError):
        raise HTTPException(status_code=404, detail="Analysis image not found")
    try:
        result = await asyncio.to_thread(generate_gradcam, image_path, row.get("prediction", ""),
                                         include_original=False, include_heatmap=False)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Analysis image not found")
    except Exception as exc:
        logger.exception("Grad-CAM request failed for analysis %s", analysis_id)
        raise HTTPException(status_code=503, detail="Model explanation is currently unavailable.") from exc
    return Response(content=result["overlay_png"], media_type="image/png",
                    headers={"Cache-Control": "private, no-store"})


@app.delete("/api/analyses/{analysis_id}", status_code=204)
def remove_analysis(analysis_id: str, user_id: str = Depends(current_user)):
    row = database.get_analysis(analysis_id, user_id)
    if not row or not database.delete_analysis(analysis_id, user_id):
        raise HTTPException(status_code=404, detail="Analysis not found")
    Path(row["image_path"]).unlink(missing_ok=True)
    return Response(status_code=204)


@app.get("/api/report/{analysis_id}")
def report(analysis_id: str, user_id: str = Depends(current_user)):
    row = database.get_analysis(analysis_id, user_id)
    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found")
    pdf = make_pdf(row)
    return Response(content=pdf, media_type="application/pdf",
                     headers={"Content-Disposition": f'attachment; filename="crop-assessment-{analysis_id}.pdf"'})
