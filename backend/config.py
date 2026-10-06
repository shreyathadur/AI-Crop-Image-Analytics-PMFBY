import os
from pathlib import Path
from dotenv import dotenv_values, load_dotenv

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
load_dotenv(ENV_FILE)


def _default_upload_dir(app_env):
    if app_env == "production":
        return Path("/tmp/crop_analytics_uploads")
    return ROOT / "uploads"


def _load_secret_key():
    # load_dotenv preserves existing process variables, including an empty one.
    # Fall back to the file only when that process value is unset or blank.
    secret = os.getenv("SECRET_KEY", "").strip()
    if secret:
        return secret
    return (dotenv_values(ENV_FILE).get("SECRET_KEY") or "").strip()


APP_ENV = os.getenv("APP_ENV", "production").strip().lower()
SECRET_KEY = _load_secret_key()
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'crop_analytics.db'}")
MONGODB_URI = os.getenv("MONGODB_URI", "").strip()
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "crop_analytics")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(_default_upload_dir(APP_ENV)))).resolve()
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ROOT / "ml" / "models" / "crop_disease.keras"))).resolve()
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
RESET_TOKEN_EXPIRE_MINUTES = int(os.getenv("RESET_TOKEN_EXPIRE_MINUTES", "30"))
SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", "").strip()
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "CropLens").strip()
WEATHER_CACHE_MINUTES = int(os.getenv("WEATHER_CACHE_MINUTES", "30"))
WEATHER_TIMEOUT_SECONDS = float(os.getenv("WEATHER_TIMEOUT_SECONDS", "5"))
DISCLAIMER = "AI-assisted preliminary assessment — not an official PMFBY claim settlement or government decision."
