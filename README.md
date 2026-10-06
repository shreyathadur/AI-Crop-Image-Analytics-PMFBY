# AI Based Real-Time Crop Image Analytics for Crop Insurance – PMFBY

CropLens is a full-stack student project that classifies supported crop leaf images and stores preliminary observations. The result is decision support only. It is not an official PMFBY claim settlement, government decision, field loss survey, or policy eligibility assessment.

## Project overview and objectives

The application combines a React dashboard, FastAPI service, Keras image classifier, and persisted user analysis history. It aims to make supported crop/disease labels easier to review and keep a traceable record of submitted leaf observations.

## Features

- Account registration and login with salted PBKDF2 password hashes and signed bearer tokens
- Single-use password reset links, with SMTP delivery or a clearly marked local development link
- Owner-scoped field records with optional location details and analysis grouping
- Optional current weather context for analyses assigned to a located field
- Validated JPEG, PNG, and WebP uploads with a 10 MB default limit
- Lazy MobileNetV2 inference, confidence score, crop and health/disease class
- Analysis history, live dashboard counts, and generated PDF reports
- MongoDB support when configured, with SQLite for local development
- Dataset preparation, model training, held-out evaluation, and CLI inference scripts
- Clear disclosure when a trained model is absent; no synthetic/fake predictions

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md), [REQUIREMENTS.md](REQUIREMENTS.md), and [API.md](API.md). At runtime the browser calls FastAPI; analysis invokes the saved Keras model and writes records through the configured repository. Uploads are placed in a private server directory and are never returned as file paths.

## Technology stack

React, Vite, Recharts, plain CSS; Python 3.12 for deployment (the current local environment also works on 3.13), FastAPI, Pydantic, Uvicorn; TensorFlow/Keras MobileNetV2, Pillow, NumPy, scikit-learn; SQLite by default and optional MongoDB; pytest and ReportLab.

## Dataset

PlantVillage is the selected public research dataset: 38 crop-disease/healthy categories covering 14 species. Release counts differ slightly: the original paper reports 54,306 images and TensorFlow Datasets' unaugmented republished release lists 54,303. Details and limitations are in [DATASET.md](DATASET.md). Training images and generated splits are local-only and ignored by Git. The approximately 10 MB trained classifier and `model_config.json` are runtime artifacts and are included so inference works from a source checkout without training. Verify the dataset release terms before redistribution or commercial use.

## ML methodology and model architecture

The pipeline uses deterministic stratified splits, MobileNetV2 preprocessing, training-only augmentation, and transfer learning with a compact pretrained backbone. Evaluation outputs are created by `ml/evaluate.py` only after a real training run. No project metric is claimed in this repository until those outputs exist. See [ML_MODEL.md](ML_MODEL.md).

Measured local run: balanced subset of 3,800 images (100 per class), 3,040 train / 380 validation / 380 test, seed 42, 3 epochs. Held-out subset metrics: accuracy **85.53%**, macro precision **0.8643**, macro recall **0.8553**, macro F1 **0.8509**. Ten examples per class is a small test sample; the result only reflects this controlled split and does not establish field performance.

Severity labels and insurance damage percentages are absent from PlantVillage. The API therefore reports a separate, documented color-based **Estimated Visual Damage Indicator**, with Healthy/Low/Moderate/Severe categories. It is an AI-assisted visual heuristic, not a validated severity measure or official PMFBY claim percentage. See [ML_MODEL.md](ML_MODEL.md) for method, thresholds, and limitations.

## Installation and environment

Use Python 3.11–3.13 and Node.js 22.12+ (the current Vite build requires Node 20.19+ or 22.12+). Python 3.12 is selected for Render via `.python-version`. MongoDB is optional for local development; SQLite requires no service. In PowerShell from the project root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and provide a private random `SECRET_KEY` (at least 32 random characters). Defaults use SQLite and local private uploads. To use MongoDB, set `MONGODB_URI` and `MONGODB_DATABASE`; a configured Mongo URI takes precedence over SQLite. Configure network access and credentials in the URI through the environment, never source code. `UPLOAD_DIR`, `MODEL_PATH`, `MAX_UPLOAD_MB`, and `FRONTEND_ORIGIN` may also be changed.

### Password reset and Gmail SMTP

Password reset works locally without mail setup when the copied `.env` sets `APP_ENV=development` (the example does). After requesting a reset, the API response includes a link marked `DEVELOPMENT ONLY`; this link is intentionally available only in development. The code defaults to `APP_ENV=production`, validates a strong secret and HTTPS frontend origin at startup, and never includes reset tokens in production responses. Without SMTP, production reset requests return the generic account-safe message but no email can be delivered.

To enable automatic email delivery, fill in these values in the project-root `.env` (never commit that file):

```dotenv
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=youraddress@gmail.com
SMTP_PASSWORD=your-google-app-password
SMTP_FROM_EMAIL=youraddress@gmail.com
SMTP_FROM_NAME=CropLens
FRONTEND_ORIGIN=http://localhost:5173
RESET_TOKEN_EXPIRE_MINUTES=30
```

For Gmail, enable 2-Step Verification and create a Google **App Password** for this application. Use that generated app password as `SMTP_PASSWORD`; do not use your regular Google account password. Keep it private in `.env`. Port 587 uses STARTTLS. When the SMTP connection settings are complete, reset requests send email automatically. Restart the backend after editing `.env`; check the server log if delivery fails. Never configure reset links for an origin you do not control.

The API loads `.env` from the project root at startup. For a one-session PowerShell configuration instead, set values directly:

```powershell
$env:SECRET_KEY = (python -c "import secrets; print(secrets.token_urlsafe(48))")
$env:DATABASE_URL = 'sqlite:///./crop_analytics.db'
$env:UPLOAD_DIR = './uploads'
$env:MODEL_PATH = './ml/models/crop_disease.keras'
```

### Weather and environmental context

When an analysis is assigned to a field, the backend can add current environmental context from [Open-Meteo](https://open-meteo.com/en/docs). Open-Meteo's forecast and geocoding endpoints do not require an API key, so no weather credential is needed and no provider key is sent to React. Configure these optional server settings in `.env`:

```dotenv
WEATHER_CACHE_MINUTES=30
WEATHER_TIMEOUT_SECONDS=5
```

Conditions include the provider observation timestamp, approximate field location, temperature, relative humidity, precipitation, wind speed, and mapped condition. The backend reuses a recent result for the same rounded location for up to `WEATHER_CACHE_MINUTES`; each successful analysis retains its own weather snapshot and observation time. If coordinates are unavailable, the backend can use the field's village/district/state through Open-Meteo geocoding. Coordinates are rounded before an external request and are not returned by the weather API. Weather lookup failures, missing location, and timeout show “Weather information currently unavailable” and never prevent image prediction. Historical analyses without weather remain valid.

Weather is contextual data, not evidence that weather caused disease, not a measure of crop loss, and not a determination of PMFBY eligibility or compensation. The PDF includes environmental information only when a successful snapshot exists, along with this limitation.

## Dataset preparation

Download the PlantVillage release referenced in [DATASET.md](DATASET.md), then place images in class-named subdirectories under `ml/data/raw` (or use the documented TensorFlow Datasets source and export the images). Do not mix generated or augmented derivatives into the validation/test splits.

```powershell
python -m ml.download_dataset
python -m ml.prepare_dataset
```

For a smaller balanced experiment, pass `python -m ml.prepare_dataset --max-per-class 300`; each class is sampled with the fixed seed and all three splits remain stratified.

The command checks image decoding, reports corrupt files and class counts, then writes fixed-seed train/validation/test folders.

## Model training and evaluation

```powershell
python -m ml.train --epochs 12 --batch-size 24
python -m ml.evaluate
python -m ml.predict path\to\leaf.jpg
```

First training downloads ImageNet MobileNetV2 weights and may take substantial time. Training requires network access for those weights unless available in the Keras cache. Actual metrics, confusion matrix, and training history are saved under `ml/results/`; the model/config are saved under `ml/models/`.

## Running backend and frontend

In one PowerShell terminal, with environment variables configured:

```powershell
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`; FastAPI docs are at `http://127.0.0.1:8000/docs`. Set `VITE_API_BASE_URL` before the frontend build if the API uses another URL (the older `VITE_API_URL` name remains a fallback). The application can register, authenticate, and show its dashboard before model artifacts exist; analysis requests return an explicit 503 until the saved model/config are present. Restart Uvicorn after changing backend code or `.env`. Authenticated analysis uploads return the visual indicator and severity fields, which are retained in history and included in PDF reports.

## Production deployment on Render

The backend service root is the repository root. Configure Python from `.python-version` (3.12):

```text
Build Command: pip install -r requirements.txt
Start Command: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

Set these backend environment variables in Render (store `SECRET_KEY` and database credentials as secrets):

```text
APP_ENV=production
SECRET_KEY=<at least 32 random characters>
FRONTEND_ORIGIN=https://<your-frontend-name>.onrender.com
MONGODB_URI=<your MongoDB connection URI>
MONGODB_DATABASE=crop_analytics
UPLOAD_DIR=/tmp/crop_analytics_uploads
```

This Render Free configuration uses the ephemeral filesystem and does not use a persistent disk. When `APP_ENV=production` and `UPLOAD_DIR` is unset, the backend defaults to `/tmp/crop_analytics_uploads` and creates it at startup. MongoDB Atlas stores account, analysis, field, and history records persistently. Uploaded crop images live on the Render filesystem and may disappear after a restart or redeploy; Grad-CAM then returns an unavailable/not-found response for those images, while the saved analysis history remains available. PDF reports are generated on demand in memory and are not stored as files; after an image is lost, a report can still be generated from the saved analysis but will omit the image and Grad-CAM visualization. This setup is suitable for a controlled college/demo deployment, not permanent production file storage. Keep `MONGODB_URI` configured to use Atlas for the persistent database.

The saved model and label config are included in the source checkout and do not require training data at runtime. For the frontend, create a Render Static Site rooted at `frontend`:

```text
Build Command: npm ci && npm run build
Publish Directory: dist
Environment: VITE_API_BASE_URL=https://<your-backend-name>.onrender.com
```

Set `FRONTEND_ORIGIN` to exactly the frontend origin (scheme and hostname, no path). Set `VITE_API_BASE_URL` to exactly the backend origin; it is public configuration, never a place for credentials. SMTP settings are optional for startup, but production password reset email requires valid SMTP settings. Open-Meteo's free API is for non-commercial use and requires attribution; commercial use needs an appropriate provider plan. See [Open-Meteo terms](https://open-meteo.com/en/terms).

## Database schema

`users` contains id, normalized email, salted password hash, and creation time. `fields` contains owner-scoped crop/field details and optional location coordinates. `analyses.field_id` is nullable so existing and unassigned analyses remain valid. `password_reset_tokens` stores only SHA-256 token hashes, user ID, expiry, use state, and creation time. Analyses retain their JSON payload, private image path, and existing fields; the API omits user IDs and server paths from returned analysis objects. Deleting a field archives it and leaves its analyses and reports available.

Field coordinates are optional and are visible only to the owning account through authenticated field endpoints. Avoid entering precise location details unless they are needed for your records. Archiving preserves field details (including any coordinates) and existing analyses, but removes the field from the active list and prevents selecting it for new analyses. Clear optional location values before archiving if they should not be retained.

## Testing and API documentation

```powershell
python -m pytest
cd frontend
npm run build
```

See [TESTING.md](TESTING.md) and [API.md](API.md). No command result is claimed until run in this environment. The API provides interactive OpenAPI documentation at `/docs`.

## AI Model Explanation (Grad-CAM)

The result page can generate a Grad-CAM view on demand, showing the original image beside an overlay for the class already predicted by the classifier. Grad-CAM taps the saved MobileNetV2 backbone's spatial output from the original model graph and checks that its predicted class matches normal inference before generating an overlay. The API checks analysis ownership before reading the private image. The PDF report may include the visualization when available.

Grad-CAM provides a qualitative explanation of image regions that influenced the model prediction. It is not a validated disease segmentation map or crop-loss measurement. Highlighted regions are not exact lesion boundaries, damage percentages, or proof that a region contains disease. A healthy-class visualization relates to the model's healthy prediction and does not imply damage.

## Screenshots

Screenshots can be added here after running the frontend and capturing the dashboard and analysis flow.

## Limitations

PlantVillage contains controlled leaf photographs and does not represent field-level scenes, whole-plant damage, every Indian crop, or weather/event evidence. Accuracy on a held-out split does not prove field generalization. Confidence is an uncalibrated model score. Severity, yield loss, cause, policy eligibility, and claim percentages are not measured. Human agronomic review is required.

## Future scope

Evaluate on geographically diverse field images; collect expert-reviewed severity labels; calibrate scores; support additional crops and languages; add consent-aware metadata and audit workflows; conduct security and deployment reviews before any operational use.

## Disclaimer

**AI-assisted preliminary assessment — not an official PMFBY claim settlement or government decision.** The software is a student demonstration and must not replace official crop-insurance or agricultural procedures.
