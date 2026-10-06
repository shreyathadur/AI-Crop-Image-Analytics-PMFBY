# Testing

Backend and ML helper tests use pytest. They cover health, registration/login, invalid credentials, image validation, inference-unavailable behavior, persistence, report generation, field CRUD and ownership, additive SQLite initialization, optional analysis association, weather normalization and failure cases, weather endpoint ownership, independent analysis behavior, weather-aware PDF output, and deterministic dataset preprocessing. Weather HTTP calls are mocked; no real provider request is made by the test suite. MongoDB field and weather-cache repository behavior has mock coverage; a live MongoDB service is not required by the suite.

## Grad-CAM verification

Grad-CAM tests use saved Corn Common Rust and healthy Corn test images to check that the generated explanation target index and label match normal inference and that generated PNG images have valid dimensions and finite pixels. A regression test simulates a different explanation-graph argmax and verifies generation is withheld. Other checks cover safe zero heatmaps, missing images, authenticated owner-scoped access, and PDF fallback. Grad-CAM uses the same predicted class as the existing inference pipeline. If the explanation graph cannot be verified against the prediction, the explanation is withheld rather than displaying a potentially misleading visualization. These checks cover generation and integration only; they do not establish scientific localization accuracy.

## Earlier verification run (2026-10-04)

- `python -m pytest -q`: **9 passed**. One Starlette deprecation warning notes the current httpx test client integration.
- `npm run build` in `frontend/`: **passed**; Vite produced a production bundle with the chart code split into a separate chunk.
- Scoped `python -m compileall -q ...`: **passed** for project Python modules. A broad compileall over the downloaded source tree also picked up two historical Python 2 scripts in the upstream repository; they are not imported by this application.
- Live backend smoke: health, register, login, and history succeeded over HTTP using SQLite.
- Live analysis integration with the trained model: upload returned 201; prediction was saved and appeared in history; report endpoint returned a PDF; delete returned 204.
- Live frontend server: Vite served the app at HTTP 200.
- ML: 3-epoch training completed; evaluation ran on 380 held-out examples. Metrics and artifacts are in `ml/results/`.

## Visual indicator verification

- `python -m pytest -q`: **19 passed**. Added coverage for the visual heuristic, healthy override, API response/history fields, and PDF severity fields/disclaimer.
- `npm run build` from `frontend/`: **passed**.
- Live API inference used a temporary Uvicorn process, temporary SQLite database, and temporary upload directory. It analyzed five existing images from `ml/data/test`, then confirmed history stored five rows and the PDF endpoint returned a report. The project database was not used or changed.

| Test image | Classifier result | Visual indicator | Category |
|---|---|---:|---|
| Cherry healthy | Cherry (including sour), healthy | 0.0% | Healthy |
| Corn Common Rust | Corn (maize), Common rust | 22.1% | Severe |
| Apple Scab | Apple, Apple scab | 0.1% | Low |
| Potato Late Blight | Potato, Late blight | 1.9% | Low |
| Tomato healthy | Tomato, healthy | 0.0% | Healthy |

These are outputs for five individual images, not measured severity accuracy. The visual indicator uses color-region heuristics without severity ground truth and is not agronomically validated or an official PMFBY loss percentage.

MongoDB was not installed/running in the environment; the SQLite fallback path was tested. No browser-driven visual interaction test was run.
