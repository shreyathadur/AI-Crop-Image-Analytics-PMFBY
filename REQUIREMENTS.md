# Requirements

## Problem and objectives

Farmers and field surveyors need an accessible way to record crop-leaf observations. This project classifies supported leaf images and provides a traceable, preliminary decision-support assessment. It does not estimate PMFBY entitlement or adjudicate claims.

## Users

- Farmers submitting an observation
- Field surveyors reviewing and tracking observations
- Project administrators managing the local demonstration

## Functional requirements

1. Register and authenticate users with hashed passwords.
2. Accept supported image files with size and image-decode validation.
3. Classify supported crop and leaf health/disease labels and display confidence.
4. Explain the prediction, its preliminary nature, and model limitations.
5. Record analysis history and provide a dashboard based only on stored analyses.
6. Provide a downloadable report for an analysis.
7. Support MongoDB when configured and a local SQLite development store otherwise.
8. Return useful errors for invalid uploads, missing model, database, and authentication failures.

## Non-functional requirements

Responsive interface, accessible labels, deterministic preprocessing, reproducible splits and seeds, safe file handling, no plaintext passwords or embedded secrets, maintainable separation of API, persistence, and ML inference, and Windows-friendly setup.

## System workflow

The user signs in, uploads a leaf image, and receives a validated prediction. The API persists metadata and a private image reference. The dashboard and history query persisted analyses; reports are generated from the stored record. All result views carry the preliminary-assessment disclaimer.

## ML workflow

Obtain the documented PlantVillage dataset; validate and split by class with a fixed seed; resize to the model input and normalize according to the selected backbone; augment training samples only; fine-tune a lightweight pretrained image classifier; evaluate on the held-out test split; save labels, preprocessing configuration, training history, and metrics. The inference API reports uncertainty and rejects unavailable model state instead of fabricating results.

Severity is a conservative visual heuristic and must be marked as an uncalibrated visual indicator. The dataset has no insurance loss-percentage labels. The app therefore does not claim to measure insured damage percentages or claim outcomes.

## Technology choices

React + Vite + plain CSS for the client; Python 3.13, FastAPI, Pydantic, Uvicorn, TensorFlow/Keras, Pillow, NumPy, scikit-learn, and pytest for the service and ML pipeline; MongoDB through a repository abstraction, with SQLite for local operation when MongoDB is unavailable.

## Limitations

PlantVillage is a controlled-background leaf dataset, not a field survey or crop-loss dataset. It does not represent all crops grown in India, whole-plant or field-level damage, weather/event causality, or policy terms. Predictions can be wrong and require qualified human review. No performance metric will be published until measured by this project's evaluation run.

## Future improvements

Evaluate on independently collected field images; add crop and region coverage with expert-reviewed labels; calibrate confidence and severity against validated agronomic observations; add consent-based survey metadata, audit trails, role management, deployment hardening, and integrations only after legal and domain review.
