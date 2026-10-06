# Architecture

The React/Vite single-page client communicates with a FastAPI JSON and multipart API. The API validates credentials and images, delegates class prediction to a TensorFlow/Keras inference service, computes a separate heuristic visual indicator, persists analysis and account metadata through a repository interface, and renders report downloads from stored records. Users can maintain owner-scoped field records and optionally associate an analysis with one of their active fields. SQLite initialization adds the `fields` table and nullable `analyses.field_id` column without rewriting existing analysis payloads. MongoDB stores equivalent owner-scoped field documents. Field deletion archives the record; it does not remove analyses or uploaded images. MongoDB is the configured production-style option; SQLite supports local development without a server. Uploaded files are stored under a non-public uploads directory and accessed only through authenticated report routes.

Weather context is a separate optional path after image prediction. The backend resolves a field's approximate location, fetches normalized current conditions from Open-Meteo, and caches them in `weather_cache` (SQLite) or `weather_cache` (MongoDB) for a configurable period. Successful weather snapshots are embedded in the corresponding analysis payload; old analyses need no migration. Missing locations, timeouts, provider errors, and invalid responses produce an unavailable status without affecting inference or analysis persistence. Precise coordinates are rounded before external lookup, and API responses contain an administrative/approximate location label rather than coordinates. Weather does not imply disease causation or measured crop loss.

Production requires an explicit strong secret, HTTPS frontend origin, and MongoDB or an explicitly absolute SQLite path. For Render Free, the default upload directory is `/tmp/crop_analytics_uploads`; startup creates it automatically. MongoDB Atlas is the persistent store for database records, but Render Free filesystem uploads are ephemeral and may disappear after restart/redeploy. Grad-CAM handles missing images as unavailable, and PDF reports can still be regenerated from stored analysis data without the image. PDFs and Grad-CAM overlays are produced in memory rather than saved as files. This storage setup is suitable for controlled demonstrations, not permanent production file storage. The included classifier and label configuration are runtime files; training data is not required by inference.

Grad-CAM is an authenticated, on-demand explanation path. The API scopes the analysis lookup to the caller, confirms its saved image resolves under private upload storage, then reuses a cached graph tapping the backbone output in the saved model's original Functional graph. It verifies the graph's predicted class matches normal inference before generating the overlay; generated images are returned in memory and not persisted. Explanation errors do not affect prediction results, and the PDF path falls back to normal report content if explanation generation fails.

```text
Browser (React/Vite) -> FastAPI -> Auth / Analysis / Report services
                                  -> Keras MobileNetV2 classifier
                                  -> Image-based visual indicator heuristic
                                  -> owner-scoped fields and analysis metadata
                                  -> optional Open-Meteo context + short-lived cache
                                  -> repository: MongoDB or SQLite
                                  -> private upload storage
```

No user-facing metric is seeded or hard-coded. Dashboard totals are derived from persisted analyses.
