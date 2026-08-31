# Edge Vision Road Anomaly & Pothole Severity Index — SIH Project

On-device pothole detection, severity scoring, and geo-tagged heatmap system.

## Repo layout

- `ml/` — detection, segmentation, depth, severity, tracking (this is the ML component only — it does **not** do GPS, database, or heatmap generation)
- `backend/` — GPS association, event storage, heatmap generation, API
- `app/` — mobile/dashcam-facing app (camera capture, GPS, upload)
- `docs/` — architecture diagrams, problem statement, notes

## Architecture (high level)

```
Camera/Video → [ml/] Detect+Segment → Track → Depth → Severity
                                                          │
                                        {track_id, class, severity, area, timestamp}
                                                          │
                                                          ▼
                              [backend/] GPS association → DB → Heatmap
```

See `ml/README.md` for how to run the ML pipeline.
