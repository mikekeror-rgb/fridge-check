# Fridge Detector

A fine-tuned YOLOv8 object detector that scans a photo of a fridge or pantry, lists what it finds, and — optionally — flags which everyday essentials are missing. Runs entirely client-side in the browser via ONNX Runtime Web: no server, no upload, no network round trip after the model loads.

**Author:** Mike Keror ([github.com/mikekeror-rgb](https://github.com/mikekeror-rgb))

---

## What this is

This started as an open-ended "build a computer vision app" exercise and turned into a full, real data-engineering and ML project: building a labeled dataset from scratch, debugging a genuinely subtle training bug, iteratively improving the model on real-world photos, and shipping it as a working browser app.

## The 9 classes

`Apple` · `Banana` · `Orange` · `Tomato` · `Egg (Food)` · `Milk` · `Cheese` · `Bottle` · `Tin can`

Each class was capped at the same target count during data collection, to avoid a model that only ever learns the most common object in the dataset.

## Pipeline

```
Open Images V7 (via FiftyOne)
        ↓  balanced, per-class pull
Immich (self-hosted)
        ↓  canonical image store
YOLOv8n fine-tune (Ultralytics, Colab GPU)
        ↓  export
ONNX model
        ↓  runs on-device
Browser app (camera → detect → list → missing-item alert)
```

A second loop runs on top of the base pipeline to improve real-world accuracy:

```
Auto-label real photos with the current model
        ↓
Human correction (Roboflow) — fix wrong boxes, add missed ones
        ↓
Merge corrected examples into the training set
        ↓
Retrain from the last checkpoint
        ↺ repeat, each round targeting one specific confusion
```

## The debugging story

The most instructive part of this project: an early training run completed "successfully" (8 hours, 50 epochs, GPU-accelerated) but produced a model that detected almost nothing — **mAP50 of 0.006**, effectively random.

Root cause: the mapping from class name → class index (`Apple` = 0, `Banana` = 1, etc.) had been redefined multiple times across different script versions, in different orders. Label files written under one ordering were being reinterpreted under a different ordering later — so a box drawn around a bottle might be trained as "Orange," silently, with no error thrown anywhere in the pipeline.

The fix: one **canonical, hardcoded class list**, defined once and never reordered, used consistently across every data-pulling and label-writing script from then on.

Result: **mAP50 went from 0.006 to 0.519** — a ~90x improvement — from that single fix, before any additional data was even added.

## Iterative real-world correction

Even after the base model worked well on held-out validation data, it underperformed on real, cluttered, unevenly-lit fridge photos — a classic train/test domain gap. Specific, repeatable confusions were identified and fixed round by round:

| Round | Confusion found | Fix |
|---|---|---|
| v3–v4 | General recall gaps (missed real objects) | Auto-label + correct + retrain on real photos |
| v5 | Tomatoes labeled "Orange" | Targeted correction batch, tomato-heavy |
| v6 | Water bottles labeled "Tin can" | Targeted correction batch, bottle-heavy |

Both targeted confusions were measurably resolved by the following round — tomatoes now detect correctly at 0.66–0.86 confidence, and water bottles at 0.61–0.84, on real test photos.

## The app

A single self-contained `index.html` + `app.js`:

1. **Capture** — `navigator.mediaDevices.getUserMedia` opens the device camera in-browser
2. **Detect** — the exported ONNX model runs locally via `onnxruntime-web`, no server call
3. **List** — every detected item is shown with its confidence score
4. **Alert (optional)** — a checkbox compares detected items against a user-configurable "essentials" list and flags what's missing

### Running it locally

```bash
cd fridge-app
python3 -m http.server 8000
```

Open `http://localhost:8000` — camera access requires `localhost` or HTTPS; it will not work over `file://` or a plain non-localhost IP.

### Deploying

Push the folder to a GitHub repo and enable **GitHub Pages** for a free, permanent HTTPS URL that works on mobile browsers with no app store submission.

## Tech stack

| Stage | Tools |
|---|---|
| Data collection | FiftyOne, Open Images V7, Immich (self-hosted API) |
| Labeling & correction | Roboflow, model-assisted auto-labeling |
| Training | Ultralytics YOLOv8n, Google Colab (T4 / A100 GPU) |
| Deployment | ONNX Runtime Web, vanilla JS, GitHub Pages |

## What's next

**Region-specific groceries.** The 9 generic classes proved the pipeline works; a real product needs to recognize what people actually keep in their fridge, which varies by country and culture (e.g. sukuma wiki, ugali flour, chapati, tilapia, and other locally-specific packaging and staples). Same auto-label → correct → retrain loop, run against region-specific real photos.

**Smart fridge integration.** The long-term scaling idea: instead of a phone app pointed at an open fridge door, run the same model against a fridge's own built-in camera (e.g. Samsung Family Hub, LG InstaView), hosted in the manufacturer's cloud, with results accessible from a companion app anywhere — check what's in the fridge from the grocery store, no phone-to-fridge photo step required. The model and fine-tuning pipeline built here are not tied to any one manufacturer's platform.

## Known limitations

- Trained on a relatively small, balanced dataset (~2,500–6,000 images depending on round); production-grade accuracy would need substantially more real-world data per class
- Some classes (e.g. `Milk`) remain under-represented relative to others
- Detection on very cluttered, low-light, or heavily-occluded shelves is weaker than on clean, well-organized fridges
- Only 9 classes — no generic "unknown item" or packaged-goods catch-all yet
