# LayerWise

**An Explainable Transfer Learning Advisor for Image Classification**

LayerWise accepts a labeled image dataset, analyzes its properties, and produces a concrete, justified fine-tuning plan — which pretrained model to use, which layers to freeze, and which hyperparameters to start with — all explained in plain English with references to your actual data.

---

## Overview

LayerWise is a developer-facing advisory tool, not an automation system. It does not train a model on your behalf. It analyzes your dataset, runs a structured rule-based decision engine, and produces a ranked set of recommendations with full reasoning exposed at every step.

The core design principle separates LayerWise from AutoML tools: **every recommendation is traceable to a specific, auditable rule, and every explanation references the user's actual dataset statistics**. A recommendation without its reasoning is not a recommendation — it is a guess.

The system is built in three independent tiers: a pure Python ML engine, a FastAPI backend, and a React frontend. The engine can be fully developed, tested, and used via CLI before the API or frontend exist.

---

## Target Users

**Primary users** are ML practitioners past the tutorial stage but not yet expert: graduate students, domain experts (radiologists, biologists, remote sensing analysts), and hobbyist engineers who understand transfer learning conceptually but struggle to make confident decisions about model selection, freeze depth, and hyperparameter initialization. These practitioners know how to write a PyTorch training loop but not what learning rate to use, how many layers to freeze, or whether ResNet-50 or EfficientNet-B0 is appropriate for their specific dataset.

**Secondary users** are educators teaching applied deep learning who want a tool that makes transfer learning pedagogy concrete, and experienced engineers who want an explainable starting-point recommendation without doing it manually.

---

## The Problem Posed

Transfer learning is widely taught but poorly guided in practice. The conceptual advice — freeze early layers, fine-tune later ones, use a lower learning rate — is well understood. The operational decisions are not: exactly how many layers, with what learning rate, for what dataset size, under what domain similarity assumption?

Practitioners default to three failure modes: freezing everything (underfitting), unfreezing everything and training with a high learning rate (catastrophic forgetting), or copying code from a tutorial without understanding whether the configuration fits their data.

Existing tools resolve this in the wrong direction. AutoML tools automate the decision loop and hide the reasoning. If AutoML recommends EfficientNet-B4 and the model overfits on your 400-sample dataset, you have no diagnostic information, no understanding of what went wrong, and no intuition to apply to the next problem.

---

## Why LayerWise Exists

There is a specific, underserved gap between two extremes:

- **AutoML tools** (Google AutoML Vision, AutoKeras, AWS Rekognition): optimize the metric, hide every decision, produce a model you cannot learn from.
- **Manual expert intuition**: produces the best results, but requires a mentor or years of accumulated experience most practitioners do not have.

LayerWise occupies the space between them. It externalizes expert reasoning — the same reasoning a senior ML engineer would apply when advising a junior on dataset-to-model fit — into a structured, auditable rule engine with natural language explanations.

A user who runs LayerWise on five different datasets will develop genuine transfer learning intuition. A user who runs AutoKeras five times will not.

---

## Key Features

### Implemented (M1 — Dataset Analyzer)

- **Dataset validation and structural analysis**: accepts an ImageFolder-format directory, validates class folders, counts samples per class, detects corrupted files, and computes the imbalance ratio.
- **Image statistics**: computes per-channel pixel mean and standard deviation, grayscale ratio, median image dimensions, size variance, and estimated dataset size in MB using random sampling (500 images max) for speed.
- **DatasetProfile dataclass**: fully populated output with all statistics needed by downstream modules.
- **CLI entry point**: `python scripts/analyze_dataset.py --path ./my_dataset --output profile.json`
- **Fault tolerance**: corrupted files are listed, not silently skipped; empty class folders are handled with warnings.

### Planned (M2–M9)

- **Domain Detection** (M2): classify dataset into NATURAL, MEDICAL, SATELLITE, DOCUMENT, MICROSCOPY, or UNKNOWN using statistical rules applied to the DatasetProfile, with confidence scores and human-readable evidence signals.
- **Model Recommendation Engine** (M3): score six candidate architectures (ResNet-50, EfficientNet-B0/B3, MobileNetV2, DenseNet-121, ViT-B/16) against dataset and domain properties using a pluggable rule system; return all six ranked with fired rules and rejection reasons.
- **Freeze/Unfreeze Strategy Generator** (M4): map the top model and dataset characteristics to one of five cases in a domain-similarity × dataset-size matrix; output named layer groups, differential learning rate parameter groups, and copy-pasteable PyTorch optimizer code.
- **Hyperparameter Recommender** (M5): derive learning rate, batch size, epochs, weight decay, optimizer, scheduler, and augmentation preset using deterministic, logged formulas tied to dataset statistics.
- **Explanation Engine** (M6): render all decisions as natural language using Jinja2 templates that reference the user's actual dataset numbers — not generic advice.
- **FastAPI Backend** (M7): expose the full pipeline over HTTP with endpoints for upload, async analysis, status polling, and report retrieval.
- **React Frontend** (M8): single-page UI with drag-and-drop upload, real-time status polling, and a results dashboard including class distribution charts, layer freeze diagrams, model comparison cards, and expandable explanation accordions.
- **MVP Integration and Polish** (M9): end-to-end validation on three real public datasets, full README, pinned dependencies, and performance targets met.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        React Frontend                        │
│  Upload → Status Polling → Results Dashboard (Recharts)     │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP (JSON)
┌───────────────────────────▼─────────────────────────────────┐
│                        FastAPI Backend                       │
│  Routers → Services → pipeline.py → Engine Modules          │
└───────────────────────────┬─────────────────────────────────┘
                            │ Python imports
┌───────────────────────────▼─────────────────────────────────┐
│                      Python ML Engine                        │
│                                                              │
│  DatasetAnalyzer → DomainDetector → ModelRecommender        │
│       → FreezeStrategyGenerator → HyperparamRecommender     │
│                    → ExplanationEngine                       │
└─────────────────────────────────────────────────────────────┘
```

**Design invariant**: `engine/` is a pure Python library. It never imports from `api/` or `frontend/`. It can be run, tested, and validated entirely from the command line with no web server present. This separation means the entire decision engine — M1 through M6 — can be built and fully validated before a single line of FastAPI is written.

The API tier translates HTTP into engine calls and engine outputs into JSON. It does not contain business logic. Routers validate input and delegate to services; services call `pipeline.py` or specific engine functions. Nothing else.

---

## Project Structure

```
layerwise/
├── engine/          # Pure Python ML library — no web, no UI
│   ├── models/      # Shared dataclasses (DatasetProfile, FiredRule, etc.)
│   ├── analyzer/    # M1: DatasetAnalyzer
│   ├── detector/    # M2: DomainDetector
│   ├── recommender/ # M3: ModelRecommender + pluggable rules/
│   ├── strategy/    # M4: FreezeStrategyGenerator + block_maps.py
│   ├── hyperparams/ # M5: HyperparamRecommender + augmentation presets
│   ├── explainer/   # M6: ExplanationEngine + Jinja2 templates/
│   └── pipeline.py  # Orchestrates all modules end-to-end
│
├── api/             # FastAPI backend
│   ├── routers/     # Thin: validate input, call service, return result
│   ├── schemas/     # Pydantic HTTP request/response models
│   └── services/    # Business logic: job store, file extraction
│
├── frontend/        # React + Vite application
│   └── src/
│       ├── api/         # All fetch calls (no direct fetch in components)
│       ├── components/  # ui/ (atomic), results/, upload/, status/
│       ├── hooks/       # useUpload, useJobStatus, useReport
│       └── types/       # TypeScript interfaces mirroring api/schemas/
│
├── tests/
│   ├── unit/        # One module, no I/O, < 1s per file
│   ├── integration/ # Multi-module or HTTP via TestClient
│   └── e2e/         # Real datasets, run at milestone completion only
│
└── scripts/         # CLI entry points for each engine module
```

**Key constraints**:
- `engine/models/` contains only dataclasses and enums — no business logic.
- `block_maps.py` is pure data — no conditionals. Special handling belongs in `freeze_strategy.py`.
- All API calls in the frontend go through `frontend/src/api/`. No component calls `fetch()` directly.
- TypeScript types in `frontend/src/types/` mirror `api/schemas/` exactly.

---

## Methodology

### Test-Driven Development

Every module is built in a fixed sequence: define the output dataclass first, write all unit tests against the not-yet-implemented module, implement the module to make the tests pass, write the CLI script, then integrate into `pipeline.py`. Tests are written before implementation — they define the contract, not describe what was built.

### Rule-Based Decision Engine

The recommendation engine uses pluggable rule classes, each implementing two methods: `applies(profile, domain) -> bool` and `apply(scores, profile, domain) -> (scores, FiredRule)`. Rules are registered in a list and run sequentially. Adding a new rule means creating one class and adding it to the list — no other files change.

### FiredRule Logging

Every rule that fires — including rules that penalize a candidate — must produce a `FiredRule` log entry recording the rule ID, the condition that was met (e.g., `total_samples=400 < 500`), and the action taken (e.g., `vit_b_16 score -= 2`). This audit trail is the raw material for the Explanation Engine. The discipline is enforced from M2 onward.

### Deterministic Outputs

All hyperparameter formulas are deterministic and logged. There are no magic numbers. Learning rate is derived from freeze strategy case; batch size is clamped to the nearest power of two with a formula; epochs are computed as `max(20, 5000 // total_samples)` with a hard cap of 100. The same input always produces the same output.

### Explanation Quality Bar

Explanations must reference the user's actual dataset statistics. "Use a smaller batch size" is rejected. "We recommend batch size 8 for your 400-image dataset — using 32 would result in fewer than 15 gradient updates per epoch" is correct. A banned-phrase test runs against every explanation to enforce this: phrases like "consider using" or "it depends" fail the test.

---

## Progress & Roadmap

```
M1  [##########] Dataset Analyzer         ✅  COMPLETE  (Weeks 1–2)
M2  [          ] Domain Detector          ⬜  Planned   (Weeks 3–4)
M3  [          ] Model Recommender        ⬜  Planned   (Weeks 4–5)
M4  [          ] Freeze Strategy          ⬜  Planned   (Weeks 5–6)
M5  [          ] Hyperparameter Rec.      ⬜  Planned   (Weeks 6–7)
M6  [          ] Explanation Engine       ⬜  Planned   (Week 8)
M7  [          ] FastAPI Backend          ⬜  Planned   (Weeks 9–10)
M8  [          ] React Frontend           ⬜  Planned   (Weeks 11–12)
M9  [          ] MVP Integration          ⬜  Planned   (Week 13)
──────────────────────────────────────────────────────────────
M10 [          ] LLM Explanations + Q&A   ⬜  Phase 2   (Week 14)
M11 [          ] Grad-CAM + PDF Export    ⬜  Phase 2   (Weeks 15–16)
```

The MVP is considered complete at the end of M9. M10 and M11 are Phase 2 enhancements that can be deferred without affecting MVP functionality.

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+

### Setup

```bash
# 1. Clone the repository
git clone <repo-url> layerwise && cd layerwise

# 2. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

# 3. Install Python dependencies
pip install -e '.[dev]'
# or using requirements files:
# pip install -r requirements.txt -r requirements-dev.txt

# 4. Install frontend dependencies (when M8 is complete)
cd frontend && npm install && cd ..

# 5. Set environment variables
cp .env.example .env

# 6. Generate synthetic test fixtures
python scripts/generate_fixtures.py

# 7. Verify the installation
pytest tests/unit/ -q
```

---

## How to Use (Current State — M1)

The Dataset Analyzer is the only implemented module. It accepts a path to an ImageFolder-format directory and produces a fully populated `DatasetProfile` as JSON.

**Expected directory structure:**

```
my_dataset/
├── class_a/
│   ├── image_001.jpg
│   └── image_002.jpg
├── class_b/
│   └── image_003.png
└── class_c/
    └── ...
```

**Run the analyzer:**

```bash
python scripts/analyze_dataset.py --path ./my_dataset --output profile.json
```

**Example output:**

```json
{
  "n_classes": 3,
  "total_samples": 847,
  "samples_per_class": { "cat": 423, "dog": 400, "rabbit": 24 },
  "min_samples_per_class": 24,
  "max_samples_per_class": 423,
  "imbalance_ratio": 17.625,
  "median_image_size": [512, 512],
  "size_variance": 14.3,
  "grayscale_ratio": 0.04,
  "pixel_mean": [0.485, 0.456, 0.406],
  "pixel_std": [0.229, 0.224, 0.225],
  "estimated_mb": 112.4,
  "corrupted_files": []
}
```

A `imbalance_ratio` above 5.0 is flagged as severe. Corrupted files are listed by path, not silently skipped. Pixel statistics are computed on a random sample of up to 500 images.

---

## Testing

The test suite is divided into three layers with different scopes and run frequencies.

| Layer | Scope | Speed | When to Run |
|---|---|---|---|
| `tests/unit/` | One module, no I/O, no HTTP | < 1s per file | On every file save |
| `tests/integration/` | Multi-module or HTTP via TestClient | < 10s per file | Before every commit |
| `tests/e2e/` | Real datasets, full pipeline | Minutes | At milestone completion only |

```bash
# Unit tests only (fast — use during development)
pytest tests/unit/ -v

# Unit + integration (run before committing)
pytest tests/unit/ tests/integration/ -v

# Single test file
pytest tests/unit/test_dataset_analyzer.py -v

# Single test by name
pytest tests/unit/test_dataset_analyzer.py::test_imbalance_detection -v

# All tests excluding slow e2e
pytest -m 'not slow' -v

# Milestone completion check
pytest -m slow -v

# With coverage report
pytest tests/unit/ --cov=engine --cov-report=term-missing
```

All tests use deterministic, fixed inputs. Random data is never used in tests. Synthetic fixtures (under 50 images each) are committed to the repository so the full test suite runs on any machine without downloading datasets. Pre-built `DatasetProfile` JSON fixtures allow M2–M6 tests to run without first executing the analyzer.

---

## Design Philosophy

**Explainability over automation.** The system's value is not that it trains the best model for you. It is that it teaches you what a good model choice looks like for your data and explains why. A practitioner who uses LayerWise on five datasets will develop intuition that persists beyond the tool. A practitioner who uses an AutoML tool five times will not.

**The engine as a pure library.** The ML decision logic in `engine/` has no knowledge of HTTP, JSON serialization, or React. It is importable and testable with a single Python command and no server running. This is not an organizational nicety — it is a constraint that makes every module independently verifiable from day one.

**Rules over black-box models.** The decision engine uses explicit, auditable rules with logged conditions and actions. Every recommendation is traceable to the rule that produced it. When the system is eventually extended with a gradient-boosted meta-learner (Phase 3), SHAP values will maintain that traceability — transparency is not sacrificed for learned sophistication.

**The LLM narrates; the rules decide.** In Phase 2, Claude API integration will replace Jinja2 templates with richer natural language explanations and a follow-up Q&A panel. The LLM receives full decision context via system prompt and narrates decisions in conversational English. It does not make recommendations. The rule engine output remains the authoritative source of truth, independently auditable without any language model.

---

## Limitations

- **M1 only**: The dataset analyzer is the only implemented module. The domain detector, model recommender, freeze strategy generator, hyperparameter recommender, explanation engine, API, and frontend are planned but not yet built.
- **No UI or HTTP API**: All current interaction is via CLI script. The web interface begins at M8.
- **Rule-based system fails on novel inputs**: Thermal infrared, underwater photography, or unusual document scans will likely be misclassified by the domain detector. Evidence signals are always shown so users can evaluate and override.
- **Small dataset advice is speculative**: For datasets under 300 images, recommendations have high uncertainty. The system flags this explicitly rather than presenting false confidence.
- **No feedback loop in MVP**: The system cannot know whether recommendations led to good outcomes. Recommendations cannot improve without user-reported results. A feedback endpoint is planned post-M9.
- **Hyperparameter recommendations are starting points**: The system provides better-than-arbitrary defaults with clear reasoning. It does not claim to produce optimal configurations.
- **Block map requires manual maintenance**: Adding new model architectures requires editing `block_maps.py` manually.

---

## Tech Stack

### ML Engine
- **PyTorch >= 2.2** + **torchvision >= 0.17**: model loading, pretrained weights, explicit parameter groups for freeze/unfreeze strategies
- **Pillow >= 10.0**: image loading, validation, and statistics computation in the analyzer
- **NumPy >= 1.26**: array operations in dataset analysis
- **Jinja2 >= 3.1**: template-based explanation generation (Phase 2: replaced by Anthropic Claude API)
- **CLIP (openai/clip)**: domain detection via embedding comparison (Phase 2)
- **pytorch-grad-cam**: Grad-CAM heatmap generation for model interpretability (Phase 2)

### Backend
- **FastAPI >= 0.111**: async endpoints, automatic OpenAPI docs, clean Pydantic validation
- **Uvicorn**: ASGI server with hot reload
- **Pydantic >= 2.0** + **pydantic-settings**: request/response validation and environment configuration
- **SQLite + SQLAlchemy**: session storage and recommendation history (Phase 2)

### Frontend
- **React + TypeScript + Vite**: component-driven UI with fast hot reload
- **Tailwind CSS**: utility-first styling without custom CSS
- **Recharts**: class distribution bar charts and model comparison visualizations
- **TanStack React Query**: async state management and status polling

### Testing
- **pytest >= 8.0** + **pytest-cov**: test runner and coverage reporting
- **httpx**: async HTTP client for FastAPI TestClient
- **Ruff**: linting and formatting
- **mypy**: static type checking
- **Vitest** + **@testing-library/react**: frontend unit and component testing

---

## Future Work

**Phase 2 (M10–M11):**
- LLM-powered explanations via Claude API — richer, contextual natural language replacing Jinja2 templates, with a follow-up Q&A chat panel maintaining the invariant that the LLM narrates and the rule engine decides
- Grad-CAM heatmap visualization — upload trained weights and a test image, receive a heatmap overlay showing attended regions; answers "is my model looking at the right parts of the image?"
- Training log analyzer — accept epoch/loss CSV, diagnose overfitting or underfitting, recommend interventions
- PDF and Markdown report export — shareable document with all recommendations, statistics, and explanations

**Post-MVP research and extensions:**
- Feedback loop database: collect user-reported validation accuracy against job IDs; analyze which recommendations correlated with good outcomes
- CLIP-based dataset similarity matching: embed sample images and compare cosine similarity to reference dataset centroids (ImageNet, CheXpert, EuroSAT) to improve domain confidence
- Meta-learning model: train XGBoost on `(dataset_profile_features, hyperparam_config) → accuracy` triples; use SHAP values to maintain explanation transparency while adding learned sophistication
- LayerWise for Medical Imaging: DICOM support, domain-specific augmentation rules, NIH ChestX-ray14 / PatchCamelyon integration
- Hugging Face Space: expand model candidate pool beyond six hardcoded architectures by pulling from Hub model cards and dataset tags

---

*LayerWise is under active development. M1 is complete. Contributions, dataset tests, and rule suggestions are welcome.*