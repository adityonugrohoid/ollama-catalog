# Ollama Catalog

## Overview
Complete Ollama model catalog aggregated from three sources: Cloud API, OCI Registry, and Local instance. All registry inspection works **without downloading model weights** — only small metadata blobs are fetched.

## Tech Stack
- **Language**: Python 3.12+
- **HTTP**: httpx (async, with concurrency control)
- **CLI Output**: rich (tables, progress bars)
- **Config**: python-dotenv (.env)
- **Frontend**: Vanilla HTML/CSS/JS (static dashboard)
- No ML dependencies (no torch, transformers, etc.)

## Architecture
```
Step 1: Scrape model list
  ollama.com/library ──→ scrape_library.py ──→ models/library.json

Step 2: Build catalog (per-source output + merged catalog)
  models/library.json ────→ build_catalog.py (registry) ──→ results/registry.json
  ollama.com/api ─────────→ build_catalog.py (cloud)    ──→ results/cloud.json
  localhost:11434/api ────→ build_catalog.py (local)    ──→ results/local.json
                                      │
                              ┌───────┴───────┐
                              ▼               ▼
                        catalog.json    catalog.html
```

## Key Files
- `scripts/scrape_library.py` — Scrapes ollama.com/library to auto-build models/library.json
- `scripts/build_catalog.py` — Unified catalog generator (cloud + local + registry → per-source output + merged catalog + HTML)
- `docs/ollama-api-guide.md` (894 lines) — Complete Ollama API reference (CLI, REST, registry, capabilities)
- `models/library.json` — Auto-generated model family + tag list (input for registry probing)
- `results/cloud.json` — Raw cloud API responses (per-model /api/tags + /api/show)
- `results/local.json` — Raw local API responses (per-model /api/tags + /api/show)
- `results/registry.json` — Raw registry blob data (per-model manifest + config + template + params)
- `results/catalog.json` — Unified catalog (all sources merged)
- `results/catalog.html` — Interactive HTML browser

## Commands
```bash
# Step 1: Scrape model list from ollama.com/library
python scripts/scrape_library.py
python scripts/scrape_library.py --concurrency 5

# Step 2: Build catalog
python scripts/build_catalog.py                                          # All sources
python scripts/build_catalog.py --source registry                        # Registry only
python scripts/build_catalog.py --source cloud --source registry         # Cloud + registry
```

## Ollama API Details

### Cloud (`ollama.com`)
- `GET /api/tags` — list curated models
- `POST /api/show` — model details (capabilities, parameters, model_info)
- Auth: `Authorization: Bearer $OLLAMA_API_KEY`
- Output: `results/cloud.json` — raw tags_response + show_response per model

### Local (`localhost:11434`)
- Same endpoints as cloud, no auth
- Lists whatever is pulled locally
- Output: `results/local.json` — raw tags_response + show_response per model

### OCI Registry (`registry.ollama.ai`)
- `GET /v2/library/{model}/manifests/{tag}` — OCI manifest (layer digests + sizes)
- `GET /v2/library/{model}/blobs/{digest}` — fetch blob (config, template, params, license)
- No auth, no model weight download
- Output: `results/registry.json` — raw manifest + blobs per model
- Capability detection via template blob pattern matching:
  - Tools: `{{ .Tools }}`, `[AVAILABLE_TOOLS]`
  - Vision: `{{ .Images }}`, `image_url`
  - Thinking: `{{ .ThinkingEnabled }}`, `<think>`

### No-Download Constraint
Registry inspection extracts all metadata from small blobs (~100 B–5 KB). Model weights are **never downloaded** — only their size is recorded from the manifest. Fields NOT available without download: `context_length`, exact `parameter_count` (int).

## Key Patterns
- httpx async client with semaphore-based concurrency control
- Cloud auth via Bearer token in headers
- Registry blob fetching follows R2 redirects (`follow_redirects=True`)
- Deduplication: cloud/local models take priority over registry entries
- Capability detection from template blob patterns for registry models

## Token Policy
Never limit tokens on any API call — no `max_tokens`, `num_ctx`, `num_predict`.
