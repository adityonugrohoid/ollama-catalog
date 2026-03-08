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
models/library.json ──→ scripts/fetch_registry.py ──→ results/registry_blobs.json
                                                              │
cloud API (ollama.com) ──────────────────────────────────────→│
local API (localhost:11434) ─────────────────────────────────→│
                                                              ▼
                                                   scripts/build_catalog.py
                                                              │
                                                      ┌───────┴───────┐
                                                      ▼               ▼
                                                catalog.json    catalog.html
```

## Key Files
- `scripts/build_catalog.py` (875 lines) — Main catalog generator (cloud + local + registry → unified catalog + HTML)
- `scripts/fetch_registry.py` (264 lines) — Low-level OCI registry blob fetcher
- `docs/ollama-api-guide.md` (894 lines) — Complete Ollama API reference (CLI, REST, registry, capabilities)
- `models/library.json` — Model family + tag list (input for registry probing)
- `results/catalog.json` — Unified catalog output
- `results/registry_blobs.json` — Raw blob data from registry
- `results/catalog.html` — Interactive HTML browser

## Commands
```bash
# Generate catalog from all sources
python scripts/build_catalog.py

# Registry only (no Ollama instance needed)
python scripts/build_catalog.py --source registry

# Cloud + registry (no local Ollama needed)
python scripts/build_catalog.py --source cloud --source registry

# Fetch raw registry blobs
python scripts/fetch_registry.py
python scripts/fetch_registry.py --models gemma3:1b,llama3.2:1b
python scripts/fetch_registry.py --smallest
```

## Ollama API Details

### Cloud (`ollama.com`)
- `GET /api/tags` — list 32 curated models
- `POST /api/show` — model details (capabilities, parameters, model_info)
- Auth: `Authorization: Bearer $OLLAMA_API_KEY`

### Local (`localhost:11434`)
- Same endpoints as cloud, no auth
- Lists whatever is pulled locally

### OCI Registry (`registry.ollama.ai`)
- `GET /v2/library/{model}/manifests/{tag}` — OCI manifest (layer digests + sizes)
- `GET /v2/library/{model}/blobs/{digest}` — fetch blob (config, template, params, license)
- No auth, no model weight download
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

## Token Policy
Never limit tokens on any API call — no `max_tokens`, `num_ctx`, `num_predict`.
