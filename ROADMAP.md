# Ollama Catalog — Roadmap

## v0.1 — Extract & Catalog (current)
- [x] Extract scripts from spatial-llm (build_catalog.py, fetch_registry.py)
- [x] Complete Ollama API reference (docs/ollama-api-guide.md)
- [x] Model library (models/library.json — 197 models, 378 tags)
- [x] Seed catalog data (results/catalog.json — 392 models)
- [x] Seed registry blobs (results/registry_blobs.json — 378 entries)
- [x] Interactive HTML browser (results/catalog.html)
- [ ] Clean scripts — remove spatial-llm-specific references
- [ ] Update library.json with latest model families from ollama.com
- [ ] Re-run catalog generation with fresh data
- [ ] Add capability columns to HTML dashboard

## v0.2 — Diff & Filter
- [ ] Model diff: compare two catalog snapshots (new/removed/changed models)
- [ ] Tag completeness check: which families have missing size variants
- [ ] CLI filtering: `--filter "tools AND thinking" --min-params 3B`
- [ ] Capability matrix: HTML heatmap of models x capabilities

## v0.3 — Automation
- [ ] Scheduled re-probe of registry for new models
- [ ] Scrape ollama.com/library for model descriptions and popularity
- [ ] Track model version changes over time (changelog)

## Version History
| Version | Date | Summary |
|---------|------|---------|
| v0.1 | 2026-03-08 | Initial scaffold — extracted from spatial-llm |
