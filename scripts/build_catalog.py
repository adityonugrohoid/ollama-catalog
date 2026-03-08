"""Generate a complete Ollama model catalog from cloud, local, and registry.

Three sources:
  - cloud:    Ollama Cloud (ollama.com) — 32 curated models, full metadata via /api/show
  - local:    Ollama Local (localhost:11434) — whatever is pulled, full metadata via /api/show
  - registry: OCI Registry (registry.ollama.ai) — ALL 200+ models from library.json,
              probes each model:tag for capabilities via manifest + template blob (no download)

Registry source reads models/library.json for the model list and curated size tags,
then probes the OCI registry at registry.ollama.ai to fetch config and template blobs.

Outputs:
  results/catalog.json   — full structured data
  results/catalog.html   — visual HTML dashboard

Usage:
    # All three sources (default)
    python scripts/build_catalog.py

    # Cloud + registry (no local Ollama needed)
    python scripts/build_catalog.py --source cloud --source registry

    # Registry only (complete library, no Ollama needed at all)
    python scripts/build_catalog.py --source registry

    # Cloud only
    python scripts/build_catalog.py --source cloud
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

LOCAL_URL = os.getenv("OLLAMA_LOCAL_HOST", "http://localhost:11434")
CLOUD_URL = os.getenv("OLLAMA_CLOUD_HOST", "https://ollama.com")
CLOUD_KEY = os.getenv("OLLAMA_API_KEY", "")
REGISTRY_URL = "https://registry.ollama.ai"
LIBRARY_JSON = ROOT / "models" / "library.json"

console = Console(width=140)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_size(size_bytes: int | None) -> str:
    if not size_bytes:
        return "—"
    gb = size_bytes / 1_000_000_000
    if gb >= 1:
        return f"{gb:.1f} GB"
    mb = size_bytes / 1_000_000
    return f"{mb:.0f} MB"


def format_param_count(count: int | None) -> str:
    if not count:
        return "—"
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.1f}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.0f}M"
    return str(count)


# ---------------------------------------------------------------------------
# API helpers — cloud & local (via /api/tags + /api/show)
# ---------------------------------------------------------------------------


async def fetch_model_list(
    client: httpx.AsyncClient,
    base_url: str,
    headers: dict[str, str],
    source: str,
) -> list[dict[str, Any]]:
    """GET /api/tags to list all models on an instance."""
    try:
        resp = await client.get(f"{base_url}/api/tags", headers=headers)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        console.print(f"  [green]{source}:[/] found {len(models)} models")
        return models
    except httpx.ConnectError:
        console.print(f"  [yellow]{source}:[/] connection refused (is Ollama running?)")
        return []
    except Exception as e:
        console.print(f"  [red]{source}:[/] error fetching model list: {e}")
        return []


async def fetch_model_details(
    client: httpx.AsyncClient,
    model_name: str,
    base_url: str,
    headers: dict[str, str],
) -> dict[str, Any] | None:
    """POST /api/show to get detailed model metadata."""
    try:
        resp = await client.post(
            f"{base_url}/api/show",
            headers=headers,
            json={"model": model_name},
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Registry helpers — OCI manifest + template blob probing
# ---------------------------------------------------------------------------


LAYER_MODEL = "application/vnd.ollama.image.model"
LAYER_TEMPLATE = "application/vnd.ollama.image.template"
LAYER_PARAMS = "application/vnd.ollama.image.params"
LAYER_LICENSE = "application/vnd.ollama.image.license"
LAYER_PROJECTOR = "application/vnd.ollama.image.projector"


def load_library_json() -> list[dict[str, Any]]:
    """Load model list from models/ollama_library.json."""
    if not LIBRARY_JSON.exists():
        console.print(f"  [red]registry:[/] {LIBRARY_JSON} not found")
        console.print("  Run scripts/fetch_ollama_registry.py or create it manually")
        return []
    data = json.loads(LIBRARY_JSON.read_text())
    models = data.get("models", [])
    total_tags = sum(len(m.get("tags", ["latest"])) for m in models)
    console.print(
        f"  [green]registry:[/] loaded {len(models)} models "
        f"({total_tags} model:tag pairs) from {LIBRARY_JSON.name}"
    )
    return models


async def probe_registry_manifest(
    client: httpx.AsyncClient, model: str, tag: str
) -> dict[str, Any] | None:
    """Fetch OCI manifest from registry.ollama.ai."""
    try:
        resp = await client.get(
            f"{REGISTRY_URL}/v2/library/{model}/manifests/{tag}"
        )
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


async def fetch_registry_blob(
    client: httpx.AsyncClient, model: str, digest: str
) -> str:
    """Fetch a blob from the registry (follows R2 redirect)."""
    try:
        resp = await client.get(
            f"{REGISTRY_URL}/v2/library/{model}/blobs/{digest}",
            follow_redirects=True,
        )
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    return ""


async def probe_model_from_registry(
    client: httpx.AsyncClient,
    model: str,
    tag: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Probe a model:tag from OCI registry. Returns (catalog_entry, raw_blob) or None."""
    manifest = await probe_registry_manifest(client, model, tag)
    if not manifest:
        return None

    layers = manifest.get("layers", [])

    # Identify layer digests by type
    weight_size = 0
    weight_digest = ""
    projector_size = 0
    projector_digest = ""
    layer_digests: dict[str, str] = {}

    for layer in layers:
        media = layer.get("mediaType", "")
        if media == LAYER_MODEL:
            weight_size += layer.get("size", 0)
            weight_digest = layer.get("digest", "")
        elif media == LAYER_PROJECTOR:
            projector_size += layer.get("size", 0)
            projector_digest = layer.get("digest", "")
        elif media in (LAYER_TEMPLATE, LAYER_PARAMS, LAYER_LICENSE):
            layer_digests[media] = layer.get("digest", "")

    # Config is in manifest.config, not in layers
    config_digest = manifest.get("config", {}).get("digest", "")

    # Fetch all blobs concurrently
    blob_coros: dict[str, Any] = {}
    if layer_digests.get(LAYER_TEMPLATE):
        blob_coros["template"] = fetch_registry_blob(client, model, layer_digests[LAYER_TEMPLATE])
    if config_digest:
        blob_coros["config"] = fetch_registry_blob(client, model, config_digest)
    if layer_digests.get(LAYER_PARAMS):
        blob_coros["params"] = fetch_registry_blob(client, model, layer_digests[LAYER_PARAMS])
    if layer_digests.get(LAYER_LICENSE):
        blob_coros["license"] = fetch_registry_blob(client, model, layer_digests[LAYER_LICENSE])

    blobs: dict[str, str] = {}
    if blob_coros:
        keys = list(blob_coros.keys())
        values = await asyncio.gather(*blob_coros.values())
        blobs = dict(zip(keys, values))

    template = blobs.get("template", "")
    config_text = blobs.get("config", "")
    params_text = blobs.get("params", "")
    license_text = blobs.get("license", "")

    # Parse config for model family, type, quantization
    family = ""
    families = None
    model_type = ""
    file_type = ""
    config_parsed = None
    if config_text:
        try:
            config_parsed = json.loads(config_text)
            family = config_parsed.get("model_family", "")
            families = config_parsed.get("model_families")
            model_type = config_parsed.get("model_type", "")
            file_type = config_parsed.get("file_type", "")
        except (json.JSONDecodeError, AttributeError):
            pass

    name = f"{model}:{tag}" if tag != "latest" else model

    # Catalog entry — structured fields + raw template, no parsed capabilities
    catalog_entry = {
        "name": name,
        "source": "registry",
        "size_bytes": weight_size,
        "digest": config_digest,
        "modified_at": "",
        "family": family,
        "families": families,
        "parameter_size": model_type,
        "quantization_level": file_type,
        "format": "gguf",
        "template": template,
    }

    # Raw blob entry — all data for internal reference
    raw_blob = {
        "name": name,
        "model": model,
        "tag": tag,
        "manifest": manifest,
        "config": config_parsed if config_parsed else config_text,
        "template": template,
        "params": params_text,
        "license": license_text,
        "model_weights_size": weight_size,
        "model_weights_digest": weight_digest,
    }
    if projector_size:
        raw_blob["projector_size"] = projector_size
        raw_blob["projector_digest"] = projector_digest

    return (catalog_entry, raw_blob)


# ---------------------------------------------------------------------------
# Data extraction (for cloud/local /api/show responses)
# ---------------------------------------------------------------------------


def extract_arch_fields(model_info: dict[str, Any]) -> dict[str, Any]:
    """Pull architecture-specific fields from model_info."""
    arch = model_info.get("general.architecture", "")
    return {
        "architecture": arch,
        "parameter_count": model_info.get("general.parameter_count"),
        "context_length": model_info.get(f"{arch}.context_length"),
        "embedding_length": model_info.get(f"{arch}.embedding_length"),
    }


def build_entry(
    tag_info: dict[str, Any],
    show_info: dict[str, Any] | None,
    source: str,
) -> dict[str, Any]:
    """Build a unified catalog entry from /api/tags + /api/show data."""
    details = tag_info.get("details", {})
    entry: dict[str, Any] = {
        "name": tag_info.get("name", tag_info.get("model", "")),
        "source": source,
        "size_bytes": tag_info.get("size"),
        "digest": tag_info.get("digest", ""),
        "modified_at": tag_info.get("modified_at", ""),
        "family": details.get("family", ""),
        "families": details.get("families"),
        "parameter_size": details.get("parameter_size", ""),
        "quantization_level": details.get("quantization_level", ""),
        "format": details.get("format", ""),
    }

    if show_info:
        entry["capabilities"] = show_info.get("capabilities", ["completion"])
        entry["system_prompt"] = (show_info.get("system", "") or "")[:200]
        entry["parameters"] = show_info.get("parameters", "")

        mi = show_info.get("model_info", {})
        if mi:
            arch = extract_arch_fields(mi)
            entry["architecture"] = arch["architecture"]
            entry["parameter_count"] = arch["parameter_count"]
            entry["context_length"] = arch["context_length"]
            entry["embedding_length"] = arch["embedding_length"]

        show_details = show_info.get("details", {})
        if show_details:
            if not entry["family"]:
                entry["family"] = show_details.get("family", "")
            if not entry["parameter_size"]:
                entry["parameter_size"] = show_details.get("parameter_size", "")
            if not entry["quantization_level"]:
                entry["quantization_level"] = show_details.get("quantization_level", "")
    else:
        entry["capabilities"] = ["completion"]

    return entry


# ---------------------------------------------------------------------------
# Catalog generation
# ---------------------------------------------------------------------------


CONCURRENCY_LIMIT = 10


async def generate_catalog(sources: list[str]) -> dict[str, Any]:
    """Fetch model lists and details from specified sources."""
    catalog: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {},
        "models": [],
    }

    seen: set[str] = set()

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        # --- Cloud and Local sources (via /api/tags + /api/show) ---
        for source in sources:
            if source == "registry":
                continue  # handled separately below
            if source == "cloud":
                if not CLOUD_KEY:
                    console.print("  [red]cloud:[/] OLLAMA_API_KEY not set, skipping")
                    continue
                base_url = CLOUD_URL
                headers = {
                    "Authorization": f"Bearer {CLOUD_KEY}",
                    "Content-Type": "application/json",
                }
            else:
                base_url = LOCAL_URL
                headers = {"Content-Type": "application/json"}

            models = await fetch_model_list(client, base_url, headers, source)
            catalog["sources"][source] = {
                "url": base_url,
                "model_count": len(models),
            }

            tasks = []
            for m in models:
                name = m.get("name", m.get("model", ""))
                tasks.append((name, m, fetch_model_details(client, name, base_url, headers)))

            for name, tag_info, coro in tasks:
                show_info = await coro
                entry = build_entry(tag_info, show_info, source)
                seen.add(name)
                catalog["models"].append(entry)
                caps = ", ".join(entry.get("capabilities", []))
                ps = entry.get("parameter_size", "?")
                ps_display = format_param_count(int(ps)) if ps and str(ps).isdigit() else ps
                console.print(f"    [dim]{name}[/] — {ps_display} — \\[{caps}]")

        # --- Registry source (OCI manifest + template blob probing) ---
        if "registry" in sources:
            library_models = load_library_json()
            catalog["sources"]["registry"] = {
                "url": REGISTRY_URL,
                "library_file": str(LIBRARY_JSON),
                "model_count": 0,
            }

            if library_models:
                # Build list of (model, tag, all_tags) to probe
                probe_list: list[tuple[str, str, list[str]]] = []
                for m in library_models:
                    name = m["name"]
                    tags = m.get("tags", ["latest"])
                    for tag in tags:
                        probe_list.append((name, tag, tags))

                console.print(f"  [dim]Probing {len(probe_list)} model:tag pairs...[/]")

                sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

                async def probe_one(
                    model_name: str, tag: str, all_tags: list[str]
                ) -> tuple[dict[str, Any], dict[str, Any]] | None:
                    async with sem:
                        result = await probe_model_from_registry(client, model_name, tag)
                        if result:
                            catalog_entry, raw_blob = result
                            catalog_entry["available_tags"] = all_tags
                            return (catalog_entry, raw_blob)
                        return None

                # Run all probes concurrently (bounded by semaphore)
                probe_tasks = [
                    probe_one(name, tag, tags)
                    for name, tag, tags in probe_list
                ]
                results = await asyncio.gather(*probe_tasks, return_exceptions=True)

                registry_count = 0
                raw_blobs: list[dict[str, Any]] = []
                for result in results:
                    if isinstance(result, Exception) or result is None:
                        continue
                    catalog_entry, raw_blob = result
                    raw_blobs.append(raw_blob)
                    name = catalog_entry["name"]
                    # If already seen from cloud/local, mark cross-availability
                    if name in seen:
                        for existing in catalog["models"]:
                            if existing["name"] == name:
                                if "also_available_on" not in existing:
                                    existing["also_available_on"] = "registry"
                                break
                        continue  # don't duplicate

                    seen.add(name)
                    catalog["models"].append(catalog_entry)
                    registry_count += 1
                    ps = catalog_entry.get("parameter_size", "?")
                    console.print(f"    [dim]{name}[/] — {ps or '?'}")

                catalog["sources"]["registry"]["model_count"] = registry_count
                console.print(f"  [green]registry:[/] {registry_count} new models added (not on cloud/local)")

                # Save raw blobs to separate file
                blobs_path = ROOT / "results" / "registry_blobs.json"
                blobs_data = {
                    "metadata": {
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                        "source": REGISTRY_URL,
                        "library_file": str(LIBRARY_JSON),
                        "total_probed": len(probe_list),
                        "total_success": len(raw_blobs),
                    },
                    "models": sorted(raw_blobs, key=lambda r: r["name"]),
                }
                blobs_path.write_text(json.dumps(blobs_data, indent=2, default=str))
                console.print(f"  [bold]Raw blobs:[/] {blobs_path} ({len(raw_blobs)} entries)")

    # Sort by parameter count (descending), then name
    def parse_size(s: str) -> float:
        """Parse '3.2B', '1B', '70B', '480M' etc. to a float for sorting."""
        if not s:
            return 0
        s = s.strip().upper()
        try:
            if s.endswith("B"):
                return float(s[:-1]) * 1e9
            if s.endswith("M"):
                return float(s[:-1]) * 1e6
            if s.endswith("K"):
                return float(s[:-1]) * 1e3
            return float(s)
        except ValueError:
            return 0

    def sort_key(m: dict) -> tuple:
        pc = m.get("parameter_count") or parse_size(m.get("parameter_size", ""))
        return (-pc, m.get("name", ""))

    catalog["models"].sort(key=sort_key)
    catalog["total_models"] = len(catalog["models"])

    return catalog


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------


def generate_html(catalog: dict[str, Any]) -> str:
    models = catalog["models"]

    # Stats
    total = len(models)
    cloud_count = sum(1 for m in models if m["source"] == "cloud")
    local_count = sum(1 for m in models if m["source"] == "local")
    registry_count = sum(1 for m in models if m["source"] == "registry")

    # Build table rows
    rows = []
    for i, m in enumerate(models):
        source = m["source"]
        source_class = source
        source_label = source
        if m.get("also_available_on"):
            source_label += f' + {m["also_available_on"]}'
            source_class = "both"

        ctx = m.get("context_length")
        ctx_str = f"{ctx:,}" if ctx else "—"

        ps = m.get("parameter_size", "") or format_param_count(m.get("parameter_count"))

        tags_count = len(m.get("available_tags", []))
        tags_str = str(tags_count) if tags_count else "—"

        rows.append(f"""        <tr>
          <td class="rank">{i + 1}</td>
          <td class="model-name">{m['name']}</td>
          <td><span class="source-tag {source_class}">{source_label}</span></td>
          <td>{m.get('family', '') or m.get('architecture', '') or '—'}</td>
          <td class="num">{ps or '—'}</td>
          <td class="num">{format_param_count(m.get('parameter_count'))}</td>
          <td class="num">{format_size(m.get('size_bytes'))}</td>
          <td class="num">{ctx_str}</td>
          <td class="num">{tags_str}</td>
        </tr>""")

    rows_html = "\n".join(rows)
    generated = catalog["generated_at"][:19].replace("T", " ") + " UTC"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ollama Model Catalog</title>
<style>
  :root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2a2d3a;
    --text: #e2e4e9;
    --dim: #8b8fa3;
    --accent: #6366f1;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', monospace;
    background: var(--bg);
    color: var(--text);
    padding: 2rem;
    line-height: 1.5;
  }}
  h1 {{
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
    color: #fff;
  }}
  .subtitle {{
    color: var(--dim);
    font-size: 0.8rem;
    margin-bottom: 1.5rem;
  }}
  .stats {{
    display: flex;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
  }}
  .stat {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem 1.25rem;
    min-width: 120px;
  }}
  .stat-value {{
    font-size: 1.5rem;
    font-weight: 700;
    color: #fff;
  }}
  .stat-label {{
    font-size: 0.7rem;
    color: var(--dim);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .filter-bar {{
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
    align-items: center;
  }}
  .filter-bar input {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.5rem 0.75rem;
    color: var(--text);
    font-family: inherit;
    font-size: 0.8rem;
    width: 280px;
  }}
  .filter-bar input:focus {{
    outline: none;
    border-color: var(--accent);
  }}
  .filter-btn {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.5rem 0.75rem;
    color: var(--dim);
    font-family: inherit;
    font-size: 0.75rem;
    cursor: pointer;
    transition: all 0.15s;
  }}
  .filter-btn:hover, .filter-btn.active {{
    border-color: var(--accent);
    color: #fff;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
  }}
  thead th {{
    text-align: left;
    padding: 0.6rem 0.75rem;
    border-bottom: 2px solid var(--border);
    color: var(--dim);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
  }}
  thead th:hover {{
    color: #fff;
  }}
  tbody tr {{
    border-bottom: 1px solid var(--border);
    transition: background 0.1s;
  }}
  tbody tr:hover {{
    background: rgba(99, 102, 241, 0.06);
  }}
  td {{
    padding: 0.5rem 0.75rem;
    white-space: nowrap;
  }}
  .rank {{
    color: var(--dim);
    width: 2rem;
  }}
  .model-name {{
    font-weight: 600;
    color: #fff;
  }}
  .num {{
    text-align: right;
    font-variant-numeric: tabular-nums;
  }}
  .source-tag {{
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 600;
  }}
  .source-tag.cloud {{
    background: rgba(37, 99, 235, 0.15);
    color: #60a5fa;
    border: 1px solid rgba(37, 99, 235, 0.3);
  }}
  .source-tag.local {{
    background: rgba(34, 197, 94, 0.15);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.3);
  }}
  .source-tag.registry {{
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.3);
  }}
  .source-tag.both {{
    background: rgba(124, 58, 237, 0.15);
    color: #a78bfa;
    border: 1px solid rgba(124, 58, 237, 0.3);
  }}
  .table-wrap {{
    overflow-x: auto;
    border: 1px solid var(--border);
    border-radius: 8px;
  }}
</style>
</head>
<body>

<h1>Ollama Model Catalog</h1>
<p class="subtitle">Generated {generated} &mdash; Cloud ({cloud_count}) + Local ({local_count}) + Registry ({registry_count}) &mdash; {total} models total</p>

<div class="stats">
  <div class="stat"><div class="stat-value">{total}</div><div class="stat-label">Total Models</div></div>
  <div class="stat"><div class="stat-value">{cloud_count}</div><div class="stat-label">Cloud</div></div>
  <div class="stat"><div class="stat-value">{local_count}</div><div class="stat-label">Local</div></div>
  <div class="stat"><div class="stat-value">{registry_count}</div><div class="stat-label">Registry</div></div>
</div>

<div class="filter-bar">
  <input type="text" id="search" placeholder="Filter by name or family...">
  <button class="filter-btn" onclick="filterSource('all')" id="btn-all">All</button>
  <button class="filter-btn" onclick="filterSource('cloud')" id="btn-cloud">Cloud</button>
  <button class="filter-btn" onclick="filterSource('local')" id="btn-local">Local</button>
  <button class="filter-btn" onclick="filterSource('registry')" id="btn-registry">Registry</button>
</div>

<div class="table-wrap">
<table id="catalog">
  <thead>
    <tr>
      <th>#</th>
      <th>Model</th>
      <th>Source</th>
      <th>Family</th>
      <th>Params</th>
      <th>Exact</th>
      <th>Size</th>
      <th>Context</th>
      <th>Tags</th>
    </tr>
  </thead>
  <tbody>
{rows_html}
  </tbody>
</table>
</div>

<script>
const rows = document.querySelectorAll('#catalog tbody tr');
let activeSource = 'all';

document.getElementById('search').addEventListener('input', applyFilters);

function filterSource(src) {{
  activeSource = activeSource === src ? 'all' : src;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('btn-' + activeSource).classList.add('active');
  applyFilters();
}}

function applyFilters() {{
  const q = document.getElementById('search').value.toLowerCase();
  rows.forEach(row => {{
    const text = row.textContent.toLowerCase();
    const matchText = !q || text.includes(q);
    const src = row.children[2].textContent.trim();
    const matchSrc = activeSource === 'all' || src.includes(activeSource);
    row.style.display = (matchText && matchSrc) ? '' : 'none';
  }});
}}

document.getElementById('btn-all').classList.add('active');
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def main_async(sources: list[str]) -> None:
    console.rule("[bold cyan]Ollama Model Catalog Generator[/]")
    console.print(f"Sources: {', '.join(sources)}")
    console.print()

    catalog = await generate_catalog(sources)

    # Save JSON
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "catalog.json"
    json_path.write_text(json.dumps(catalog, indent=2, default=str))
    console.print(f"\n[bold]JSON:[/] {json_path}")

    # Save HTML
    html_path = out_dir / "catalog.html"
    html_path.write_text(generate_html(catalog))
    console.print(f"[bold]HTML:[/] {html_path}")

    # Summary table
    console.print()
    table = Table(title=f"Catalog Summary — {catalog['total_models']} models")
    table.add_column("Source")
    table.add_column("Models", justify="right")
    table.add_column("URL")
    for src, info in catalog["sources"].items():
        table.add_row(src, str(info["model_count"]), info["url"])
    console.print(table)

    cap_counts: dict[str, int] = {}
    for m in catalog["models"]:
        for c in m.get("capabilities", []):
            cap_counts[c] = cap_counts.get(c, 0) + 1

    table2 = Table(title="Capabilities Breakdown")
    table2.add_column("Capability")
    table2.add_column("Count", justify="right")
    for cap, count in sorted(cap_counts.items(), key=lambda x: -x[1]):
        table2.add_row(cap, str(count))
    console.print(table2)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate Ollama model catalog")
    parser.add_argument(
        "--source",
        action="append",
        choices=["cloud", "local", "registry"],
        help="Sources to query (repeatable, default: cloud + local + registry)",
    )
    args = parser.parse_args()

    sources = args.source or ["cloud", "local", "registry"]
    asyncio.run(main_async(sources))


if __name__ == "__main__":
    main()
