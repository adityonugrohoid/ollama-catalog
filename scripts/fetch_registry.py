"""Fetch raw Ollama model data from OCI registry — no parsing, full blobs.

For each model:tag, fetches:
1. OCI manifest (layer digests + sizes)
2. Config blob (architecture, model_family, model_type, file_type)
3. Template blob (full chat template text)
4. Params blob (runtime parameters like stop tokens)
5. License blob (license text)

Saves everything raw to results/registry_blobs.json for inspection.
The model weights blob is NOT fetched (too large) — only its size is recorded.

Usage:
    # Specific models
    python scripts/fetch_registry.py --models gemma3:1b,llama3.2:1b,ministral-3:3b

    # All models from library (reads models/library.json)
    python scripts/fetch_registry.py

    # All models, smallest tag per model
    python scripts/fetch_registry.py --smallest
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = "https://registry.ollama.ai"
LIBRARY_JSON = ROOT / "models" / "library.json"

console = Console(width=140)

# Layer media types
LAYER_MODEL = "application/vnd.ollama.image.model"
LAYER_TEMPLATE = "application/vnd.ollama.image.template"
LAYER_PARAMS = "application/vnd.ollama.image.params"
LAYER_LICENSE = "application/vnd.ollama.image.license"
CONFIG_TYPE = "application/vnd.docker.container.image.v1+json"


async def fetch_blob(
    client: httpx.AsyncClient,
    model: str,
    digest: str,
    semaphore: asyncio.Semaphore,
    is_json: bool = False,
) -> str | dict | None:
    """Fetch a blob by digest. Returns text or parsed JSON."""
    async with semaphore:
        url = f"{REGISTRY}/v2/library/{model}/blobs/{digest}"
        try:
            resp = await client.get(url, timeout=30)
            if resp.status_code == 200:
                if is_json:
                    try:
                        return resp.json()
                    except Exception:
                        return resp.text
                return resp.text
        except Exception as e:
            return f"ERROR: {e}"
    return None


async def probe_model(
    client: httpx.AsyncClient,
    model: str,
    tag: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Fetch manifest + all non-weight blobs for a model:tag."""
    entry: dict[str, Any] = {
        "model": model,
        "tag": tag,
        "full_name": f"{model}:{tag}",
    }

    # Fetch manifest
    async with semaphore:
        url = f"{REGISTRY}/v2/library/{model}/manifests/{tag}"
        try:
            resp = await client.get(url, timeout=15)
            if resp.status_code != 200:
                entry["error"] = f"manifest HTTP {resp.status_code}"
                return entry
            manifest = resp.json()
        except Exception as e:
            entry["error"] = str(e)
            return entry

    entry["manifest"] = manifest

    # Identify layers by type
    layer_map: dict[str, dict] = {}
    for layer in manifest.get("layers", []):
        mt = layer.get("mediaType", "")
        layer_map[mt] = layer

    # Record model weights size (don't download)
    if LAYER_MODEL in layer_map:
        entry["model_weights_size"] = layer_map[LAYER_MODEL].get("size", 0)
        entry["model_weights_digest"] = layer_map[LAYER_MODEL].get("digest", "")

    # Fetch config blob
    config_meta = manifest.get("config", {})
    if config_meta.get("digest"):
        entry["config"] = await fetch_blob(
            client, model, config_meta["digest"], semaphore, is_json=True
        )

    # Fetch template blob
    if LAYER_TEMPLATE in layer_map:
        entry["template"] = await fetch_blob(
            client, model, layer_map[LAYER_TEMPLATE]["digest"], semaphore
        )

    # Fetch params blob
    if LAYER_PARAMS in layer_map:
        entry["params"] = await fetch_blob(
            client, model, layer_map[LAYER_PARAMS]["digest"], semaphore
        )

    # Fetch license blob
    if LAYER_LICENSE in layer_map:
        entry["license"] = await fetch_blob(
            client, model, layer_map[LAYER_LICENSE]["digest"], semaphore
        )

    return entry


def load_targets(model_filter: list[str] | None, smallest: bool) -> list[tuple[str, str]]:
    """Build list of (model, tag) pairs to probe."""
    if model_filter:
        targets = []
        for spec in model_filter:
            if ":" in spec:
                m, t = spec.split(":", 1)
                targets.append((m, t))
            else:
                targets.append((spec, "latest"))
        return targets

    # Load from library JSON
    if not LIBRARY_JSON.exists():
        console.print(f"[red]Library file not found: {LIBRARY_JSON}[/]")
        console.print("Run with --models or create models/ollama_library.json first")
        raise SystemExit(1)

    library = json.loads(LIBRARY_JSON.read_text())
    targets = []
    for m in library["models"]:
        name = m["name"]
        tags = m.get("tags", ["latest"])
        if smallest:
            # Pick the smallest tag (first numeric one, or first)
            targets.append((name, tags[0]))
        else:
            # All tags
            for tag in tags:
                targets.append((name, tag))

    return targets


async def run(
    model_filter: list[str] | None = None,
    concurrency: int = 5,
    smallest: bool = False,
) -> dict[str, Any]:
    """Main runner."""
    targets = load_targets(model_filter, smallest)
    semaphore = asyncio.Semaphore(concurrency)

    console.print(f"[bold]Probing {len(targets)} model:tag pairs (concurrency={concurrency})...[/]")

    async with httpx.AsyncClient(
        follow_redirects=True,
        headers={"User-Agent": "ollama-catalog/1.0"},
    ) as client:
        tasks = [probe_model(client, model, tag, semaphore) for model, tag in targets]
        results = await asyncio.gather(*tasks)

    # Build output
    registry: dict[str, Any] = {
        "metadata": {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": REGISTRY,
            "total_probed": len(results),
            "total_success": sum(1 for r in results if "error" not in r),
            "total_errors": sum(1 for r in results if "error" in r),
        },
        "models": sorted(results, key=lambda r: r["full_name"]),
    }

    # Save
    out_path = ROOT / "results" / "registry_blobs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(registry, indent=2, default=str))
    console.print(f"[bold]Saved:[/] {out_path}")

    # Summary table
    table = Table(title=f"Ollama Registry Raw ({len(results)} entries)")
    table.add_column("Model", style="cyan", max_width=30)
    table.add_column("Weights", justify="right")
    table.add_column("Config", justify="right")
    table.add_column("Template", justify="right")
    table.add_column("Params", justify="right")
    table.add_column("License", justify="right")
    table.add_column("Status")

    for r in sorted(results, key=lambda x: x["full_name"]):
        weights = f"{r['model_weights_size'] / 1e6:.0f}MB" if "model_weights_size" in r else "-"
        config_len = str(len(json.dumps(r["config"]))) if "config" in r and r["config"] else "-"
        template_len = str(len(r["template"])) if "template" in r and r["template"] else "-"
        params_len = str(len(r["params"])) if "params" in r and r["params"] else "-"
        license_len = str(len(r["license"])) if "license" in r and r["license"] else "-"
        status = "OK" if "error" not in r else f"ERR: {r['error']}"
        table.add_row(r["full_name"], weights, config_len, template_len, params_len, license_len, status)

    console.print(table)

    return registry


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch raw Ollama model data from OCI registry (no parsing)"
    )
    parser.add_argument(
        "--models", default=None,
        help="Comma-separated model:tag list (default: all from ollama_library.json)"
    )
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Max parallel requests (default: 5)"
    )
    parser.add_argument(
        "--smallest", action="store_true",
        help="Only probe the smallest tag per model (default: all tags)"
    )
    args = parser.parse_args()

    model_filter = None
    if args.models:
        model_filter = [m.strip() for m in args.models.split(",")]

    asyncio.run(run(model_filter, args.concurrency, args.smallest))


if __name__ == "__main__":
    main()
