"""Scrape ollama.com/library to auto-build models/library.json.

Step 1 of the catalog pipeline:
  1. Fetch the library listing page — extract all model family names
  2. Fetch each model page — extract available tags
  3. Save to models/library.json

Usage:
    python scripts/scrape_library.py
    python scripts/scrape_library.py --concurrency 5
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
LIBRARY_URL = "https://ollama.com/library"
OUTPUT = ROOT / "models" / "library.json"

console = Console(width=140)


# ---------------------------------------------------------------------------
# HTML parsing helpers
# ---------------------------------------------------------------------------


def parse_model_names(html: str) -> list[str]:
    """Extract model family names from the library listing page.

    Matches <a href="/library/model-name"> links (without colons = no tag).
    """
    pattern = r'href="/library/([a-zA-Z0-9][a-zA-Z0-9._-]*)"'
    matches = re.findall(pattern, html)
    # Deduplicate preserving order, exclude any with colons (tag links)
    seen: set[str] = set()
    names: list[str] = []
    for m in matches:
        if ":" not in m and m not in seen:
            seen.add(m)
            names.append(m)
    return names


def parse_model_tags(html: str, model_name: str) -> list[str]:
    """Extract tag names from a model page.

    Matches <a href="/library/model:tag"> links.
    """
    escaped = re.escape(model_name)
    pattern = rf'href="/library/{escaped}:([^"]+)"'
    matches = re.findall(pattern, html)
    # Deduplicate preserving order
    seen: set[str] = set()
    tags: list[str] = []
    for t in matches:
        if t not in seen:
            seen.add(t)
            tags.append(t)
    return tags


# ---------------------------------------------------------------------------
# Async scraping
# ---------------------------------------------------------------------------


async def scrape_model_page(
    client: httpx.AsyncClient,
    model_name: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Fetch a single model page and extract its tags."""
    async with semaphore:
        try:
            resp = await client.get(
                f"{LIBRARY_URL}/{model_name}",
                timeout=20,
            )
            if resp.status_code != 200:
                return {
                    "name": model_name,
                    "tags": [],
                    "error": f"HTTP {resp.status_code}",
                }
            tags = parse_model_tags(resp.text, model_name)
            return {"name": model_name, "tags": tags}
        except Exception as e:
            return {"name": model_name, "tags": [], "error": str(e)}


async def run(concurrency: int = 10) -> dict[str, Any]:
    """Main scraping pipeline."""
    console.rule("[bold cyan]Ollama Library Scraper[/]")
    console.print()

    async with httpx.AsyncClient(
        follow_redirects=True,
        headers={"User-Agent": "ollama-catalog/1.0"},
    ) as client:
        # --- Step 1: Fetch library listing page ---
        console.print("[bold]Step 1:[/] Fetching ollama.com/library...")
        resp = await client.get(LIBRARY_URL, timeout=30)
        resp.raise_for_status()

        model_names = parse_model_names(resp.text)
        console.print(f"  Found [bold]{len(model_names)}[/] model families\n")

        # --- Step 2: Scrape each model page for tags ---
        console.print(
            f"[bold]Step 2:[/] Scraping model pages for tags "
            f"(concurrency={concurrency})..."
        )
        semaphore = asyncio.Semaphore(concurrency)
        tasks = [
            scrape_model_page(client, name, semaphore)
            for name in model_names
        ]
        results = await asyncio.gather(*tasks)

    # --- Build output ---
    models: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    total_tags = 0

    for r in results:
        if "error" in r:
            errors.append(r)
            console.print(f"  [red]✗[/] {r['name']}: {r['error']}")
        else:
            tag_count = len(r["tags"])
            total_tags += tag_count
            tag_preview = ", ".join(r["tags"][:6])
            if len(r["tags"]) > 6:
                tag_preview += f" (+{len(r['tags']) - 6} more)"
            console.print(
                f"  [green]✓[/] {r['name']}: "
                f"[bold]{tag_count}[/] tags — {tag_preview}"
            )
        models.append({"name": r["name"], "tags": r["tags"]})

    library = {
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "source": LIBRARY_URL,
        "total_models": len(models),
        "total_tags": total_tags,
        "models": sorted(models, key=lambda m: m["name"]),
    }

    # --- Save ---
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(library, indent=2))

    # --- Summary ---
    console.print()
    console.rule("[bold cyan]Summary[/]")

    table = Table(title=f"Library — {len(models)} models, {total_tags} tags")
    table.add_column("#", style="dim", justify="right", width=4)
    table.add_column("Model", style="cyan", max_width=30)
    table.add_column("Tags", justify="right")
    table.add_column("Tag List", style="dim", max_width=80)

    for i, m in enumerate(sorted(models, key=lambda x: x["name"]), 1):
        tag_list = ", ".join(m["tags"][:8])
        if len(m["tags"]) > 8:
            tag_list += f" (+{len(m['tags']) - 8})"
        table.add_row(str(i), m["name"], str(len(m["tags"])), tag_list)

    console.print(table)

    if errors:
        console.print(f"\n[yellow]Errors: {len(errors)}[/]")
        for e in errors:
            console.print(f"  {e['name']}: {e['error']}")

    console.print(f"\n[bold]Saved:[/] {OUTPUT}")
    console.print(
        f"  {library['total_models']} models, "
        f"{library['total_tags']} tags"
    )

    return library


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape ollama.com/library to build models/library.json"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max parallel requests (default: 10)",
    )
    args = parser.parse_args()

    asyncio.run(run(args.concurrency))


if __name__ == "__main__":
    main()
