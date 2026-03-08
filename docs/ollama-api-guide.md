# Ollama Complete API & CLI Guide

Complete reference for interacting with Ollama models across all three access methods: **CLI**, **REST API** (local + cloud), and **OCI Registry** (capability discovery).

---

## Table of Contents

1. [Access Methods Overview](#access-methods-overview)
2. [CLI Commands](#cli-commands)
3. [REST API — Model Management](#rest-api--model-management)
4. [REST API — Inference](#rest-api--inference)
5. [OpenAI-Compatible Endpoint](#openai-compatible-endpoint)
6. [Cloud API (`ollama.com`)](#cloud-api-ollamacom)
7. [OCI Registry — Capability Discovery](#oci-registry--capability-discovery)
8. [Capability Reference](#capability-reference)
9. [Cloud vs Local Differences](#cloud-vs-local-differences)

---

## Access Methods Overview

| Method | URL | Auth | Use Case |
|--------|-----|------|----------|
| **CLI** | n/a (local binary) | None | Pull, run, manage models interactively |
| **Local API** | `http://localhost:11434` | None | Programmatic inference & management |
| **Cloud API** | `https://ollama.com` | `Bearer $OLLAMA_API_KEY` | Remote inference, 32 curated models |
| **OCI Registry** | `https://registry.ollama.ai` | None | Capability discovery via manifests |

---

## CLI Commands

### Installation

```bash
# Linux / WSL2
curl -fsSL https://ollama.com/install.sh | sh

# Start the server (runs on port 11434)
ollama serve
```

### Model Lifecycle

```bash
# Pull a model from the registry
ollama pull llama3.2:1b

# List locally downloaded models
ollama list

# Show model details (template, parameters, license)
ollama show llama3.2:1b

# Show specific sections
ollama show llama3.2:1b --modelfile    # Full Modelfile
ollama show llama3.2:1b --template     # Chat template only
ollama show llama3.2:1b --parameters   # Runtime parameters
ollama show llama3.2:1b --license      # License text
ollama show llama3.2:1b --system       # System prompt

# Run model interactively (pulls if not local)
ollama run llama3.2:1b

# Run with an initial prompt (non-interactive)
ollama run llama3.2:1b "What is 2+2?"

# List currently running/loaded models
ollama ps

# Stop a running model (free VRAM)
ollama stop llama3.2:1b

# Copy a model under a new name
ollama cp llama3.2:1b my-custom-model

# Delete a local model
ollama rm llama3.2:1b
```

### Model Creation (Custom Models)

```bash
# Create from an existing model with custom system prompt
ollama create my-assistant --from llama3.2:1b

# Create with a Modelfile
ollama create my-model -f ./Modelfile

# Quantize during creation
ollama create my-model-q4 --from llama3.2:1b --quantize q4_K_M
```

### Push to Registry (Requires Account)

```bash
ollama push username/my-model
```

---

## REST API — Model Management

Base URL: `http://localhost:11434` (local) or `https://ollama.com` (cloud, requires auth)

### GET /api/tags — List Models

Returns all models available on the instance.

```bash
# Local
curl http://localhost:11434/api/tags

# Cloud (32 curated models)
curl -H "Authorization: Bearer $OLLAMA_API_KEY" https://ollama.com/api/tags
```

**Response:**
```json
{
  "models": [
    {
      "name": "llama3.2:1b",
      "model": "llama3.2:1b",
      "modified_at": "2025-12-02T00:00:00Z",
      "size": 1300000000,
      "digest": "sha256:abc123...",
      "details": {
        "parent_model": "",
        "format": "gguf",
        "family": "llama",
        "families": ["llama"],
        "parameter_size": "1B",
        "quantization_level": "Q4_0"
      }
    }
  ]
}
```

### POST /api/show — Model Details

Returns detailed metadata including template, parameters, capabilities, and architecture info.

```bash
# Local
curl http://localhost:11434/api/show -d '{"model": "llama3.2:1b"}'

# Cloud
curl -H "Authorization: Bearer $OLLAMA_API_KEY" \
     https://ollama.com/api/show -d '{"model": "ministral-3:3b"}'
```

**Request body:**
```json
{
  "model": "ministral-3:3b"
}
```

**Response fields:**

| Field | Type | Description |
|-------|------|-------------|
| `modelfile` | string | Complete Modelfile (FROM, TEMPLATE, SYSTEM, PARAMETER directives) |
| `template` | string | Chat template (Go template syntax with `{{ .System }}`, `{{ .Prompt }}`, tool blocks) |
| `parameters` | string | Runtime parameters (temperature, top_p, stop tokens, etc.) |
| `system` | string | Default system prompt |
| `license` | string | License text |
| `details` | object | `parent_model`, `format`, `family`, `families`, `parameter_size`, `quantization_level` |
| `model_info` | object | Architecture details: `general.architecture`, `general.parameter_count`, `{arch}.context_length`, `{arch}.embedding_length` |
| `capabilities` | array | **Key field** — `["completion", "tools", "vision", "thinking"]` |

**Example response (abbreviated):**
```json
{
  "modelfile": "FROM ministral-3:3b\nTEMPLATE ...\nPARAMETER stop ...",
  "template": "{{- range $i, $_ := .Messages }}...",
  "parameters": "stop [INST]\nstop [/INST]",
  "system": "",
  "license": "Apache 2.0 ...",
  "details": {
    "parent_model": "ministral-3:3b",
    "format": "",
    "family": "mistral3",
    "families": null,
    "parameter_size": "3000000000",
    "quantization_level": "FP8"
  },
  "model_info": {
    "general.architecture": "mistral3",
    "general.parameter_count": 3000000000,
    "mistral3.context_length": 131072,
    "mistral3.embedding_length": 2560
  },
  "capabilities": ["completion", "tools", "vision"]
}
```

### POST /api/pull — Pull Model

```bash
curl http://localhost:11434/api/pull -d '{"model": "llama3.2:1b", "stream": true}'
```

**Request:**
```json
{
  "name": "llama3.2:1b",
  "stream": true
}
```

**Streamed response (progress updates):**
```json
{"status": "pulling manifest"}
{"status": "downloading sha256:abc123...", "total": 1300000000, "completed": 650000000}
{"status": "success"}
```

### POST /api/create — Create Model

```json
{
  "model": "my-spatial-model",
  "from": "llama3.2:1b",
  "system": "You are a spatial reasoning assistant.",
  "parameters": {"temperature": 0},
  "quantize": "q4_K_M",
  "adapters": {"path/to/lora": 1.0}
}
```

### POST /api/copy — Copy Model

```json
{
  "source": "llama3.2:1b",
  "destination": "my-backup"
}
```

### DELETE /api/delete — Delete Model

```json
{
  "model": "my-backup"
}
```

### POST /api/push — Push Model

```json
{
  "model": "username/my-model",
  "stream": true
}
```

### GET /api/ps — Running Models

```bash
curl http://localhost:11434/api/ps
```

**Response:**
```json
{
  "models": [
    {
      "name": "llama3.2:1b",
      "model": "llama3.2:1b",
      "size": 1300000000,
      "digest": "sha256:abc123...",
      "expires_at": "2025-03-07T12:00:00Z"
    }
  ]
}
```

---

## REST API — Inference

### POST /api/chat — Chat Completion (Native)

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2:1b",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": false
}'
```

**Response:**
```json
{
  "model": "llama3.2:1b",
  "created_at": "2025-03-07T12:00:00Z",
  "message": {
    "role": "assistant",
    "content": "Hello! How can I help you?"
  },
  "done": true,
  "done_reason": "stop",
  "total_duration": 1234567890,
  "prompt_eval_count": 26,
  "eval_count": 12
}
```

**Non-streaming response metrics (local only):**

| Field | Description |
|-------|-------------|
| `done_reason` | `"stop"` (natural end) or `"length"` (token limit hit) |
| `total_duration` | Total time in nanoseconds |
| `load_duration` | Time to load model (null on cloud) |
| `prompt_eval_count` | Prompt token count |
| `prompt_eval_duration` | Prompt processing time (null on cloud) |
| `eval_count` | Completion token count |
| `eval_duration` | Generation time (null on cloud) |

### POST /api/chat — With Tool Calling

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "ministral-3:3b",
  "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
  "stream": false,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
          "type": "object",
          "required": ["city"],
          "properties": {
            "city": {"type": "string", "description": "City name"}
          }
        }
      }
    }
  ]
}'
```

**Response with tool call:**
```json
{
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "type": "function",
        "function": {
          "index": 0,
          "name": "get_weather",
          "arguments": {"city": "Paris"}
        }
      }
    ]
  },
  "done": true
}
```

**Multi-turn tool flow (send tool result back):**
```json
{
  "model": "ministral-3:3b",
  "messages": [
    {"role": "user", "content": "What is the weather in Paris?"},
    {"role": "assistant", "tool_calls": [{"type": "function", "function": {"index": 0, "name": "get_weather", "arguments": {"city": "Paris"}}}]},
    {"role": "tool", "tool_name": "get_weather", "content": "22°C, sunny"}
  ],
  "stream": false
}
```

### POST /api/chat — Structured JSON Output

```json
{
  "model": "llama3.2:1b",
  "messages": [{"role": "user", "content": "Tell me about France."}],
  "stream": false,
  "format": {
    "type": "object",
    "properties": {
      "name": {"type": "string"},
      "capital": {"type": "string"},
      "languages": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["name", "capital", "languages"]
  }
}
```

The `format` field accepts a full JSON schema (not just `{"type": "json_object"}`). Ollama uses constrained decoding to enforce the schema.

### POST /api/chat — With Thinking

```json
{
  "model": "deepseek-v3.2",
  "messages": [{"role": "user", "content": "Solve step by step: 15 * 23"}],
  "stream": false,
  "think": true
}
```

**Response:**
```json
{
  "message": {
    "role": "assistant",
    "content": "The answer is 345.",
    "thinking": "Let me multiply 15 by 23...\n15 × 20 = 300\n15 × 3 = 45\n300 + 45 = 345"
  },
  "done": true
}
```

### POST /api/generate — Text Completion

For non-chat, raw text completion:

```json
{
  "model": "llama3.2:1b",
  "prompt": "The capital of France is",
  "stream": false
}
```

### POST /api/embed — Embeddings

```json
{
  "model": "llama3.2:1b",
  "input": ["Hello world", "Spatial reasoning"]
}
```

---

## OpenAI-Compatible Endpoint

Ollama exposes an OpenAI-compatible API at `/v1/chat/completions`. Works with any OpenAI SDK.

```bash
# Local
curl http://localhost:11434/v1/chat/completions -d '{
  "model": "llama3.2:1b",
  "messages": [{"role": "user", "content": "Hello!"}]
}'

# Cloud
curl -H "Authorization: Bearer $OLLAMA_API_KEY" \
     https://ollama.com/v1/chat/completions -d '{
  "model": "ministral-3:3b",
  "messages": [{"role": "user", "content": "Hello!"}]
}'
```

**Supported OpenAI fields:**

| Field | Supported | Notes |
|-------|-----------|-------|
| `model` | Yes | Required |
| `messages` | Yes | system/user/assistant/tool roles |
| `temperature` | Yes | |
| `top_p` | Yes | |
| `max_tokens` | Yes | (but we never limit — see token policy) |
| `stream` | Yes | SSE format |
| `stop` | Yes | Stop sequences |
| `frequency_penalty` | Yes | |
| `presence_penalty` | Yes | |
| `seed` | Yes | Best-effort reproducibility |
| `tools` | Yes | Function definitions |
| `response_format` | Yes | `{"type": "json_object"}` or JSON schema |
| `tool_choice` | Not supported | |
| `logit_bias` | Not supported | |
| `n` | Not supported | |

**Response format** matches OpenAI's `chat.completion` shape:
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1772738213,
  "model": "llama3.2:1b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello!"},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 5,
    "total_tokens": 14
  }
}
```

---

## Cloud API (`ollama.com`)

### Authentication

```bash
export OLLAMA_API_KEY="your-key-here"

# All cloud requests need the auth header
curl -H "Authorization: Bearer $OLLAMA_API_KEY" \
     https://ollama.com/api/tags
```

### Available Models

The cloud API serves **32 curated models** (as of 2026-03). Use `GET /api/tags` to list them.

### Python Client (Ollama SDK)

```python
from ollama import Client

client = Client(
    host="https://ollama.com",
    headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
)

# Chat
response = client.chat(
    model="ministral-3:3b",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.message.content)

# Show model details
info = client.show("ministral-3:3b")
print(info.capabilities)  # ['completion', 'tools', 'vision']
```

### Python Client (httpx, OpenAI-compatible)

```python
import httpx

response = httpx.post(
    "https://ollama.com/v1/chat/completions",
    headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
    json={
        "model": "ministral-3:3b",
        "messages": [{"role": "user", "content": "Hello!"}]
    }
)
print(response.json()["choices"][0]["message"]["content"])
```

### Cloud Limitations vs Local

- No `load_duration`, `prompt_eval_duration`, `eval_duration` in responses (always null)
- No `logprobs`
- Cloud may ignore `num_ctx` (seen responses exceeding specified context)
- Only 32 models available (vs unlimited via local `ollama pull`)

---

## OCI Registry — Capability Discovery

The Ollama model registry at `registry.ollama.ai` uses OCI (Open Container Initiative) format. You can inspect model manifests and layers **without downloading the full model** to discover capabilities.

### Why Use the Registry?

- **Cloud `/api/show`** only works for the 32 cloud-served models
- **Local `ollama show`** requires the model to be downloaded first
- **Registry manifests** work for ALL models in the Ollama library — no download needed

### Step 1: Fetch the Manifest

```bash
# Check if a model exists (HEAD request)
curl -sI https://registry.ollama.ai/v2/library/llama3.2/manifests/1b

# Fetch full manifest
curl -s https://registry.ollama.ai/v2/library/llama3.2/manifests/1b | python3 -m json.tool
```

**Manifest structure:**
```json
{
  "schemaVersion": 2,
  "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
  "config": {
    "mediaType": "application/vnd.docker.container.image.v1+json",
    "digest": "sha256:config-digest...",
    "size": 485
  },
  "layers": [
    {
      "mediaType": "application/vnd.ollama.image.model",
      "digest": "sha256:model-weights-digest...",
      "size": 1300000000
    },
    {
      "mediaType": "application/vnd.ollama.image.template",
      "digest": "sha256:template-digest...",
      "size": 1234
    },
    {
      "mediaType": "application/vnd.ollama.image.params",
      "digest": "sha256:params-digest...",
      "size": 56
    },
    {
      "mediaType": "application/vnd.ollama.image.license",
      "digest": "sha256:license-digest...",
      "size": 8000
    }
  ]
}
```

### Step 2: Fetch the Template Layer

The **template layer** (`application/vnd.ollama.image.template`) contains the chat template — this reveals tool calling, vision, and thinking support.

```bash
# Extract template digest from manifest, then fetch the blob
TEMPLATE_DIGEST="sha256:template-digest-from-step-1"
curl -s https://registry.ollama.ai/v2/library/llama3.2/blobs/$TEMPLATE_DIGEST
```

### Step 3: Read Capabilities from Template

The template text contains markers that indicate capability support:

| Capability | Template Indicator |
|------------|-------------------|
| **Tool calling** | `"You are a helpful assistant with tool calling capabilities"`, `{{- if .Tools }}`, `[AVAILABLE_TOOLS]`, `<tools>`, `<\|plugin\|>` |
| **Vision** | `{{- if .Images }}`, `[img-` |
| **Thinking** | `{{- if .ThinkingEnabled }}`, `<think>`, `reasoning_content` |

**Example: Llama 3.2 1B template snippet (tools enabled)**
```
{{- if .Tools }}
You are a helpful assistant with tool calling capabilities. When you receive a tool call response, use the output to format an answer to the original user question.
{{- end }}
```

**Example: Gemma 3 1B template snippet (no tools)**
```
{{- range $i, $_ := .Messages }}
<start_of_turn>{{ .Role }}
{{ .Content }}
<end_of_turn>
```
No `{{ .Tools }}` block = no native tool support.

### Step 4: Fetch Config Layer (Architecture Info)

The **config layer** (`application/vnd.docker.container.image.v1+json`) contains architecture details.

```bash
CONFIG_DIGEST="sha256:config-digest-from-manifest"
curl -s https://registry.ollama.ai/v2/library/llama3.2/blobs/$CONFIG_DIGEST
```

### Automated Capability Probe Script

```python
#!/usr/bin/env python3
"""Probe Ollama registry for model capabilities without downloading."""

import httpx
import json

REGISTRY = "https://registry.ollama.ai"

def probe_capabilities(model: str, tag: str = "latest") -> dict:
    """Fetch manifest and template to determine capabilities."""
    # Step 1: Get manifest
    manifest_url = f"{REGISTRY}/v2/library/{model}/manifests/{tag}"
    resp = httpx.get(manifest_url)
    if resp.status_code != 200:
        return {"error": f"Model not found: {model}:{tag}"}

    manifest = resp.json()

    # Step 2: Find template layer
    template_digest = None
    for layer in manifest.get("layers", []):
        if layer["mediaType"] == "application/vnd.ollama.image.template":
            template_digest = layer["digest"]
            break

    if not template_digest:
        return {"model": model, "tag": tag, "capabilities": ["completion"]}

    # Step 3: Fetch template
    blob_url = f"{REGISTRY}/v2/library/{model}/blobs/{template_digest}"
    template_resp = httpx.get(blob_url)
    template = template_resp.text

    # Step 4: Detect capabilities
    caps = ["completion"]

    # Tool calling indicators
    tool_markers = [".Tools", "tool calling capabilities", "[AVAILABLE_TOOLS]",
                    "<tools>", "tool_calls", "<|plugin|>"]
    if any(marker in template for marker in tool_markers):
        caps.append("tools")

    # Vision indicators
    vision_markers = [".Images", "[img-", "image_url"]
    if any(marker in template for marker in vision_markers):
        caps.append("vision")

    # Thinking indicators
    thinking_markers = [".ThinkingEnabled", "<think>"]
    if any(marker in template for marker in thinking_markers):
        caps.append("thinking")

    return {"model": model, "tag": tag, "capabilities": caps, "template_length": len(template)}


if __name__ == "__main__":
    models = [
        ("gemma3", "1b"),
        ("llama3.2", "1b"),
        ("ministral-3", "3b"),
        ("deepseek-r1", "1.5b"),
        ("qwen3", "0.6b"),
    ]

    for model, tag in models:
        result = probe_capabilities(model, tag)
        print(f"{model}:{tag} -> {result.get('capabilities', result)}")
```

---

## Capability Reference

### Capability Values (from `/api/show`)

| Value | Meaning |
|-------|---------|
| `completion` | Basic chat/text generation (all models) |
| `tools` | Native function calling via `tools` parameter |
| `vision` | Image input support (base64 or URL) |
| `thinking` | Extended reasoning with `think` parameter |

### Our Three Candidate Models

| Model | Params | Capabilities | Spatial IoU |
|-------|--------|-------------|-------------|
| `gemma3:1b` | 1.0B | `completion` | 0.204 (full grid fill) |
| `llama3.2:1b` | 1.2B | `completion, tools` | 0.000 (4 corners) |
| `ministral-3:3b` | 3.8B | `completion, tools, vision` | 0.348 (thick rectangle) |

### Example Capabilities by Model Family

| Model | capabilities |
|-------|-------------|
| `gemma3:4b` | `completion, vision` |
| `ministral-3:3b` | `completion, tools, vision` |
| `deepseek-v3.2` | `completion, tools, thinking` |
| `qwen3-coder:480b` | `completion, tools` |
| `llama3.2:1b` | `completion, tools` |

---

## Cloud vs Local Differences

| Feature | Local (`localhost:11434`) | Cloud (`ollama.com`) |
|---------|--------------------------|---------------------|
| Auth | None | `Bearer $OLLAMA_API_KEY` |
| Models | Unlimited (`ollama pull`) | 32 curated |
| `/api/tags` | Lists locally downloaded models | Lists all 32 cloud models |
| `/api/show` | Works for any local model | Works for cloud models only |
| Response metrics | Full (duration, eval counts) | Partial (durations null) |
| Tool calling | Depends on model template | Depends on model template |
| Structured output | Full JSON schema in `format` | Full JSON schema in `format` |
| Registry access | Same registry (`registry.ollama.ai`) | Same registry |
| OpenAI endpoint | `/v1/chat/completions` | `/v1/chat/completions` |
| `num_ctx` control | Respected | May be ignored |

### Endpoint Comparison (Ollama Native vs OpenAI-Compatible)

| Feature | `/api/chat` (Native) | `/v1/chat/completions` (OpenAI) |
|---------|---------------------|-------------------------------|
| Tool result role | `"tool"` with `tool_name` field | `"tool"` with `tool_call_id` field |
| Tool call args | Object (parsed) | String (JSON-encoded) |
| Response shape | `message` at top level, `done` flag | `choices` array, `finish_reason` |
| Metrics | `eval_count`, `total_duration`, etc. | `usage` object (token counts only) |
| JSON mode | `format` with full JSON schema | `response_format` with `json_object` type |
| Thinking | `think: true`, response in `message.thinking` | Not exposed |
| Streaming | `"stream": true`, newline-delimited JSON | `"stream": true`, SSE `data:` format |

---

## Cloud API vs Registry Blob Equivalence

The OCI registry blobs are the decomposed equivalent of what `/api/tags` + `/api/show` return as a single JSON response. Use the registry when you need metadata on models outside the 32 cloud-served set (214 models total in the Ollama library).

| What you need | Cloud API | Registry (no download) |
|---|---|---|
| **List all models** | `GET /api/tags` (32 cloud models only) | Scrape `ollama.com/library` HTML (214 models, no list API) |
| **Model details** | `POST /api/show` (one call, full JSON) | Config blob + template blob + params blob (2-3 calls) |
| **Capabilities** | `/api/show` -> `capabilities` array | Parse template blob for markers (`.Tools`, `.Images`, etc.) |
| **Template** | `/api/show` -> `template` field | Template blob (`application/vnd.ollama.image.template`) |
| **Parameters** | `/api/show` -> `parameters` field | Params blob (`application/vnd.ollama.image.params`) |
| **License** | `/api/show` -> `license` field | License blob (`application/vnd.ollama.image.license`) |
| **Architecture info** | `/api/show` -> `model_info` / `details` | Config blob -> `model_family`, `model_type`, `file_type` |

### Key Differences

- **Cloud `/api/show`** gives a clean `capabilities` array in one call — easy but limited to 32 models
- **Registry blobs** require fetching the manifest first (for digests), then each blob individually — the blob endpoint returns **307 redirects** so clients must follow redirects
- **No list API exists** for the full download library — the only way to enumerate all 214 models is to scrape `ollama.com/library` (HTML page)
- **No `/v2/_catalog`** or `/v2/library/{model}/tags/list`** — the registry does not implement these standard OCI endpoints
- **Config blob** only has `model_family`, `model_type` (e.g. "3.8B"), `file_type` (e.g. "Q4_K_M") — less detail than `/api/show`'s `model_info` which includes `context_length`, `embedding_length`, `parameter_count`
- **Vision detection caveat**: some models (e.g. ministral-3) declare vision in template text as `"You have the ability to read images"` rather than Go template markers like `{{ .Images }}`

### Complete Blob Fetch Example

```bash
# 1. Get manifest (contains digests for all layers)
curl -sL https://registry.ollama.ai/v2/library/ministral-3/manifests/3b | python3 -m json.tool

# 2. Fetch config blob (architecture info) — note -L to follow 307 redirect
CONFIG_DIGEST="sha256:..."  # from manifest.config.digest
curl -sL https://registry.ollama.ai/v2/library/ministral-3/blobs/$CONFIG_DIGEST | python3 -m json.tool

# 3. Fetch template blob (chat template, capability markers)
TEMPLATE_DIGEST="sha256:..."  # from layers where mediaType contains "template"
curl -sL https://registry.ollama.ai/v2/library/ministral-3/blobs/$TEMPLATE_DIGEST

# 4. Fetch params blob (runtime parameters like stop tokens)
PARAMS_DIGEST="sha256:..."  # from layers where mediaType contains "params"
curl -sL https://registry.ollama.ai/v2/library/ministral-3/blobs/$PARAMS_DIGEST

# 5. Fetch license blob
LICENSE_DIGEST="sha256:..."  # from layers where mediaType contains "license"
curl -sL https://registry.ollama.ai/v2/library/ministral-3/blobs/$LICENSE_DIGEST
```

---

## Quick Reference Card

```bash
# ---- CLI ----
ollama pull MODEL            # Download model
ollama list                  # List local models
ollama show MODEL            # Show model details
ollama show MODEL --template # Show chat template only
ollama run MODEL             # Interactive chat
ollama ps                    # Running models
ollama stop MODEL            # Unload from VRAM
ollama rm MODEL              # Delete local model
ollama cp SRC DST            # Copy/rename model
ollama create NAME --from MODEL  # Create custom model
ollama serve                 # Start API server

# ---- REST (local) ----
GET  /api/tags               # List models
POST /api/show               # Model details + capabilities
POST /api/chat               # Chat completion (native format)
POST /api/generate           # Text completion
POST /api/embed              # Embeddings
POST /api/pull               # Download model
POST /api/create             # Create custom model
POST /api/copy               # Copy model
DELETE /api/delete            # Delete model
GET  /api/ps                 # Running models
GET  /v1/chat/completions    # OpenAI-compatible chat

# ---- Registry (no auth, no download needed) ----
HEAD /v2/library/MODEL/manifests/TAG   # Check existence
GET  /v2/library/MODEL/manifests/TAG   # Get OCI manifest
GET  /v2/library/MODEL/blobs/DIGEST    # Fetch layer (template, config, etc.)
```
