"""Serve a local OpenRouter model browser UI and update .env slugs."""

from __future__ import annotations

import argparse
import json
import logging
import math
import threading
import webbrowser
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
ENV_KEYS = {"ACTIVE_MODEL_SLUG", "REFERENCE_MODEL_SLUG", "CANDIDATE_MODEL_SLUGS"}
FAMILIES = {"deepseek", "gemini", "gemma", "anthropic", "openai", "grok", "other"}
TIERS = ("budget", "mid", "premium")
MAX_UPDATE_PAYLOAD_BYTES = 16 * 1024


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def family_from_model_id(model_id: str) -> str:
    provider, _, name = model_id.partition("/")
    if provider == "deepseek":
        return "deepseek"
    if provider == "anthropic":
        return "anthropic"
    if provider == "openai":
        return "openai"
    if provider == "x-ai" and name.startswith("grok"):
        return "grok"
    if provider == "google" and name.startswith("gemini"):
        return "gemini"
    if provider == "google" and name.startswith("gemma"):
        return "gemma"
    return "other"


def fetch_models() -> list[dict]:
    try:
        resp = requests.get(OPENROUTER_MODELS_URL, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        raise SystemExit(f"Failed to fetch {OPENROUTER_MODELS_URL}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"{OPENROUTER_MODELS_URL} did not return valid JSON (got a non-JSON response, "
            f"e.g. an HTML error/proxy page): {exc}"
        ) from exc

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise SystemExit(
            f"Unexpected response shape from {OPENROUTER_MODELS_URL}: expected an object with "
            "a 'data' list"
        )

    models: list[dict] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        model_id = row.get("id")
        if not model_id:
            continue
        pricing = row.get("pricing") or {}
        prompt = _as_float(pricing.get("prompt")) * 1_000_000
        completion = _as_float(pricing.get("completion")) * 1_000_000
        request = _as_float(pricing.get("request"))
        image = _as_float(pricing.get("image"))
        models.append(
            {
                "id": model_id,
                "family": family_from_model_id(model_id),
                "prompt_per_m": prompt,
                "completion_per_m": completion,
                "request": request,
                "image": image,
                "token_pair_per_m": prompt + completion,
            }
        )
    return models


def assign_pricing_tiers(models: list[dict]) -> None:
    for model in models:
        if model["token_pair_per_m"] == 0:
            model["tier"] = "budget"

    priced = sorted((m for m in models if m["token_pair_per_m"] > 0), key=lambda m: m["token_pair_per_m"])
    if not priced:
        return

    n = len(priced)
    budget_end = math.ceil(n / 3)
    mid_end = math.ceil((2 * n) / 3)
    for idx, model in enumerate(priced):
        if idx < budget_end:
            model["tier"] = "budget"
        elif idx < mid_end:
            model["tier"] = "mid"
        else:
            model["tier"] = "premium"


def read_env_value(env_path: Path, key: str) -> str | None:
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip().strip("'\"")
    return None


def write_env_value(env_path: Path, key: str, value: str) -> None:
    newline = "\n"
    if env_path.exists():
        original_bytes = env_path.read_bytes()
        if b"\r\n" in original_bytes:
            newline = "\r\n"
        original_text = original_bytes.decode("utf-8")
        lines = original_text.splitlines()
    else:
        lines = []

    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            updated = True
            break
    if not updated:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f"{key}={value}")
    # write_bytes (not write_text) deliberately -- write_text opens in text mode, which on
    # Windows re-translates the "\r\n" this function already inserted for CRLF files into
    # "\r\r\n", corrupting the very line endings this is trying to preserve.
    content = newline.join(lines) + newline
    env_path.write_bytes(content.encode("utf-8"))


def update_env_model(env_path: Path, key: str, slug: str, append_candidate: bool) -> None:
    if key == "CANDIDATE_MODEL_SLUGS" and append_candidate:
        existing = read_env_value(env_path, key) or ""
        slugs = [part.strip() for part in existing.split(",") if part.strip()]
        if slug not in slugs:
            slugs.append(slug)
        write_env_value(env_path, key, ",".join(slugs))
        return
    write_env_value(env_path, key, slug)


def sorted_models(models: list[dict]) -> list[dict]:
    return sorted(models, key=lambda m: (m["token_pair_per_m"], m["id"]))


def build_page(models: list[dict], env_path: Path) -> str:
    models_json = json.dumps(models).replace("</", "<\\/")
    env_display = escape(str(env_path))
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Model Selector</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 20px; }}
    .controls {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }}
    input, select, button {{ font: inherit; padding: 6px 8px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    th button {{ background: none; border: 0; padding: 0; cursor: pointer; font-weight: 600; }}
    tr:nth-child(even) {{ background: #fafafa; }}
    .selection {{ margin: 12px 0; padding: 10px; border: 1px solid #ddd; }}
    .status {{ margin-top: 8px; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>OpenRouter model selector</h1>
  <p>Loaded <strong id=\"count\"></strong> models. Updates write to <code>{env_display}</code>.</p>

  <div class=\"controls\">
    <label>Search slug <input id=\"search\" type=\"search\" placeholder=\"provider/model\"></label>
    <label>Family
      <select id=\"family\">
        <option value=\"\">All</option>
        <option value=\"anthropic\">anthropic</option>
        <option value=\"deepseek\">deepseek</option>
        <option value=\"gemini\">gemini</option>
        <option value=\"gemma\">gemma</option>
        <option value=\"grok\">grok</option>
        <option value=\"openai\">openai</option>
        <option value=\"other\">other</option>
      </select>
    </label>
    <label>Tier
      <select id=\"tier\">
        <option value=\"\">All</option>
        <option value=\"budget\">budget</option>
        <option value=\"mid\">mid</option>
        <option value=\"premium\">premium</option>
      </select>
    </label>
  </div>

  <table>
    <thead>
      <tr>
        <th><button data-sort=\"id\">Model slug</button></th>
        <th><button data-sort=\"family\">Family</button></th>
        <th><button data-sort=\"tier\">Tier</button></th>
        <th><button data-sort=\"prompt_per_m\">Prompt/M</button></th>
        <th><button data-sort=\"completion_per_m\">Completion/M</button></th>
        <th><button data-sort=\"token_pair_per_m\">Prompt+Completion/M</button></th>
        <th>Use</th>
      </tr>
    </thead>
    <tbody id=\"rows\"></tbody>
  </table>

  <div class=\"selection\">
    <div>Selected model: <code id=\"selected\">(none)</code></div>
    <div class=\"controls\">
      <label>Env key
        <select id=\"key\">
          <option value=\"ACTIVE_MODEL_SLUG\">ACTIVE_MODEL_SLUG</option>
          <option value=\"REFERENCE_MODEL_SLUG\">REFERENCE_MODEL_SLUG</option>
          <option value=\"CANDIDATE_MODEL_SLUGS\">CANDIDATE_MODEL_SLUGS</option>
        </select>
      </label>
      <label><input id=\"appendCandidate\" type=\"checkbox\"> Append candidate value</label>
      <button id=\"save\" type=\"button\">Write to .env</button>
    </div>
    <div id=\"status\" class=\"status\"></div>
  </div>

  <script>
    const MODELS = {models_json};
    let selectedSlug = "";
    let sortKey = "token_pair_per_m";
    let sortAsc = true;

    const searchEl = document.getElementById("search");
    const familyEl = document.getElementById("family");
    const tierEl = document.getElementById("tier");
    const rowsEl = document.getElementById("rows");
    const countEl = document.getElementById("count");
    const selectedEl = document.getElementById("selected");
    const statusEl = document.getElementById("status");

    const fmtUsd = (v) => "$" + Number(v).toFixed(3);

    function setStatus(message, ok=true) {{
      statusEl.textContent = message;
      statusEl.style.color = ok ? "#0a7" : "#b00";
    }}

    function filteredModels() {{
      const query = searchEl.value.trim().toLowerCase();
      const family = familyEl.value;
      const tier = tierEl.value;
      return MODELS.filter((m) => {{
        if (family && m.family !== family) return false;
        if (tier && m.tier !== tier) return false;
        if (query && !m.id.toLowerCase().includes(query)) return false;
        return true;
      }});
    }}

    function sorted(list) {{
      return [...list].sort((a, b) => {{
        const av = a[sortKey];
        const bv = b[sortKey];
        const cmp = typeof av === "number" && typeof bv === "number"
          ? av - bv
          : String(av).localeCompare(String(bv));
        return sortAsc ? cmp : -cmp;
      }});
    }}

    function render() {{
      const list = sorted(filteredModels());
      countEl.textContent = `${{list.length}} / ${{MODELS.length}}`;
      rowsEl.innerHTML = "";
      for (const m of list) {{
        const tr = document.createElement("tr");
        const slugTd = document.createElement("td");
        const slugCode = document.createElement("code");
        slugCode.textContent = m.id;
        slugTd.appendChild(slugCode);
        tr.appendChild(slugTd);

        const familyTd = document.createElement("td");
        familyTd.textContent = m.family;
        tr.appendChild(familyTd);

        const tierTd = document.createElement("td");
        tierTd.textContent = m.tier;
        tr.appendChild(tierTd);

        const promptTd = document.createElement("td");
        promptTd.textContent = fmtUsd(m.prompt_per_m);
        tr.appendChild(promptTd);

        const completionTd = document.createElement("td");
        completionTd.textContent = fmtUsd(m.completion_per_m);
        tr.appendChild(completionTd);

        const totalTd = document.createElement("td");
        totalTd.textContent = fmtUsd(m.token_pair_per_m);
        tr.appendChild(totalTd);

        const actionTd = document.createElement("td");
        const useBtn = document.createElement("button");
        useBtn.type = "button";
        useBtn.dataset.slug = m.id;
        useBtn.textContent = "Use this";
        actionTd.appendChild(useBtn);
        tr.appendChild(actionTd);
        rowsEl.appendChild(tr);
      }}
      rowsEl.querySelectorAll("button[data-slug]").forEach((btn) => {{
        btn.addEventListener("click", () => {{
          selectedSlug = btn.dataset.slug || "";
          selectedEl.textContent = selectedSlug || "(none)";
          setStatus(`Selected ${{selectedSlug}}`);
        }});
      }});
    }}

    searchEl.addEventListener("input", render);
    familyEl.addEventListener("change", render);
    tierEl.addEventListener("change", render);

    document.querySelectorAll("button[data-sort]").forEach((btn) => {{
      btn.addEventListener("click", () => {{
        const key = btn.dataset.sort;
        if (sortKey === key) sortAsc = !sortAsc;
        else {{
          sortKey = key;
          sortAsc = true;
        }}
        render();
      }});
    }});

    document.getElementById("save").addEventListener("click", async () => {{
      if (!selectedSlug) {{
        setStatus("Select a model first.", false);
        return;
      }}
      const payload = {{
        key: document.getElementById("key").value,
        slug: selectedSlug,
        append_candidate: document.getElementById("appendCandidate").checked,
      }};
      try {{
        const resp = await fetch("/update", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(payload),
        }});
        const data = await resp.json();
        if (!resp.ok) {{
          setStatus(data.message || "Update failed.", false);
          return;
        }}
        setStatus(data.message + " (server will stop)");
      }} catch (err) {{
        setStatus(`Request failed: ${{err}}`, false);
      }}
    }});

    render();
  </script>
</body>
</html>
"""


class ModelSelectorHandler(BaseHTTPRequestHandler):
    def _json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        models: list[dict] = getattr(self.server, "models", [])
        env_path: Path = getattr(self.server, "env_path", Path(".env"))
        page = build_page(models, env_path).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/update":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json(HTTPStatus.BAD_REQUEST, {"message": "Invalid Content-Length."})
            return
        if length < 0 or length > MAX_UPDATE_PAYLOAD_BYTES:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"message": "Payload too large."})
            return
        if length == 0:
            self._json(HTTPStatus.BAD_REQUEST, {"message": "Request body is required."})
            return
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._json(HTTPStatus.BAD_REQUEST, {"message": "Invalid JSON payload."})
            return

        key = payload.get("key")
        slug = payload.get("slug")
        append_candidate = bool(payload.get("append_candidate"))

        if not isinstance(key, str) or key not in ENV_KEYS:
            self._json(HTTPStatus.BAD_REQUEST, {"message": "Invalid .env key."})
            return

        models: list[dict] = getattr(self.server, "models", [])
        env_path: Path = getattr(self.server, "env_path", Path(".env"))
        known_slugs = {m["id"] for m in models}
        if not isinstance(slug, str):
            self._json(HTTPStatus.BAD_REQUEST, {"message": "Invalid model slug."})
            return
        if slug not in known_slugs:
            self._json(HTTPStatus.BAD_REQUEST, {"message": "Unknown model slug."})
            return

        try:
            update_env_model(env_path, key, slug, append_candidate=append_candidate)
        except Exception as exc:
            self._json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"message": f"Failed to update {env_path}: {exc}"},
            )
            return
        message = f"Updated {env_path} -> {key}={slug}"
        print(message, flush=True)
        self._json(HTTPStatus.OK, {"message": message})
        try:
            self.wfile.flush()
        except OSError:
            pass

        threading.Thread(target=self.server.shutdown).start()

    def log_message(self, format: str, *args: object) -> None:
        logging.getLogger(__name__).debug("selector server: " + format, *args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env", help="Path to the .env file to update.")
    parser.add_argument("--json", action="store_true", help="Print fetched model catalog as JSON.")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open a browser tab; print the local URL only.",
    )
    return parser.parse_args()


def serve(models: list[dict], env_path: Path, open_browser: bool) -> None:
    with ThreadingHTTPServer(("127.0.0.1", 0), ModelSelectorHandler) as server:
        server.models = models  # type: ignore[attr-defined]
        server.env_path = env_path  # type: ignore[attr-defined]
        host, port = server.server_address
        url = f"http://{host}:{port}/"
        print(f"Model selector UI: {url}")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        print("Pick a model in the page to update .env; server exits after one successful write.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def main() -> None:
    args = parse_args()

    models = fetch_models()
    assign_pricing_tiers(models)
    models = sorted_models(models)

    if args.json:
        print(json.dumps(models, indent=2))
        return

    serve(models, Path(args.env_file), open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
