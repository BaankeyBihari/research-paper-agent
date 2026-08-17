"""Fetch OpenRouter models, bucket them by live pricing tiers, and update .env slugs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import requests

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
ENV_KEYS = {"ACTIVE_MODEL_SLUG", "REFERENCE_MODEL_SLUG", "CANDIDATE_MODEL_SLUGS"}
FAMILIES = {"deepseek", "gemini", "gemma", "anthropic", "openai", "grok", "other"}
TIERS = ("budget", "mid", "premium")


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


def filter_models(models: list[dict], families: set[str], tier: str | None) -> list[dict]:
    rows = [m for m in models if (not families or m["family"] in families)]
    if tier:
        rows = [m for m in rows if m.get("tier") == tier]
    return sorted(rows, key=lambda m: (m["token_pair_per_m"], m["id"]))


def print_table(models: list[dict], limit: int | None = None) -> None:
    rows = models[:limit] if limit is not None else models
    print(
        " # | tier    | family    | prompt/M | completion/M | prompt+completion/M | model slug"
    )
    print("-" * 108)
    for idx, m in enumerate(rows, start=1):
        print(
            f"{idx:>2} | "
            f"{m.get('tier', 'n/a'):<7} | "
            f"{m['family']:<9} | "
            f"${m['prompt_per_m']:>7.3f} | "
            f"${m['completion_per_m']:>11.3f} | "
            f"${m['token_pair_per_m']:>19.3f} | "
            f"{m['id']}"
        )
    print(f"\nShowing {len(rows)} of {len(models)} matching models.")


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


def choose_env_key_interactive() -> str:
    while True:
        answer = input(
            "Choose env key to update "
            "(ACTIVE_MODEL_SLUG / REFERENCE_MODEL_SLUG / CANDIDATE_MODEL_SLUGS): "
        ).strip()
        if answer in ENV_KEYS:
            return answer
        print("Invalid key. Please choose one of the listed values.")


def choose_index_interactive(max_idx: int) -> int:
    while True:
        answer = input(f"Choose model number (1-{max_idx}): ").strip()
        try:
            idx = int(answer)
        except ValueError:
            idx = -1
        if 1 <= idx <= max_idx:
            return idx
        print("Invalid number.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        action="append",
        choices=sorted(FAMILIES),
        help="Filter by family. Repeat to include multiple families.",
    )
    parser.add_argument(
        "--tier",
        choices=TIERS,
        help="Filter by dynamic pricing tier (budget/mid/premium).",
    )
    parser.add_argument("--limit", type=int, default=50, help="Max table rows to print (0 = all).")
    parser.add_argument("--interactive", action="store_true", help="Prompt and update .env in place.")
    parser.add_argument("--env-file", default=".env", help="Path to the .env file to update.")
    parser.add_argument(
        "--set-key",
        choices=sorted(ENV_KEYS),
        help="Non-interactive env key to update (requires --index).",
    )
    parser.add_argument(
        "--index",
        type=int,
        help="1-based row index from the printed/filter list to use for env updates.",
    )
    parser.add_argument(
        "--append-candidate",
        action="store_true",
        help="When setting CANDIDATE_MODEL_SLUGS, append instead of replace.",
    )
    parser.add_argument("--json", action="store_true", help="Print filtered results as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise SystemExit("--limit must be >= 0.")
    if args.interactive and (args.set_key or args.index is not None):
        raise SystemExit("--interactive cannot be combined with --set-key/--index.")
    if args.interactive and args.json:
        raise SystemExit("--interactive cannot be combined with --json.")
    if args.json and (args.set_key or args.index is not None):
        raise SystemExit("--json cannot be combined with --set-key/--index.")
    if args.index is not None and not args.set_key and not args.interactive:
        raise SystemExit("--index requires --set-key (or use --interactive).")

    models = fetch_models()
    assign_pricing_tiers(models)
    selected = filter_models(models, set(args.family or []), args.tier)
    if not selected:
        raise SystemExit("No models matched the requested filters.")

    if args.json:
        print(json.dumps(selected, indent=2))
        return

    # --index (interactive or not) must resolve against exactly what was printed --
    # otherwise --limit truncating the table could let an index the user never saw
    # (e.g. 51+ under the default limit) silently select and write an unrelated model.
    display_limit = args.limit if args.limit > 0 else None
    displayed = selected[:display_limit] if display_limit is not None else selected
    print_table(displayed)

    key = args.set_key
    index = args.index
    if args.interactive:
        key = choose_env_key_interactive()
        index = choose_index_interactive(len(displayed))

    if key:
        if index is None:
            raise SystemExit("--set-key requires --index (or use --interactive).")
        if not (1 <= index <= len(displayed)):
            raise SystemExit(f"--index must be between 1 and {len(displayed)} for the displayed rows.")
        slug = displayed[index - 1]["id"]
        env_path = Path(args.env_file)
        update_env_model(env_path, key, slug, append_candidate=args.append_candidate)
        print(f"Updated {env_path} -> {key}={slug}")


if __name__ == "__main__":
    main()
