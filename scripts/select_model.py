"""Fetch OpenRouter models, bucket them by live pricing tiers, and update .env slugs."""

from __future__ import annotations

import argparse
import json
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
    except requests.RequestException as exc:
        raise SystemExit(f"Failed to fetch {OPENROUTER_MODELS_URL}: {exc}") from exc
    payload = resp.json()
    data = payload.get("data", [])
    models: list[dict] = []
    for row in data:
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
    ranked = sorted((m["token_pair_per_m"] for m in models))
    if not ranked:
        return
    low_cut = ranked[len(ranked) // 3]
    high_cut = ranked[(2 * len(ranked)) // 3]
    for model in models:
        price = model["token_pair_per_m"]
        if price <= low_cut:
            model["tier"] = "budget"
        elif price <= high_cut:
            model["tier"] = "mid"
        else:
            model["tier"] = "premium"


def filter_models(models: list[dict], families: set[str], tier: str | None) -> list[dict]:
    rows = [m for m in models if (not families or m["family"] in families)]
    if tier:
        rows = [m for m in rows if m.get("tier") == tier]
    return sorted(rows, key=lambda m: (m["token_pair_per_m"], m["id"]))


def print_table(models: list[dict], limit: int | None = None) -> None:
    rows = models[:limit] if limit else models
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
    lines = env_path.read_text().splitlines() if env_path.exists() else []
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
    env_path.write_text("\n".join(lines) + "\n")


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
    parser.add_argument("--limit", type=int, default=50, help="Max table rows to print.")
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
    models = fetch_models()
    assign_pricing_tiers(models)
    selected = filter_models(models, set(args.family or []), args.tier)
    if not selected:
        raise SystemExit("No models matched the requested filters.")

    if args.json:
        print(json.dumps(selected, indent=2))
    else:
        print_table(selected, limit=args.limit if args.limit > 0 else None)

    key = args.set_key
    index = args.index
    if args.interactive:
        key = choose_env_key_interactive()
        index = choose_index_interactive(len(selected))

    if key:
        if index is None:
            raise SystemExit("--set-key requires --index (or use --interactive).")
        if not (1 <= index <= len(selected)):
            raise SystemExit(f"--index must be between 1 and {len(selected)} for current filters.")
        slug = selected[index - 1]["id"]
        env_path = Path(args.env_file)
        update_env_model(env_path, key, slug, append_candidate=args.append_candidate)
        print(f"Updated {env_path} -> {key}={slug}")


if __name__ == "__main__":
    main()
