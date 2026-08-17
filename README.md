# research-paper-agent

A small Dockerized research assistant built on [NVIDIA NOOA](https://github.com/NVIDIA-NeMo/labs-OO-Agents)
(Object-Oriented Agents), used to try the framework's "agent as a Python class" model and to compare
OpenRouter models on summarization quality/speed.

`ResearchAgent` scans `/data/papers` for PDFs, extracts each paper's Main Objective, Key Methodology,
and Resulting Metrics via an LLM call, and records the result in a local SQLite file so re-runs (even
under a different model) don't reprocess the same paper.

## Setup

```bash
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY
docker compose up --build
```

## Viewing results

**The processed summaries** (title, objective, methodology, metrics) live in a SQLite table inside
the container and are the most reliable way to check what happened:

```bash
docker compose exec research-agent python -c "
import sqlite3, json
with sqlite3.connect('/data/papers/.agent_state.sqlite3') as conn:
    conn.row_factory = sqlite3.Row
    for r in conn.execute('SELECT * FROM processed_papers ORDER BY processed_at'):
        print(json.dumps(dict(r), indent=2))
"
```

**The trace viewer** (every LLM call, prompt, and method invocation NOOA makes) is trickier to reach
from a browser on Windows/Mac. It runs inside the container on port 5001, but its API refuses any
request that doesn't look like loopback traffic once bound to `0.0.0.0` — and Docker's port
publishing (`-p 5001:5001`) always makes host-side requests look like they came from a bridge/NAT
address, not `127.0.0.1`, so a plain `docker compose up` + `localhost:5001` in your browser gets a
403. Running `nooa start-dev` directly on Windows as a workaround doesn't work either — its storage
layer imports the Unix-only `fcntl` module.

The fix that actually works: open this folder in **VS Code with the Dev Containers extension**
("Reopen in Container"). VS Code's port forwarding tunnels from *inside* the container's own network
namespace rather than through Docker's external NAT, so the connection genuinely looks like loopback
traffic to the viewer and it just works — no token, no extra config. `.devcontainer/devcontainer.json`
is already set up to forward port 5001 and open it in your browser automatically. If you're not using
VS Code, the SQLite query above is the dependable fallback.

## Switching models

Set `ACTIVE_MODEL_SLUG` in `.env` (or inline) to any OpenRouter slug, then rebuild/rerun:

| Tier | Slug |
|---|---|
| Fast/Cheap | `nvidia/nemotron-3.5-lightning` |
| Efficient | `nvidia/nemotron-3-nano-30b-a3b` |
| High-Reasoning | `meta-llama/llama-3.1-70b-instruct` |

```bash
ACTIVE_MODEL_SLUG=nvidia/nemotron-3.5-lightning docker compose up --build
```

## Testing efficiency at home

1. **Nano run**: `ACTIVE_MODEL_SLUG=nvidia/nemotron-3-nano-30b-a3b docker compose up --build`.
   Watch time-to-first-token and summary quality in the trace viewer.
2. **Lightning run**: switch the env var and rerun. Compare summary quality, especially on
   math/code-heavy papers.
3. **Persistence check**: the `papers_data` Docker volume holds both the downloaded PDFs and
   `.agent_state.sqlite3`. Because that volume persists across runs regardless of which model is
   active, a paper processed under Nano is still recorded (title + extracted fields) when you
   switch to Lightning — inspect it directly:
   ```bash
   docker compose run --rm research-agent python -c \
     "from agent import ResearchAgent; import asyncio; a = ResearchAgent(); print(a.lookup_paper('<filename>.pdf'))"
   ```

## Known deviations from the original spec, and why

- **Structured output is a Pydantic model, not a raw dict.** NOOA's documented convention for
  agentic methods with a structured return type is a `pydantic.BaseModel` subclass — that's what
  `summarize_paper` returns (`PaperSummary`). Functionally equivalent to a dict (same four fields:
  title, main_objective, key_methodology, resulting_metrics) but validated by the framework.
- **"NOOA's SQLite memory" is not the `nooa-memory` package.** `nooa-memory` (included in
  `requirements.txt` for you to explore) is a vector-search long-term recall subsystem meant for
  semantic recall, not exact-match dedup, and its API wasn't documented in enough depth to wire up
  with confidence. Instead, `ResearchAgent` tracks processed papers via a plain `sqlite3` table
  (`processed_papers`) in the same `/data/papers` volume — deterministic, and it's what actually
  makes the persistence test in step 3 above work.

## Verified

Confirmed end-to-end with a live run against `nvidia/nemotron-3-nano-30b-a3b` on OpenRouter:
- Image builds cleanly; `simulator.py` downloads real papers from the live arXiv API; `pypdf`
  extracts their text; `summarize_paper` produces valid `PaperSummary` objects from real model
  output; the SQLite `processed_papers` table persists correctly across `docker compose down`/`up`
  cycles (i.e. across the same volume you'd reuse when switching `ACTIVE_MODEL_SLUG`), confirming
  the cross-model persistence test in step 3 actually holds.
- `nooa start-dev` binds `0.0.0.0:5001` by default and `GET /` returns 200 (serves the SPA shell),
  **but** its data API 403s from outside the container over a published Docker port — the page loads
  and looks blank. This is what the "Viewing results" section above walks through; the
  `.devcontainer` setup is in place but hasn't been confirmed working from an actual VS Code session
  yet (it needs a real Dev Containers session to verify — see below).

One tuning note from the live run: on `nemotron-3-nano-30b-a3b`, the first version of the
`summarize_paper` docstring sometimes produced a `title` containing the full author list, and a
`main_objective` that was a verbatim sentence lift rather than a synthesis. The docstring now gives
explicit per-field instructions (title = paper title only, no authors; objective = synthesized, not
copied; metrics = "not reported" when absent) — this measurably cleaned up the output in a follow-up
run. Worth watching for the same failure mode if you try other budget/free-tier models.
