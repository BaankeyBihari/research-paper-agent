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

Trace viewer: http://localhost:5001 (every LLM call and method invocation NOOA makes is traced here).

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

The image builds cleanly and everything that doesn't require a live OpenRouter call has been
exercised directly in the built container:
- `nooa start-dev --help` confirms it already binds `0.0.0.0:5001` by default, so `entrypoint.sh`'s
  explicit `--host`/`--port` flags are redundant but harmless.
- `agent.py` imports cleanly and `ResearchAgent`'s deterministic SQLite dedup/lookup logic
  (`get_local_papers`, `lookup_paper`) works correctly against an empty state.
- `simulator.py` successfully downloads real papers from the live arXiv API, and `pypdf` extracts
  text from them correctly.

**Not yet verified**: an actual OpenRouter model call (`summarize_paper`) and a live trace-viewer
session, since both need your `OPENROUTER_API_KEY`. Run `docker compose up --build` and check
`localhost:5001` to confirm end-to-end.
