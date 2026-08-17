# Instructions

Three flows: run the agent, compare models, and view the dashboard. See `README.md` for the
full reference and `CLAUDE.md` for implementation detail — this is the short, task-oriented version.

## Prerequisites

- Docker (with Docker Compose)
- An [OpenRouter](https://openrouter.ai/) API key

```bash
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY
```

Everything below assumes you're in the repo root.

## Flow 1: Run the agent

Downloads a few papers from arXiv and summarizes them with the active model.

```bash
docker compose up --build
```

What happens, in order:
1. `entrypoint.sh` starts the trace viewer (`nooa start-dev`) in the background.
2. `simulator.py` downloads a few recent arXiv papers into the `papers_data` volume (skips any
   already on disk).
3. `agent.py` processes every pending PDF: extracts text, gets a structured summary from the LLM
   (`ACTIVE_MODEL_SLUG`, default `nvidia/nemotron-3-nano-30b-a3b`), and records it — both in the
   exact-match dedup table and, separately, as an embedded record for semantic search.
4. The container keeps running afterward (serving the trace viewer) until you stop it.

To run again in the background and watch logs separately:

```bash
docker compose up --build -d
docker compose logs -f research-agent
```

Check what got processed without touching the viewer:

```bash
docker compose exec research-agent python -c "
import sqlite3, json
with sqlite3.connect('/data/papers/.agent_state.sqlite3') as conn:
    conn.row_factory = sqlite3.Row
    for r in conn.execute('SELECT * FROM processed_papers ORDER BY processed_at'):
        print(json.dumps(dict(r), indent=2))
"
```

Or search processed papers semantically:

```bash
docker compose exec research-agent python -c \
  "from agent import ResearchAgent; print(ResearchAgent().find_similar_papers('your query here'))"
```

To change which model processes papers, or which/how many papers get fetched, see README.md's
"Switching models" and "Choosing which papers to fetch/process" sections (`ACTIVE_MODEL_SLUG`,
`NUM_PAPERS`, `ARXIV_PAPER_IDS`).

## Flow 2: Compare models

Runs the same papers through a stronger "reference" model and one or more cheaper "candidate"
models, then scores each candidate against the reference. Needs the container from Flow 1 already
running (`docker compose up -d` if you haven't).

```bash
docker compose exec research-agent python compare_models.py
```

**Cost note**: the default reference model (`openai/gpt-4o-mini`) is paid — one call per paper.
Defaults to 2 papers to keep this cheap; override with `COMPARE_NUM_PAPERS` in `.env` if you want a
larger sample. Candidate models default to `:free`-suffixed OpenRouter slugs (no cost).

The comparison score is a blunt `difflib` text diff, not a semantic judge — see README.md's
"Reading the scores" for why low similarity scores don't necessarily mean the candidate did badly
(and why that scorer is intentionally basic and staying that way).

## Flow 3: See the dashboard

The trace viewer (every LLM call, prompt, and method invocation, plus the Evaluations and Memory
tabs) runs on port 5001 inside the container — but it rejects any request that doesn't look like
genuine loopback traffic, and Docker's normal port publishing always makes host-side requests look
like they came from a bridge address instead. Plain `localhost:5001` in your browser will 403.

The fix: open this repo folder in **VS Code with the Dev Containers extension**, then
"Reopen in Container." VS Code's port forwarding tunnels from inside the container's own network
namespace, so the connection genuinely looks like loopback and the viewer just works —
`.devcontainer/devcontainer.json` is already set up to forward port 5001 and open it automatically.

Steps:
1. Have Flow 1 running (`docker compose up -d`), or let the Dev Container start it.
2. Open this folder in VS Code.
3. Command Palette → "Dev Containers: Reopen in Container" (or click the prompt VS Code shows).
4. Once attached, check the **Ports** panel (a tab near Terminal/Problems/Output) for port `5001`
   ("NOOA Trace Viewer") and click its globe/"Open in Browser" icon — don't just type the URL by
   hand, the forwarding needs to actually establish first.

What you should see:
- **Evaluations** tab — populated after running Flow 2 (`compare_models.py`).
- **Memory** tab — populated after running Flow 1 (each processed paper gets an embedded record).
- General trace data — every LLM call/method invocation from either flow.

If you're not using VS Code, there's no other way to reach the viewer from a browser (`nooa` also
can't run natively on Windows outside a container, and running it directly won't help since the
issue is the loopback check, not where `nooa` runs) — fall back to the SQLite/`find_similar_papers`
queries shown in Flow 1 instead.
