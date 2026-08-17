"""ResearchAgent: reads PDFs from /data/papers and extracts structured findings using NOOA."""

import asyncio
import json
import os
import re
import sqlite3
from pathlib import Path

from pydantic import BaseModel
from pypdf import PdfReader

from nooa import Agent
from nooa.unifiedllm.registry import get_llm_client
from nooa_memory import Memory, MemoryStore, MemoryType
from nooa_memory.config import EmbeddingConfig
from nooa_memory.embeddings import get_embedder

MODEL_SLUG = os.environ.get("ACTIVE_MODEL_SLUG", "nvidia/nemotron-3-nano-30b-a3b")
PAPERS_DIR = Path(os.environ.get("PAPERS_DIR", "/data/papers"))
STATE_DB = PAPERS_DIR / ".agent_state.sqlite3"
# Deliberately NOT under PAPERS_DIR: the trace viewer's Memory tab only auto-discovers
# stores at <its cwd>/.nooa/memory/*.sqlite (nooa_memory's own MemoryManager convention),
# and refuses to open a store outside its cwd even via ?db=. entrypoint.sh runs
# `nooa start-dev` from the Dockerfile's WORKDIR (/app) without cd'ing elsewhere, and
# agent.py itself always runs from the same cwd, so this relative path lands in the same
# place the viewer looks -- see docker-compose.yml's ./nooa-memory volume for persistence.
MEMORY_DB = Path(".nooa/memory/memory.sqlite")
MAX_CHARS = 20_000

def _parse_int_env(name: str) -> int | None:
    """Parse an optional integer env var, failing with a clear message rather than
    a raw ValueError traceback if it's set to something non-numeric."""
    raw = os.environ.get(name) or None
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(f"{name}={raw!r} is not a valid integer") from None
    if value < 0:
        raise SystemExit(f"{name}={raw!r} must not be negative")
    return value


NUM_PAPERS = _parse_int_env("NUM_PAPERS")
ARXIV_PAPER_IDS = [i.strip() for i in os.environ.get("ARXIV_PAPER_IDS", "").split(",") if i.strip()]

llm = get_llm_client(f"openrouter/{MODEL_SLUG}")
# Hashing backend: deterministic, offline, no LLM/network call -- semantic search on
# paper summaries costs nothing extra beyond the summarize_paper call already made.
_embedder = get_embedder(EmbeddingConfig())

_memory_store: MemoryStore | None = None


def _get_memory_store() -> MemoryStore:
    """Lazily construct the singleton MemoryStore (mirrors how `llm` is built once at
    module load). MemoryStore's own __init__ creates MEMORY_DB's parent dir and runs its
    idempotent schema migration, so no separate _init step is needed here."""
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore(str(MEMORY_DB))
    return _memory_store


class PaperSummary(BaseModel):
    title: str
    main_objective: str
    key_methodology: str
    resulting_metrics: str


def extract_pdf_text(path: Path, max_chars: int = MAX_CHARS) -> str:
    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return text[:max_chars]


def _version_num(stem: str) -> int:
    """Extract the trailing "vN" version number from a filename stem, or -1 if there isn't one."""
    match = re.search(r"v(\d+)$", stem)
    return int(match.group(1)) if match else -1


def select_papers(candidates: list[Path], count: int | None, arxiv_ids: list[str]) -> list[Path]:
    """Narrow a sorted candidate list down to what should actually be used.

    A given count takes precedence over arxiv_ids entirely (mirrors simulator.py's
    fetch-stage precedence): if count is set, just take the first `count` candidates
    and ignore arxiv_ids. Otherwise, if arxiv_ids is non-empty, keep only candidates
    matching one of those IDs:
    - A versioned ID like "2608.11597v1" is an exact stem match against that one file.
    - An unversioned ID like "2608.11597" matches on the filename stem with its trailing
      "vN" stripped, but selects only the highest-versioned matching file rather than
      every version -- simulator.py keeps an old PDF on disk if arXiv later serves a
      newer version of the same paper (it only skips re-downloading, never deletes), so
      without this an unversioned ID could otherwise select both v1 and v2 and produce
      duplicate summaries/evaluations plus extra paid LLM calls.
    This is deliberately not a substring match -- "2608.1159" must not match
    "2608.11597v1.pdf", which it would under `in`.

    IDs have their "/" replaced with "_" before matching, mirroring how simulator.py's
    _download_entries encodes filenames from the arXiv Atom feed's id URL. This must keep
    (not drop) an old-style archive prefix like "hep-th/" -- "hep-th/9901001" and
    "hep-ph/9901001" are different papers that happen to share a numeric suffix, so
    stripping the prefix instead of encoding it would let one collide with or shadow the
    other's file.
    """
    if count is not None:
        return candidates[:count]
    if arxiv_ids:
        normalized_ids = {aid.replace("/", "_") for aid in arxiv_ids}
        versioned_ids = {nid for nid in normalized_ids if re.search(r"v\d+$", nid)}
        unversioned_ids = normalized_ids - versioned_ids

        latest_by_base: dict[str, Path] = {}
        for p in candidates:
            base = re.sub(r"v\d+$", "", p.stem)
            if base not in unversioned_ids:
                continue
            current = latest_by_base.get(base)
            if current is None or _version_num(p.stem) > _version_num(current.stem):
                latest_by_base[base] = p

        selected = set(latest_by_base.values()) | {p for p in candidates if p.stem in versioned_ids}
        return [p for p in candidates if p in selected]
    return candidates


class ResearchAgent(Agent, llm=llm):
    """You are a meticulous research assistant that reads academic papers and extracts structured findings."""

    papers_processed: int = 0

    def _init_state_db(self) -> None:
        """Create the processed-papers tracking table if it doesn't exist yet (deterministic, idempotent)."""
        PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(STATE_DB) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_papers (
                    filename TEXT PRIMARY KEY,
                    model_slug TEXT,
                    title TEXT,
                    main_objective TEXT,
                    key_methodology TEXT,
                    resulting_metrics TEXT,
                    processed_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def get_local_papers(self) -> list[Path]:
        """Scan PAPERS_DIR for PDFs that have not been processed yet (deterministic, no LLM call).

        Narrowed by NUM_PAPERS / ARXIV_PAPER_IDS via select_papers -- see its docstring
        for the precedence rule between the two.

        For an ARXIV_PAPER_IDS-driven, unversioned-ID selection, latest-version picking
        must run over every matching file on disk, seen or not, before the seen filter is
        applied -- not the other way around. Otherwise, once the latest version has been
        processed once (and is therefore "seen"), a later run for the same unversioned ID
        would only see the older, unprocessed version left over on disk and select that
        one as if it were new, causing exactly the duplicate paid re-summarize this ID
        matching is meant to prevent.
        """
        self._init_state_db()
        if not PAPERS_DIR.exists():
            return []
        with sqlite3.connect(STATE_DB) as conn:
            seen = {row[0] for row in conn.execute("SELECT filename FROM processed_papers")}
        all_candidates = sorted(PAPERS_DIR.glob("*.pdf"))
        if NUM_PAPERS is None and ARXIV_PAPER_IDS:
            selected = select_papers(all_candidates, None, ARXIV_PAPER_IDS)
            return [p for p in selected if p.name not in seen]
        pending = [p for p in all_candidates if p.name not in seen]
        return select_papers(pending, NUM_PAPERS, ARXIV_PAPER_IDS)

    def lookup_paper(self, filename: str) -> dict | None:
        """Look up a previously processed paper by filename, regardless of which model processed it."""
        self._init_state_db()
        with sqlite3.connect(STATE_DB) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM processed_papers WHERE filename = ?", (filename,)
            ).fetchone()
        return dict(row) if row else None

    def _record_summary(self, filename: str, summary: PaperSummary) -> None:
        with sqlite3.connect(STATE_DB) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO processed_papers
                (filename, model_slug, title, main_objective, key_methodology, resulting_metrics)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    filename,
                    MODEL_SLUG,
                    summary.title,
                    summary.main_objective,
                    summary.key_methodology,
                    summary.resulting_metrics,
                ),
            )
        self.papers_processed += 1
        try:
            self._record_memory(filename, summary)
        except Exception:
            pass

    def _record_memory(self, filename: str, summary: PaperSummary) -> None:
        """Embed the paper's summary and store it for later semantic similarity search.

        Deterministic -- the hashing embedder makes no LLM/network call. Uses the
        filename as the Memory id (not the default random id) so INSERT OR REPLACE
        semantics match processed_papers: reprocessing a file under a different model
        overwrites its memory the same way it overwrites its SQLite row, instead of
        accumulating duplicates.
        """
        content = "\n".join(
            [summary.title, summary.main_objective, summary.key_methodology, summary.resulting_metrics]
        )
        memory = Memory(id=filename, type=MemoryType.INFO, title=summary.title, content=content)
        _get_memory_store().add(memory, embedding=_embedder.embed(content))

    def find_similar_papers(self, query: str, k: int = 5) -> list[dict]:
        """Semantic search over previously processed papers' summaries.

        Deterministic, no LLM call -- uses the same offline hashing embedder as
        _record_memory. Separate from the exact-match processed_papers dedup table;
        this is for "find papers similar to X", not "have I processed this file".
        """
        if k <= 0:
            return []
        store = _get_memory_store()
        hits = store.knn(_embedder.embed(query), k)
        results = []
        for memory_id, score in hits:
            mem = store.get(memory_id)
            if mem is not None:
                results.append({"filename": mem.id, "title": mem.title, "score": score})
        return results

    async def summarize_paper(self, text: str) -> PaperSummary:
        """Read the paper text and extract structured findings.

        - title: the paper's title only, verbatim. Never include author names or affiliations.
        - main_objective: a one- or two-sentence synthesis, in your own words, of the problem the
          paper addresses and what it sets out to do. Do not copy a sentence verbatim from the text.
        - key_methodology: a concise description of the core method/approach used.
        - resulting_metrics: the key quantitative results reported, with units/context (e.g.
          "98.5% accuracy on X benchmark"), or "not reported" if the excerpt doesn't include any.
        """
        ...

    async def process_pending_papers(self) -> list[PaperSummary]:
        """Deterministic orchestration: summarize every unprocessed PDF and persist the results."""
        summaries = []
        for pdf_path in self.get_local_papers():
            text = extract_pdf_text(pdf_path)
            summary = await self.summarize_paper(text)
            self._record_summary(pdf_path.name, summary)
            summaries.append(summary)
        return summaries


async def main() -> None:
    agent = ResearchAgent()
    summaries = await agent.process_pending_papers()
    for summary in summaries:
        print(summary.model_dump_json(indent=2))
    print(json.dumps({"model_slug": MODEL_SLUG, "papers_processed": agent.papers_processed}))


if __name__ == "__main__":
    asyncio.run(main())
