"""ResearchAgent: reads PDFs from /data/papers and extracts structured findings using NOOA."""

import asyncio
import json
import os
import sqlite3
from pathlib import Path

from pydantic import BaseModel
from pypdf import PdfReader

from nooa import Agent
from nooa.unifiedllm.registry import get_llm_client

MODEL_SLUG = os.environ.get("ACTIVE_MODEL_SLUG", "nvidia/nemotron-3-nano-30b-a3b")
PAPERS_DIR = Path(os.environ.get("PAPERS_DIR", "/data/papers"))
STATE_DB = PAPERS_DIR / ".agent_state.sqlite3"
MAX_CHARS = 20_000

llm = get_llm_client(f"openrouter/{MODEL_SLUG}")


class PaperSummary(BaseModel):
    title: str
    main_objective: str
    key_methodology: str
    resulting_metrics: str


def extract_pdf_text(path: Path, max_chars: int = MAX_CHARS) -> str:
    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return text[:max_chars]


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
        """Scan PAPERS_DIR for PDFs that have not been processed yet (deterministic, no LLM call)."""
        self._init_state_db()
        if not PAPERS_DIR.exists():
            return []
        with sqlite3.connect(STATE_DB) as conn:
            seen = {row[0] for row in conn.execute("SELECT filename FROM processed_papers")}
        return sorted(p for p in PAPERS_DIR.glob("*.pdf") if p.name not in seen)

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

    async def summarize_paper(self, text: str) -> PaperSummary:
        """Read the paper text and extract its Main Objective, Key Methodology, and Resulting Metrics."""
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
