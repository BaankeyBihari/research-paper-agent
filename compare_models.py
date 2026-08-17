"""Compares cheaper models against a stronger reference model on the same papers.

Runs summarize_paper across REFERENCE_SLUG and CANDIDATE_SLUGS for each paper via
eval_pipeline.Evaluator, treating the reference model's output as a silver-standard
answer and scoring the candidates against it with a difflib-based text-similarity
scorer (stdlib only -- no extra API calls or LLM-judge circularity). Results are
written to a .noo-eval.jsonl file under PAPERS_DIR, viewable in the trace viewer's
Evaluations tab, and summarized on stdout.

Usage:
    python compare_models.py
    COMPARE_NUM_PAPERS=5 python compare_models.py
    COMPARE_ARXIV_PAPER_IDS=2608.11597,2608.11590 python compare_models.py
    REFERENCE_MODEL_SLUG=anthropic/claude-3-5-sonnet python compare_models.py
"""

import asyncio
import difflib
import os
from pathlib import Path

from eval_pipeline import Evaluator, ScoreResult
from eval_pipeline.models import ScoringContext
from nooa.unifiedllm.registry import get_llm_client

from agent import PAPERS_DIR, PaperSummary, ResearchAgent, extract_pdf_text, select_papers

def _parse_int_env(name: str) -> int | None:
    """Parse an optional integer env var, failing with a clear message rather than
    a raw ValueError traceback if it's set to something non-numeric."""
    raw = os.environ.get(name) or None
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"{name}={raw!r} is not a valid integer") from None


COMPARE_ARXIV_PAPER_IDS = [
    i.strip() for i in os.environ.get("COMPARE_ARXIV_PAPER_IDS", "").split(",") if i.strip()
]
# Default to 2 papers only when neither knob is set, to preserve today's behavior --
# an explicit COMPARE_ARXIV_PAPER_IDS with no count should use exactly that list.
NUM_PAPERS = _parse_int_env("COMPARE_NUM_PAPERS")
if NUM_PAPERS is None and not COMPARE_ARXIV_PAPER_IDS:
    NUM_PAPERS = 2
# meta-llama/llama-3.1-70b-instruct (the original default) consistently fails to
# return valid structured output under NOOA's codeact tool-calling strategy via
# OpenRouter -- reproduces across multiple papers, unrelated to content/extraction
# quality. gpt-4o-mini is a verified-working, inexpensive substitute.
REFERENCE_SLUG = os.environ.get("REFERENCE_MODEL_SLUG", "openai/gpt-4o-mini")
CANDIDATE_SLUGS = [
    s.strip()
    for s in os.environ.get(
        "CANDIDATE_MODEL_SLUGS",
        "nvidia/nemotron-3-nano-30b-a3b,nvidia/nemotron-3.5-lightning",
    ).split(",")
    if s.strip()
]

PAPER_SUMMARY_FIELDS = list(PaperSummary.model_fields)


def _as_field_dict(value: PaperSummary | dict | None) -> dict[str, str]:
    """Normalize a PaperSummary, an equivalent dict, or None into field->str.

    eval_pipeline may hand scorers a dict (e.g. after serialization) rather than the
    original PaperSummary instance, and ctx.actual is None if a candidate's run
    failed/timed out. Missing or non-string values become "" rather than raising, so a
    single failed sample scores as maximally dissimilar instead of aborting the run.
    """
    if value is None:
        data: dict = {}
    elif isinstance(value, PaperSummary):
        data = value.model_dump()
    elif isinstance(value, dict):
        data = value
    else:
        data = {}
    return {field: str(data[field]) if data.get(field) is not None else "" for field in PAPER_SUMMARY_FIELDS}


class PaperSimilarityScorer:
    """Scores a candidate PaperSummary against a reference one via per-field difflib ratio."""

    def score(self, ctx: ScoringContext) -> ScoreResult:
        expected = _as_field_dict(ctx.expected)
        actual = _as_field_dict(ctx.actual)
        field_scores = {
            field: round(
                difflib.SequenceMatcher(None, expected[field].lower(), actual[field].lower()).ratio(),
                3,
            )
            for field in PAPER_SUMMARY_FIELDS
        }
        overall = round(sum(field_scores.values()) / len(field_scores), 3)
        return ScoreResult(
            score=overall,
            reasoning=f"Average field similarity to reference: {overall}",
            metadata=field_scores,
        )


async def main() -> None:
    candidates = sorted(PAPERS_DIR.glob("*.pdf"))
    papers = select_papers(candidates, NUM_PAPERS, COMPARE_ARXIV_PAPER_IDS)
    if not papers:
        raise SystemExit(f"No PDFs found in {PAPERS_DIR}; run simulator.py first.")
    paper_texts: dict[Path, str] = {p: extract_pdf_text(p) for p in papers}

    print(f"Running reference model ({REFERENCE_SLUG}) on {len(papers)} paper(s)...", flush=True)
    reference_agent = ResearchAgent(llm=get_llm_client(f"openrouter/{REFERENCE_SLUG}"))
    reference_summaries: dict[Path, PaperSummary] = {
        path: await reference_agent.summarize_paper(text) for path, text in paper_texts.items()
    }

    models = {slug: get_llm_client(f"openrouter/{slug}") for slug in CANDIDATE_SLUGS}
    evaluator = Evaluator(
        models=models, output_dir=PAPERS_DIR / "eval_results", name="compare_models"
    )
    # Workaround for an eval_pipeline bug (commit 8622fc4, the version pinned in
    # requirements.txt): Evaluator.run() writes the run's metadata line from
    # self._model_metadata, which the plain Python API (this one, per its own README)
    # never populates -- only the YAML/from_config path does. Without this, run() raises
    # a pydantic ValidationError building EvalMetadata.models. Pre-populate it directly.
    # Filed upstream: https://github.com/NVIDIA-NeMo/labs-OO-Agents/issues/152 -- drop this
    # workaround once a fix lands.
    evaluator._model_metadata = {slug: {"id": slug, "model_name": slug} for slug in CANDIDATE_SLUGS}
    evaluator.add_test(
        name="summarize_paper",
        agent_class=ResearchAgent,
        method="summarize_paper",
        data=[
            {"kwargs": {"text": paper_texts[path]}, "expected": summary}
            for path, summary in reference_summaries.items()
        ],
        scorers=[PaperSimilarityScorer()],
    )

    print(f"Running candidates ({', '.join(CANDIDATE_SLUGS)})...", flush=True)
    results = await evaluator.run(models=CANDIDATE_SLUGS)

    print(f"\nReference model: {REFERENCE_SLUG}")
    print(results.summary())
    for slug in CANDIDATE_SLUGS:
        scores = [
            r.scores["PaperSimilarityScorer"].score for r in results.results if r.model == slug
        ]
        if scores:
            print(f"{slug}: avg similarity to reference = {sum(scores) / len(scores):.3f} ({len(scores)} papers)")
    if results.output_file:
        print(f"\nFull results: {results.output_file} (viewable in the trace viewer's Evaluations tab)")


if __name__ == "__main__":
    asyncio.run(main())
