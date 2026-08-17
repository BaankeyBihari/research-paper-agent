"""Compares cheaper models against a stronger reference model on the same papers.

Runs summarize_paper across REFERENCE_SLUG and CANDIDATE_SLUGS for each paper, treats
the reference model's output as a silver-standard answer, and scores the candidates
against it with simple text similarity (difflib, stdlib only -- no extra API calls or
LLM-judge circularity). Prints a per-paper report and an aggregate summary, and saves
the full report as JSON next to the papers.

Usage:
    python compare_models.py
    COMPARE_NUM_PAPERS=5 python compare_models.py
    REFERENCE_MODEL_SLUG=meta-llama/llama-3.1-70b-instruct:free python compare_models.py
"""

import asyncio
import difflib
import json
import os
import time
from pathlib import Path

from nooa import Agent
from nooa.unifiedllm.registry import get_llm_client

from agent import PAPERS_DIR, PaperSummary, extract_pdf_text

NUM_PAPERS = int(os.environ.get("COMPARE_NUM_PAPERS", "2"))
REFERENCE_SLUG = os.environ.get("REFERENCE_MODEL_SLUG", "meta-llama/llama-3.1-70b-instruct")
CANDIDATE_SLUGS = os.environ.get(
    "CANDIDATE_MODEL_SLUGS",
    "nvidia/nemotron-3-nano-30b-a3b,nvidia/nemotron-3.5-lightning",
).split(",")
MODEL_SLUGS = [REFERENCE_SLUG, *CANDIDATE_SLUGS]

def make_agent(model_slug: str) -> Agent:
    """Build a single-purpose Agent bound to model_slug, matching ResearchAgent's summarize_paper.

    The docstring below is kept identical to agent.py's summarize_paper -- update both together.
    """
    llm = get_llm_client(f"openrouter/{model_slug}")

    class _CompareAgent(Agent, llm=llm):
        """You are a meticulous research assistant that reads academic papers and extracts structured findings."""

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

    return _CompareAgent()


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def score_against_reference(candidate: PaperSummary, reference: PaperSummary) -> dict:
    fields = ["title", "main_objective", "key_methodology", "resulting_metrics"]
    scores = {f: round(similarity(getattr(candidate, f), getattr(reference, f)), 3) for f in fields}
    scores["overall"] = round(sum(scores.values()) / len(fields), 3)
    return scores


async def compare_paper(pdf_path: Path, agents: dict[str, Agent]) -> dict:
    text = extract_pdf_text(pdf_path)
    results = {}
    for slug, agent in agents.items():
        start = time.perf_counter()
        summary = await agent.summarize_paper(text)
        results[slug] = {"summary": summary, "elapsed_s": round(time.perf_counter() - start, 2)}

    reference = results[REFERENCE_SLUG]["summary"]
    report = {"filename": pdf_path.name, "reference_model": REFERENCE_SLUG, "results": {}}
    for slug, r in results.items():
        entry = {"elapsed_s": r["elapsed_s"], "summary": r["summary"].model_dump()}
        if slug != REFERENCE_SLUG:
            entry["similarity_vs_reference"] = score_against_reference(r["summary"], reference)
        report["results"][slug] = entry
    return report


def print_aggregate_summary(reports: list[dict]) -> None:
    print("\n=== Aggregate summary ===")
    ref_avg_latency = sum(r["results"][REFERENCE_SLUG]["elapsed_s"] for r in reports) / len(reports)
    print(f"{REFERENCE_SLUG} (reference): avg latency = {ref_avg_latency:.2f}s")
    for slug in CANDIDATE_SLUGS:
        avg_overall = sum(r["results"][slug]["similarity_vs_reference"]["overall"] for r in reports) / len(reports)
        avg_latency = sum(r["results"][slug]["elapsed_s"] for r in reports) / len(reports)
        print(f"{slug}: avg similarity to reference = {avg_overall:.3f}, avg latency = {avg_latency:.2f}s")


async def main() -> None:
    papers = sorted(PAPERS_DIR.glob("*.pdf"))[:NUM_PAPERS]
    if not papers:
        raise SystemExit(f"No PDFs found in {PAPERS_DIR}; run simulator.py first.")

    agents = {slug: make_agent(slug) for slug in MODEL_SLUGS}

    reports = []
    for pdf_path in papers:
        print(f"Comparing models on {pdf_path.name}...", flush=True)
        report = await compare_paper(pdf_path, agents)
        reports.append(report)
        print(json.dumps(report, indent=2))

    print_aggregate_summary(reports)

    out_path = PAPERS_DIR / "model_comparison.json"
    out_path.write_text(json.dumps(reports, indent=2))
    print(f"\nSaved full comparison to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
