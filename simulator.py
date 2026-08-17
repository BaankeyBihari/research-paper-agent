"""Downloads a few random recent arXiv papers into /data/papers for the agent to find."""

import os
import random
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

PAPERS_DIR = Path(os.environ.get("PAPERS_DIR", "/data/papers"))
ARXIV_API = "http://export.arxiv.org/api/query"
CATEGORIES = ["cs.AI", "cs.CL", "cs.LG", "stat.ML"]
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def fetch_random_papers(n: int = 3) -> None:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    category = random.choice(CATEGORIES)
    start = random.randint(0, 500)
    params = {
        "search_query": f"cat:{category}",
        "start": start,
        "max_results": n,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    resp = requests.get(ARXIV_API, params=params, timeout=30)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    entries = root.findall("atom:entry", ATOM_NS)

    for entry in entries:
        pdf_link = None
        for link in entry.findall("atom:link", ATOM_NS):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_link = link.attrib["href"]
                break
        if not pdf_link:
            continue

        arxiv_id = entry.find("atom:id", ATOM_NS).text.rsplit("/", 1)[-1]
        dest = PAPERS_DIR / f"{arxiv_id}.pdf"
        if dest.exists():
            continue

        pdf_resp = requests.get(pdf_link, timeout=60)
        pdf_resp.raise_for_status()
        dest.write_bytes(pdf_resp.content)
        print(f"Downloaded {dest.name}")
        time.sleep(1)  # be polite to arXiv


if __name__ == "__main__":
    fetch_random_papers(3)
