"""Literature search in OpenAlex for the positioning of spicefault.

    python scripts/openalex_search.py

Each query is a boolean search over title and abstract. For every query the script
stores the works found (most relevant first, up to MAX_PER_QUERY) with the abstract
rebuilt from the inverted index of OpenAlex, and then a master table without
duplicates that says in which queries each work appears. Results go to
docs/literature/: one JSON per query, `master.csv`, and `search.json` with the
queries, the date and the counts, so that the search can be repeated.

OpenAlex asks for a contact address: set OPENALEX_MAILTO.
"""

from __future__ import annotations

import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "literature"
MAILTO = os.environ.get("OPENALEX_MAILTO", "")
MAX_PER_QUERY = 200
PER_PAGE = 200
FIELDS = (
    "id,doi,title,publication_year,type,primary_location,cited_by_count,abstract_inverted_index"
)

# Query strings must not contain commas: OpenAlex separates filters with them.
QUERIES = {
    "Q1_spice_fault_injection_tools": {
        "question": "Frameworks or tools that automate fault injection or fault simulation "
        "of analog circuits in SPICE",
        "q": '("fault injection" OR "fault simulation" OR "defect simulation" OR "fault simulator") '
        'AND ("SPICE" OR "ngspice" OR "HSPICE" OR "Spectre" OR "circuit simulator") '
        'AND ("analog" OR "mixed-signal" OR "analogue") '
        'AND ("framework" OR "tool" OR "automated" OR "automatic" OR "environment")',
        "years": (2000, 2026),
    },
    "Q2_python_spice_automation": {
        "question": "Open tools that drive SPICE simulators programmatically",
        "q": '("PySpice" OR "ngspice" OR "LTspice" OR "PyLTSpice" OR "Xyce") '
        'AND ("Python" OR "open-source" OR "open source") '
        'AND ("Monte Carlo" OR "fault" OR "automation" OR "framework" OR "dataset")',
        "years": (2010, 2026),
    },
    "Q3_benchmark_circuits_fault_diagnosis": {
        "question": "Which circuits the analog fault-diagnosis literature uses as benchmarks",
        "q": '("fault diagnosis" OR "fault classification") AND ("analog circuit" OR "analog circuits" '
        'OR "analogue circuit") AND ("Sallen-Key" OR "Sallen Key" OR "state variable filter" '
        'OR "biquad" OR "four-opamp" OR "leapfrog filter" OR "benchmark circuit")',
        "years": (2005, 2026),
    },
    "Q4_tolerance_and_detectability": {
        "question": "Studies of how component tolerance affects fault detectability, testability "
        "or ambiguity in analog circuits",
        "q": '("fault detectability" OR "detectability" OR "testability" OR "ambiguity group" '
        'OR "fault coverage") AND ("analog circuit" OR "analog circuits" OR "analogue circuit") '
        'AND ("tolerance" OR "tolerances" OR "Monte Carlo" OR "process variation")',
        "years": (2000, 2026),
    },
    "Q5_simulated_datasets_and_reproducibility": {
        "question": "How simulated fault datasets for analog diagnosis are generated and whether "
        "the generation is reported or shared",
        "q": '("fault diagnosis") AND ("analog circuit" OR "analog circuits") AND ("SPICE" OR "PSpice" '
        'OR "simulation data" OR "simulated data") AND ("dataset" OR "data generation" '
        'OR "reproducible" OR "reproducibility" OR "open-source" OR "publicly available")',
        "years": (2010, 2026),
    },
    "Q6_analog_defect_coverage": {
        "question": "Defect-oriented analog test and analog fault coverage, including the IEEE "
        "P2427 effort and commercial simulators",
        "q": '("analog defect" OR "defect-oriented" OR "analog fault simulation" OR "P2427" '
        'OR "defect coverage") AND ("analog" OR "mixed-signal") AND ("simulation" OR "coverage")',
        "years": (2005, 2026),
    },
    "Q7_circuit_level_fault_injection_for_safety": {
        "question": "Circuit-level fault injection used for reliability or functional-safety "
        "assessment",
        "q": '("fault injection") AND ("analog" OR "mixed-signal" OR "circuit-level" OR "SPICE") '
        'AND ("reliability" OR "functional safety" OR "ISO 26262" OR "FMEDA" OR "diagnostic coverage")',
        "years": (2005, 2026),
    },
    "Q8_open_fault_injection_software": {
        "question": "Open-source or scripted fault-injection software for analog circuits",
        "q": '("fault injection" OR "fault simulation" OR "fault modeling") AND ("analog circuit" '
        'OR "analog circuits" OR "SPICE" OR "netlist") AND ("open-source" OR "open source" '
        'OR "Python" OR "MATLAB" OR "script" OR "toolbox" OR "software tool")',
        "years": (2005, 2026),
    },
}


def abstract_of(index: dict | None) -> str:
    """The abstract, rebuilt from OpenAlex's {word: [positions]} form."""
    if not index:
        return ""
    words = sorted((position, word) for word, positions in index.items() for position in positions)
    return " ".join(word for _, word in words)


def get_json(url: str) -> dict:
    """GET with a few patient retries: OpenAlex answers 429 when asked too fast."""
    for wait in (0, 5, 20, 60):
        time.sleep(wait)
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != 429:
                raise
    raise RuntimeError("OpenAlex keeps answering 429 (too many requests); try again later")


def fetch(query: str, years: tuple[int, int] | None) -> tuple[int, list[dict]]:
    """(number of works that match, the most relevant of them)."""
    filters = [f"title_and_abstract.search:{query}"]
    if years:
        filters.append(f"publication_year:{years[0]}-{years[1]}")
    works, cursor, total = [], "*", 0
    while cursor and len(works) < MAX_PER_QUERY:
        parameters = {"filter": ",".join(filters), "per-page": PER_PAGE, "cursor": cursor,
                      "select": FIELDS, "sort": "relevance_score:desc"}  # fmt: skip
        if MAILTO:
            parameters["mailto"] = MAILTO
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(parameters)
        page = get_json(url)
        total = page["meta"]["count"]
        works += page["results"]
        cursor = page["meta"].get("next_cursor") if page["results"] else None
        time.sleep(1.0)
    return total, works[:MAX_PER_QUERY]


def record(work: dict) -> dict:
    source = ((work.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
    return {
        "id": work["id"].rsplit("/", 1)[-1],
        "doi": work.get("doi") or "",
        "year": work.get("publication_year"),
        "type": work.get("type") or "",
        "venue": source,
        "cited_by": work.get("cited_by_count", 0),
        "title": work.get("title") or "",
        "abstract": abstract_of(work.get("abstract_inverted_index")),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    master: dict[str, dict] = {}
    summary = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "source": "OpenAlex, title_and_abstract.search", "max_per_query": MAX_PER_QUERY,
               "queries": {}}  # fmt: skip
    for name, spec in QUERIES.items():
        total, works = fetch(spec["q"], spec["years"])
        records = [record(w) for w in works]
        (OUT / f"{name}.json").write_text(json.dumps(records, indent=1, ensure_ascii=False))
        summary["queries"][name] = {**spec, "matches": total, "stored": len(records)}
        for rank, item in enumerate(records, start=1):
            entry = master.setdefault(item["id"], {**item, "queries": [], "best_rank": rank})
            entry["queries"].append(name.split("_")[0])
            entry["best_rank"] = min(entry["best_rank"], rank)
        print(f"{name}: {total} matches, {len(records)} stored")
    summary["distinct_works"] = len(master)
    (OUT / "search.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    columns = ["id", "year", "cited_by", "best_rank", "queries", "venue", "type", "title", "doi"]
    with (OUT / "master.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for item in sorted(master.values(), key=lambda e: (e["best_rank"], -e["cited_by"])):
            writer.writerow({**item, "queries": " ".join(item["queries"])})
    print(f"{len(master)} distinct works; results in {OUT}")


if __name__ == "__main__":
    main()
