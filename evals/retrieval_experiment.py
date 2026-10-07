"""Run the search over a Langfuse dataset of test questions and record one experiment run per mode.

The dataset, codexa-verse-retrieval, is the 59 questions in verse_cases.json: a verse described in plain
English and the references it should find. The script creates or refreshes it before running.

    python evals/retrieval_experiment.py
"""

import json
import time
from pathlib import Path

from langfuse import Evaluation, get_client

from codexa_api.config import get_settings
from codexa_api.embedder import Embedder
from codexa_api.packs import load_packs
from codexa_api.search import Mode, Searcher

DATASET = "codexa-verse-retrieval"
CASES = Path(__file__).with_name("verse_cases.json")
WORKS = ["kjv", "bbe", "sblgnt", "wlc"]  # two English translations, the Greek NT and the Hebrew OT


def first_correct(refs: list[str], expected: list[str]) -> int | None:
    """Rank of the first correct reference. "Book 1:" in `expected` accepts any verse of that chapter."""
    for rank, ref in enumerate(refs):
        if any(ref.startswith(e) if e.endswith(":") else ref == e for e in expected):
            return rank
    return None


def score_item(*, output, expected_output, **_) -> list[Evaluation]:
    rank = first_correct(output["refs"], expected_output["refs"])
    found = rank is not None
    return [
        Evaluation(name="hit@1", value=float(found and rank < 1)),
        Evaluation(name="hit@5", value=float(found and rank < 5)),
        Evaluation(name="hit@10", value=float(found and rank < 10)),
        Evaluation(name="reciprocal_rank", value=1 / (rank + 1) if found else 0.0),
        Evaluation(name="latency_ms", value=round(output["latency_ms"], 1)),
    ]


def mean(item_results, name: str) -> float:
    values = [e.value for r in item_results for e in r.evaluations if e.name == name]
    return sum(values) / len(values) if values else 0.0


def score_run(*, item_results, **_) -> list[Evaluation]:
    return [
        Evaluation(name="hit@1_rate", value=mean(item_results, "hit@1")),
        Evaluation(name="hit@5_rate", value=mean(item_results, "hit@5")),
        Evaluation(name="hit@10_rate", value=mean(item_results, "hit@10")),
        Evaluation(name="mrr", value=mean(item_results, "reciprocal_rank")),
        Evaluation(name="mean_latency_ms", value=round(mean(item_results, "latency_ms"), 1)),
    ]


def search_task(searcher: Searcher, mode: Mode):
    def task(*, item, **_):
        start = time.perf_counter()
        hits = searcher.search(item.input["query"], mode=mode, limit=10)
        return {"refs": [h.ref for h in hits], "latency_ms": (time.perf_counter() - start) * 1000}

    return task


def upsert_dataset(langfuse) -> None:
    """Create the dataset, or refresh its items. Item ids come from the cases, so reruns don't duplicate."""
    langfuse.create_dataset(name=DATASET, description="Described verses and the references they should find")
    for case in json.loads(CASES.read_text()):
        langfuse.create_dataset_item(
            dataset_name=DATASET,
            id=f"verse-{case['id']}",
            input={"query": case["query"]},
            expected_output={"refs": case["expect"]},
            metadata={"case": case["id"], "category": case["category"]},
        )


def main() -> None:
    settings = get_settings()
    searcher = Searcher(load_packs(settings.data_dir, WORKS), Embedder(settings.model_dir))
    searcher.search("warm-up query about loving your neighbour")  # load the model before timing
    langfuse = get_client()
    upsert_dataset(langfuse)
    dataset = langfuse.get_dataset(DATASET)
    print(f"{DATASET}: {len(dataset.items)} questions\n")
    print("| Run | hit@1 | hit@5 | hit@10 | MRR | mean latency |\n|---|---|---|---|---|---|")
    for mode in ("keyword", "semantic", "hybrid"):
        result = dataset.run_experiment(
            name=f"search · {mode}",
            description=f"{mode} search over {', '.join(WORKS)}",
            task=search_task(searcher, mode),
            evaluators=[score_item],
            run_evaluators=[score_run],
            max_concurrency=1,
            metadata={"index": "sqlite-fts5-int8", "mode": mode},
        )
        r = result.item_results
        print(
            f"| {mode} | {mean(r, 'hit@1'):.0%} | {mean(r, 'hit@5'):.0%} | {mean(r, 'hit@10'):.0%} "
            f"| {mean(r, 'reciprocal_rank'):.3f} | {mean(r, 'latency_ms'):.1f} ms |"
        )
    langfuse.flush()


if __name__ == "__main__":
    main()
