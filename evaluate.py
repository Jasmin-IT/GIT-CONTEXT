"""
GitContext evaluation for a selected repository and its gold-label dataset.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from api import initialize_backend  # reuses your exact pipeline setup

KS = (1, 3, 5)
CANDIDATES = 10  # same as top_k in api.py before reranking


def overlaps(meta, gold):
    if not paths_match(meta["file_path"], gold["file_path"]):
        return False
    if "start_line" not in gold and "end_line" not in gold:
        return True
    return (
        int(meta["start_line"]) <= int(gold["end_line"])
        and int(gold["start_line"]) <= int(meta["end_line"])
    )


def paths_match(left, right):
    left = str(left).replace("\\", "/").removeprefix("./")
    right = str(right).replace("\\", "/").removeprefix("./")
    return (
        left == right
        or left.endswith("/" + right)
        or right.endswith("/" + left)
    )


def unknown_gold_paths(dataset, repository_files):
    repository_paths = [file.file_path for file in repository_files]
    return sorted({
        label["file_path"]
        for item in dataset
        for label in item["expected"]
        if not any(paths_match(label["file_path"], path) for path in repository_paths)
    })


def is_relevant(chunk, expected):
    return any(overlaps(chunk["metadata"], g) for g in expected)


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def retrieve(backend, question, mode, rewrite):
    """mode: bm25 | vector | hybrid | rerank. Returns (ranked_chunks, seconds)."""
    t0 = time.perf_counter()
    llm = backend["llm"]
    queries = llm.generate_search_queries(question, []) if rewrite else [question]
    embeddings = [backend["embedding_model"].embed_query(q) for q in queries]

    weights = {"bm25": (0.0, 1.0), "vector": (1.0, 0.0)}.get(mode, (1.0, 1.0))
    chunks = backend["hybrid_retriever"].search_multiple(
        queries=queries,
        query_embeddings=embeddings,
        top_k=CANDIDATES,
        rrf_k=60,
        repository_id=backend["repository_id"],
        vector_weight=weights[0],
        bm25_weight=weights[1],
    )
    if mode == "rerank" and chunks:
        chunks = backend["reranker"].rerank(query=question, retrieved_chunks=chunks, top_k=5)
    return chunks, time.perf_counter() - t0


def evaluate_mode(backend, dataset, mode, rewrite):
    recall = {k: 0.0 for k in KS}
    hit_rate = {k: 0 for k in KS}
    precision = {k: 0.0 for k in KS}
    rr_total, latencies, misses = 0.0, [], []
    details = []

    for item in dataset:
        chunks, secs = retrieve(backend, item["question"], mode, rewrite)
        latencies.append(secs * 1000)
        flags = [is_relevant(c, item["expected"]) for c in chunks]
        details.append({
            "question": item["question"],
            "expected": item["expected"],
            "top_results": [
                {
                    "file_path": chunk["metadata"]["file_path"],
                    "start_line": chunk["metadata"]["start_line"],
                    "end_line": chunk["metadata"]["end_line"],
                    "relevant": flags[index],
                }
                for index, chunk in enumerate(chunks[:5])
            ],
        })

        for k in KS:
            top = flags[:k]
            hit_rate[k] += any(top)
            recall[k] += sum(
                any(is_relevant(chunk, [gold]) for chunk in chunks[:k])
                for gold in item["expected"]
            ) / len(item["expected"])
            precision[k] += sum(top) / k
        first = next((i for i, f in enumerate(flags, 1) if f), None)
        rr_total += 1 / first if first else 0
        if not any(flags[:5]):
            misses.append(item["question"])

    n = len(dataset)
    return {
        "mode": mode,
        **{f"recall@{k}": recall[k] / n for k in KS},
        **{f"hit_rate@{k}": hit_rate[k] / n for k in KS},
        **{f"precision@{k}": precision[k] / n for k in KS},
        "mrr": rr_total / n,
        "latency_p50_ms": statistics.median(latencies),
        "latency_p95_ms": percentile(latencies, 95),
        "misses": misses,
        "question_results": details,
    }


def evaluate_answers(backend, dataset, delay):
    """Measure citation evidence validity, gold-label accuracy, and latency."""
    llm = backend["llm"]
    valid_answers = 0
    gold_citations = 0
    citation_count = 0
    answered = 0
    retrieval_latencies = []
    end_to_end_latencies = []

    for item in dataset:
        end_to_end_start = time.perf_counter()
        chunks, retrieval_secs = retrieve(
            backend, item["question"], "rerank", rewrite=False
        )
        retrieval_latencies.append(retrieval_secs * 1000)
        if not chunks:
            continue

        answer = llm.generate_answer(query=item["question"], retrieved_chunks=chunks, chat_history=[])
        end_to_end_latencies.append(
            (time.perf_counter() - end_to_end_start) * 1000
        )
        answered += 1

        normalized_answer = llm.normalize_citations(answer)
        citations = llm.CITATION_PATTERN.findall(normalized_answer)
        if llm.validate_citations(normalized_answer, chunks):
            valid_answers += 1

        citation_count += len(citations)
        for file_path, start_line, end_line in citations:
            citation = {
                "metadata": {
                    "file_path": file_path.strip(),
                    "start_line": int(start_line),
                    "end_line": int(end_line or start_line),
                }
            }
            if is_relevant(citation, item["expected"]):
                gold_citations += 1

        if delay:
            time.sleep(delay)

    return {
        "questions_answered": answered,
        "citation_validity_rate": valid_answers / answered if answered else 0,
        "citation_accuracy": gold_citations / citation_count if citation_count else 0,
        "citations_checked": citation_count,
        "gold_citations": gold_citations,
        "retrieval_latency_p50_ms": (
            statistics.median(retrieval_latencies) if retrieval_latencies else 0
        ),
        "end_to_end_latency_p50_ms": (
            statistics.median(end_to_end_latencies)
            if end_to_end_latencies
            else 0
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    repository_group = parser.add_mutually_exclusive_group()
    repository_group.add_argument(
        "--repository-url", help="GitHub repository URL to index and evaluate"
    )
    repository_group.add_argument(
        "--repository-path", help="local repository directory to index and evaluate"
    )
    parser.add_argument(
        "--eval-file",
        help="JSON gold-label dataset for the selected repository",
    )
    parser.add_argument("--rewrite", action="store_true", help="use LLM query rewriting")
    parser.add_argument("--answers", action="store_true", help="measure citation validity and accuracy")
    parser.add_argument("--delay", type=float, default=4.0, help="seconds between LLM calls")
    parser.add_argument("--list-chunks", action="store_true")
    args = parser.parse_args()

    if args.delay < 0:
        parser.error("--delay must be non-negative")

    if args.list_chunks:
        backend = initialize_backend(
            repository_url=args.repository_url,
            repository_path=args.repository_path,
        )
        from src.gitcontext.chunking import chunk_repository_file
        for f in backend["repository_files"]:
            for c in chunk_repository_file(f, repository_id=backend["repository_id"]):
                print(f"{c.file_path}  lines {c.start_line}-{c.end_line}  [{c.chunk_type}] {c.chunk_name}")
        return

    if not args.eval_file:
        parser.error(
            "Provide --eval-file for the selected repository. "
            "Use --list-chunks first to prepare its gold labels."
        )

    try:
        dataset = json.loads(Path(args.eval_file).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"Could not read evaluation file {args.eval_file!r}: {error}")

    if not isinstance(dataset, list) or not dataset:
        parser.error("Evaluation file must contain a non-empty JSON array.")

    for index, item in enumerate(dataset):
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("question"), str)
            or not item["question"].strip()
            or not isinstance(item.get("expected"), list)
            or not item["expected"]
        ):
            parser.error(
                f"Invalid evaluation item at index {index}: "
                "each item needs a question and non-empty expected list."
            )
        for expected in item["expected"]:
            if (
                not isinstance(expected, dict)
                or not isinstance(expected.get("file_path"), str)
                or not expected["file_path"].strip()
            ):
                parser.error(
                    f"Invalid expected label at index {index}: "
                    "each label needs a file_path."
                )
            has_start = "start_line" in expected
            has_end = "end_line" in expected
            if has_start != has_end:
                parser.error(
                    f"Expected labels at index {index} must include both "
                    "start_line and end_line."
                )
            if has_start:
                try:
                    start_line = int(expected["start_line"])
                    end_line = int(expected["end_line"])
                except (TypeError, ValueError):
                    parser.error(
                        f"Invalid line range in expected labels at index {index}."
                    )
                if start_line < 1 or end_line < start_line:
                    parser.error(
                        f"Invalid line range in expected labels at index {index}."
                    )

    backend = initialize_backend(
        repository_url=args.repository_url,
        repository_path=args.repository_path,
    )
    unknown_paths = unknown_gold_paths(dataset, backend["repository_files"])
    if unknown_paths:
        parser.error(
            "Gold labels reference files not found in the selected repository: "
            + ", ".join(unknown_paths)
        )

    print(f"\nEvaluating {len(dataset)} questions on {backend['repository_id']}\n")

    rows = [evaluate_mode(backend, dataset, m, args.rewrite) for m in ("bm25", "vector", "hybrid", "rerank")]

    header = f"{'mode':<8} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'P@5':>6} {'MRR':>6} {'p50 ms':>8} {'p95 ms':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['mode']:<8} {r['recall@1']:>6.2f} {r['recall@3']:>6.2f} {r['recall@5']:>6.2f} "
              f"{r['precision@5']:>6.2f} {r['mrr']:>6.2f} {r['latency_p50_ms']:>8.0f} {r['latency_p95_ms']:>8.0f}")

    final = rows[-1]
    if final["misses"]:
        print("\nQuestions missed at Recall@5 (rerank):")
        for q in final["misses"]:
            print("  -", q)

    report = {
        "dataset_questions": len(dataset),
        "retrieval": rows,
        "comparison": {
            "hybrid_vs_rerank": {
                "recall@5_delta": (
                    rows[-1]["recall@5"] - rows[2]["recall@5"]
                ),
                "mrr_delta": rows[-1]["mrr"] - rows[2]["mrr"],
                "latency_p50_ms_delta": (
                    rows[-1]["latency_p50_ms"] - rows[2]["latency_p50_ms"]
                ),
            }
        },
    }
    if args.answers:
        report["answers"] = evaluate_answers(backend, dataset, args.delay)
        print("\nAnswer-level:", json.dumps(report["answers"], indent=2))

    Path("eval_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nSaved eval_results.json")


if __name__ == "__main__":
    main()