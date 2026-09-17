from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import time


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, int(q * len(ordered) + 0.999999) - 1))
    return ordered[position]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare HNSW retrieval with the exact Flat development baseline."
    )
    parser.add_argument("--flat-index", type=Path, required=True)
    parser.add_argument("--hnsw-index", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--arrow-cache-dir", type=Path)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.threads <= 0
        or args.top_k <= 0
        or args.ef_search <= 0
        or args.concurrency <= 0
    ):
        raise ValueError("threads, concurrency, top-k, and efSearch must be positive")

    import faiss
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    faiss.omp_set_num_threads(args.threads)
    corpus = load_dataset(
        "json",
        data_files=str(args.corpus.resolve()),
        split="train",
        cache_dir=(
            str(args.arrow_cache_dir.resolve()) if args.arrow_cache_dir is not None else None
        ),
    )
    queries = [
        json.loads(line)
        for line in args.queries.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not queries:
        raise ValueError("development query set is empty")
    exact = faiss.read_index(
        str(args.flat_index), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
    )
    approximate = faiss.read_index(str(args.hnsw_index))
    if getattr(approximate, "hnsw", None) is None:
        raise ValueError("approximate index is not HNSW")
    approximate.hnsw.efSearch = args.ef_search
    if int(exact.ntotal) != len(corpus) or int(approximate.ntotal) != len(corpus):
        raise ValueError("index rows and Arrow corpus rows differ")

    encoder = SentenceTransformer(args.model, device=args.device)
    vectors = encoder.encode(
        [f"query: {row['query']}" for row in queries],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    exact.search(vectors[:1], args.top_k)
    approximate.search(vectors[:1], args.top_k)
    exact_latencies: list[float] = []
    recalls: list[float] = []
    for vector in vectors:
        started = time.perf_counter()
        _, exact_indexes = exact.search(vector[None, :], args.top_k)
        exact_latencies.append(time.perf_counter() - started)
        _, hnsw_indexes = approximate.search(vector[None, :], args.top_k)
        exact_rows = {int(value) for value in exact_indexes[0] if int(value) >= 0}
        hnsw_rows = {int(value) for value in hnsw_indexes[0] if int(value) >= 0}
        recalls.append(len(exact_rows & hnsw_rows) / max(1, len(exact_rows)))

    def timed_hnsw_search(vector):
        started = time.perf_counter()
        approximate.search(vector[None, :], args.top_k)
        return time.perf_counter() - started

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        hnsw_latencies = list(pool.map(timed_hnsw_search, vectors))

    result = {
        "schema_version": 2,
        "flat_index_sha256": sha256_file(args.flat_index),
        "hnsw_index_sha256": sha256_file(args.hnsw_index),
        "corpus_sha256": sha256_file(args.corpus),
        "arrow_fingerprint": str(corpus._fingerprint),
        "passage_count": len(corpus),
        "query_count": len(queries),
        "top_k": args.top_k,
        "hnsw_m": int(approximate.hnsw.nb_neighbors(1)),
        "hnsw_ef_search": args.ef_search,
        "hnsw_search_concurrency": args.concurrency,
        "recall_at_k_mean": statistics.fmean(recalls),
        "exact_latency_s_p50": percentile(exact_latencies, 0.50),
        "exact_latency_s_p95": percentile(exact_latencies, 0.95),
        "hnsw_latency_s_p50": percentile(hnsw_latencies, 0.50),
        "hnsw_latency_s_p95": percentile(hnsw_latencies, 0.95),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
