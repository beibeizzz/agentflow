from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the pinned Wikipedia-18 E5 IndexHNSWFlat index."
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--arrow-cache-dir", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--index-revision", required=True)
    parser.add_argument("--index-output", type=Path, required=True)
    parser.add_argument("--flat-reference-output", type=Path)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--ef-construction", type=int, default=256)
    args = parser.parse_args()
    if args.m != 64:
        raise ValueError("the formal index requires HNSW M=64")
    if args.batch_size <= 0 or args.threads <= 0 or args.ef_construction <= 0:
        raise ValueError("batch size, threads, and efConstruction must be positive")

    import faiss
    import numpy as np
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    faiss.omp_set_num_threads(args.threads)
    encoder = SentenceTransformer(args.model, device=args.device)
    dimension = int(encoder.get_sentence_embedding_dimension())
    flat_reference = (
        faiss.IndexFlatIP(dimension) if args.flat_reference_output is not None else None
    )
    index = None
    if flat_reference is None:
        index = faiss.IndexHNSWFlat(dimension, args.m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = args.ef_construction

    args.index_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    corpus = load_dataset(
        "json",
        data_files=str(args.corpus.resolve()),
        split="train",
        cache_dir=(
            str(args.arrow_cache_dir.resolve()) if args.arrow_cache_dir is not None else None
        ),
    )
    if len(corpus) <= 0 or "id" not in corpus.column_names or "contents" not in corpus.column_names:
        raise ValueError("Wikipedia corpus must contain nonempty id and contents columns")
    started = time.monotonic()
    for start in range(0, len(corpus), args.batch_size):
        stop = min(start + args.batch_size, len(corpus))
        rows = corpus[start:stop]
        if any(value is None or str(value) == "" for value in rows["id"]):
            raise ValueError(f"Wikipedia corpus contains an empty id near row {start}")
        texts = [f"passage: {value}" for value in rows["contents"]]
        vectors = encoder.encode(
            texts,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        target = flat_reference if flat_reference is not None else index
        target.add(np.asarray(vectors, dtype="float32"))
        if stop % 100_000 < args.batch_size or stop == len(corpus):
            print(json.dumps({"indexed_passages": stop}), flush=True)

    encoded_index = flat_reference if flat_reference is not None else index
    if int(encoded_index.ntotal) != len(corpus):
        raise ValueError("index rows and Arrow corpus rows are misaligned")
    flat_reference_sha256 = None
    if flat_reference is not None:
        args.flat_reference_output.parent.mkdir(parents=True, exist_ok=True)
        temporary_flat = args.flat_reference_output.with_suffix(
            args.flat_reference_output.suffix + ".tmp"
        )
        faiss.write_index(flat_reference, str(temporary_flat))
        temporary_flat.replace(args.flat_reference_output)
        flat_reference_sha256 = sha256_file(args.flat_reference_output)
        # Avoid holding an exact vector copy and the growing HNSW graph in the
        # Python heap together.  The exact index is reopened read-only and its
        # vectors are streamed into HNSW in bounded batches.
        del encoded_index
        del flat_reference
        gc.collect()
        exact = faiss.read_index(
            str(args.flat_reference_output), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
        )
        index = faiss.IndexHNSWFlat(dimension, args.m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = args.ef_construction
        for start in range(0, len(corpus), args.batch_size):
            count = min(args.batch_size, len(corpus) - start)
            index.add(np.asarray(exact.reconstruct_n(start, count), dtype="float32"))
        del exact
        gc.collect()
    temporary_index = args.index_output.with_suffix(args.index_output.suffix + ".tmp")
    faiss.write_index(index, str(temporary_index))
    temporary_index.replace(args.index_output)
    manifest = {
        "schema_version": 2,
        "index_revision": args.index_revision,
        "index_type": "hnsw64",
        "metric": "inner_product_on_normalized_e5_embeddings",
        "hnsw_m": args.m,
        "hnsw_ef_construction": args.ef_construction,
        "runtime_hnsw_ef_search": 256,
        "dimension": dimension,
        "passage_count": len(corpus),
        "passage_order": "wiki-18.jsonl row order",
        "arrow_fingerprint": str(corpus._fingerprint),
        "encoder": args.model,
        "encoder_revision": args.model_revision,
        "faiss_version": str(getattr(faiss, "__version__", "unknown")),
        "build_threads": args.threads,
        "build_mode": (
            "flat_mmap_then_hnsw" if args.flat_reference_output is not None else "direct_hnsw"
        ),
        "build_seconds": time.monotonic() - started,
        "corpus_sha256": sha256_file(args.corpus),
        "index_sha256": sha256_file(args.index_output),
        "flat_reference_sha256": flat_reference_sha256,
    }
    write_json(args.manifest_output, manifest)
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
