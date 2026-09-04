#!/usr/bin/env bash
set -euo pipefail
CORPUS_DIR="${1:-data/deepresearch/hotpot_corpus}"
INDEX_DIR="${2:-data/indexes/hotpotqa}"
THREADS="${THREADS:-22}"
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:--Xms2g -Xmx32g}"
mkdir -p "${INDEX_DIR}"
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input "${CORPUS_DIR}" \
  --index "${INDEX_DIR}" \
  --generator DefaultLuceneDocumentGenerator \
  --threads "${THREADS}" \
  --storePositions --storeDocvectors --storeRaw
