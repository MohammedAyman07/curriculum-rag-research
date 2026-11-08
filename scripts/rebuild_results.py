from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PASSAGES_PATH = DATA_DIR / "passages.csv"
QUERIES_PATH = DATA_DIR / "queries.csv"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RRF_CONSTANT = 60
TOP_K = 5
LATENCY_REPEATS = 5


def parse_relevant_ids(value: str) -> list[str]:
    ids = [item.strip() for item in str(value).split("|") if item.strip()]
    if not ids:
        raise ValueError("Every query must have at least one relevant passage ID.")
    return ids


def validate_data(passages: pd.DataFrame, queries: pd.DataFrame) -> None:
    expected_passage_columns = {"doc_id", "topic", "text", "source"}
    expected_query_columns = {
        "query_id", "question", "relevant_doc_ids", "split"
    }
    if not expected_passage_columns.issubset(passages.columns):
        raise ValueError("passages.csv is missing required columns.")
    if not expected_query_columns.issubset(queries.columns):
        raise ValueError("queries.csv is missing required columns.")
    if len(passages) != 40 or len(queries) != 60:
        raise ValueError(
            f"Unexpected dataset dimensions: {len(passages)} passages, "
            f"{len(queries)} queries."
        )
    if not passages["doc_id"].is_unique or not queries["query_id"].is_unique:
        raise ValueError("Passage IDs and query IDs must be unique.")
    if set(queries["split"]) != {"dev", "test"}:
        raise ValueError("Query split must contain only dev and test.")
    dev = queries[queries["split"] == "dev"]
    test = queries[queries["split"] == "test"]
    if len(dev) != 15 or len(test) != 45:
        raise ValueError(
            f"Expected 15 dev and 45 test queries; found {len(dev)} and {len(test)}."
        )

    available_ids = set(passages["doc_id"].astype(str))
    relevant_by_split: dict[str, set[str]] = {"dev": set(), "test": set()}
    for row in queries.itertuples(index=False):
        relevant_ids = parse_relevant_ids(row.relevant_doc_ids)
        unknown = set(relevant_ids) - available_ids
        if unknown:
            raise ValueError(f"{row.query_id} references unknown IDs: {sorted(unknown)}")
        relevant_by_split[row.split].update(relevant_ids)

    overlap = relevant_by_split["dev"] & relevant_by_split["test"]
    if overlap:
        raise ValueError(
            "Data leakage detected: relevant passage IDs appear in both splits: "
            + ", ".join(sorted(overlap))
        )


def rank_from_scores(scores: np.ndarray, document_ids: list[str]) -> list[str]:
    indices = np.argsort(-scores, kind="stable")
    return [document_ids[int(i)] for i in indices]


def metric_values(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = TOP_K,
) -> tuple[float, float, float, int]:
    relevant = set(relevant_ids)
    top = retrieved_ids[:k]
    hits = set(top) & relevant
    precision = len(hits) / k
    recall = len(hits) / len(relevant)
    first_rank = next(
        (rank for rank, doc_id in enumerate(top, start=1) if doc_id in relevant),
        None,
    )
    reciprocal_rank = 0.0 if first_rank is None else 1.0 / first_rank
    hit_at_1 = int(bool(top) and top[0] in relevant)
    return precision, recall, reciprocal_rank, hit_at_1


def median_latency_ms(callable_fn, question: str) -> float:
    samples = []
    for _ in range(LATENCY_REPEATS):
        start = time.perf_counter()
        callable_fn(question)
        samples.append((time.perf_counter() - start) * 1000)
    return float(statistics.median(samples))


def main(split: str = "dev") -> None:
    passages = pd.read_csv(PASSAGES_PATH).fillna("")
    queries = pd.read_csv(QUERIES_PATH).fillna("")
    validate_data(passages, queries)

    documents = passages["text"].astype(str).tolist()
    document_ids = passages["doc_id"].astype(str).tolist()
    id_to_position = {doc_id: i for i, doc_id in enumerate(document_ids)}
    evaluation_queries = queries[queries["split"] == split].copy()
    expected_count = 15 if split == "dev" else 45
    if len(evaluation_queries) != expected_count:
        raise ValueError(f"Expected {expected_count} {split} queries; found {len(evaluation_queries)}.")

    # TF-IDF baseline: fit only on the fixed passage corpus.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    tfidf_index_start = time.perf_counter()
    tfidf_matrix = vectorizer.fit_transform(documents)
    tfidf_index_seconds = time.perf_counter() - tfidf_index_start

    def rank_tfidf(question: str) -> list[str]:
        query_vector = vectorizer.transform([question])
        scores = (query_vector @ tfidf_matrix.T).toarray().ravel()
        return rank_from_scores(scores, document_ids)

    # Dense model and passage embeddings.
    model_load_start = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model_load_seconds = time.perf_counter() - model_load_start

    document_embedding_start = time.perf_counter()
    if hasattr(model, "encode_document"):
        document_embeddings = model.encode_document(
            documents,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    else:
        document_embeddings = model.encode(
            documents,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    document_embeddings = np.asarray(document_embeddings, dtype=np.float32)
    document_embedding_seconds = time.perf_counter() - document_embedding_start

    if document_embeddings.shape[0] != len(documents):
        raise ValueError("Dense embedding count does not match the corpus.")

    def encode_query(question: str) -> np.ndarray:
        if hasattr(model, "encode_query"):
            vector = model.encode_query(
                [question],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
        else:
            vector = model.encode(
                [question],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
        return np.asarray(vector, dtype=np.float32)

    def rank_dense(question: str) -> list[str]:
        query_embedding = encode_query(question)
        scores = document_embeddings @ query_embedding
        return rank_from_scores(scores, document_ids)

    def rank_rrf(question: str) -> list[str]:
        rankings = [rank_tfidf(question), rank_dense(question)]
        fused_scores = {doc_id: 0.0 for doc_id in document_ids}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking, start=1):
                fused_scores[doc_id] += 1.0 / (RRF_CONSTANT + rank)
        # Stable tie-break by the original corpus order.
        return sorted(
            document_ids,
            key=lambda doc_id: (
                -fused_scores[doc_id],
                id_to_position[doc_id],
            ),
        )

    # Warm up model query encoding before collecting timing samples.
    rank_dense("Warm-up question for dense retrieval.")
    rank_rrf("Warm-up question for hybrid retrieval.")

    method_specs = [
        ("TF-IDF", rank_tfidf),
        ("Dense", rank_dense),
        ("Hybrid RRF", rank_rrf),
    ]
    prediction_frames: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []

    for method_name, ranker in method_specs:
        records = []
        for row in evaluation_queries.itertuples(index=False):
            question = str(row.question)
            relevant_ids = parse_relevant_ids(row.relevant_doc_ids)
            ranked_ids = ranker(question)
            precision, recall, mrr, _ = metric_values(
                ranked_ids, relevant_ids, TOP_K
            )
            latency = median_latency_ms(ranker, question)
            records.append(
                {
                    "query_id": row.query_id,
                    "question": question,
                    "relevant_doc_ids": row.relevant_doc_ids,
                    "retrieved_doc_ids": "|".join(ranked_ids[:TOP_K]),
                    "precision_at_5": precision,
                    "recall_at_5": recall,
                    "mrr": mrr,
                    "median_query_latency_ms": latency,
                }
            )

        frame = pd.DataFrame(records)
        prediction_frames[method_name] = frame
        key = (
            "tfidf"
            if method_name == "TF-IDF"
            else "dense"
            if method_name == "Dense"
            else "hybrid"
        )
        frame.to_csv(
            RESULTS_DIR / f"{key}_{split}_predictions.csv",
            index=False,
        )
        summary_rows.append(
            {
                "Method": method_name,
                "Precision@5": frame["precision_at_5"].mean(),
                "Recall@5": frame["recall_at_5"].mean(),
                "MRR": frame["mrr"].mean(),
                "Hit@1": float(
                    np.mean(
                        [
                            metric_values(
                                str(r.retrieved_doc_ids).split("|"),
                                parse_relevant_ids(r.relevant_doc_ids),
                            )[3]
                            for r in frame.itertuples(index=False)
                        ]
                    )
                ),
                "Median Query Latency (ms)": frame[
                    "median_query_latency_ms"
                ].median(),
                "Index Creation Time (seconds)": (
                    tfidf_index_seconds if key == "tfidf" else np.nan
                ),
                "Model": MODEL_NAME if key != "tfidf" else "",
                "Device": "cpu" if key != "tfidf" else "",
                "Model Loading Time (seconds)": (
                    model_load_seconds if key != "tfidf" else np.nan
                ),
                "Document Embedding Time (seconds)": (
                    document_embedding_seconds if key != "tfidf" else np.nan
                ),
                "RRF Constant": RRF_CONSTANT if key == "hybrid" else np.nan,
            }
        )

    # Preserve per-method summary files for backwards compatibility.
    metric_columns = [
        "Method",
        "Precision@5",
        "Recall@5",
        "MRR",
        "Median Query Latency (ms)",
        "Index Creation Time (seconds)",
        "Model",
        "Device",
        "Model Loading Time (seconds)",
        "Document Embedding Time (seconds)",
        "RRF Constant",
    ]
    summaries = pd.DataFrame(summary_rows)
    for key, method_name in [
        ("tfidf", "TF-IDF"),
        ("dense", "Dense"),
        ("hybrid", "Hybrid RRF"),
    ]:
        row = summaries[summaries["Method"] == method_name]
        row.to_csv(RESULTS_DIR / f"{key}_{split}_metrics.csv", index=False)

    # Shared comparison table.
    summaries[
        [
            "Method",
            "Precision@5",
            "Recall@5",
            "MRR",
            "Hit@1",
            "Median Query Latency (ms)",
        ]
    ].to_csv(RESULTS_DIR / f"{split}_method_comparison.csv", index=False)

    # Per-question ranking errors (not ranked first).
    error_records = []
    for method_name, frame in prediction_frames.items():
        for row in frame.itertuples(index=False):
            top_ids = str(row.retrieved_doc_ids).split("|")
            relevant = set(parse_relevant_ids(row.relevant_doc_ids))
            first_rank = next(
                (
                    rank
                    for rank, doc_id in enumerate(top_ids, start=1)
                    if doc_id in relevant
                ),
                None,
            )
            if first_rank != 1:
                error_records.append(
                    {
                        "Method": method_name,
                        "Query ID": row.query_id,
                        "Question": row.question,
                        "Relevant Passage": row.relevant_doc_ids,
                        "Retrieved Passages": row.retrieved_doc_ids,
                        "Correct Passage Rank": first_rank,
                    }
                )

    pd.DataFrame(error_records).to_csv(
        RESULTS_DIR / f"{split}_query_error_analysis.csv",
        index=False,
    )

    metadata = {
        "model": MODEL_NAME,
        "device": "cpu",
        "rrf_constant": RRF_CONSTANT,
        "top_k": TOP_K,
        "latency_repeats_per_query": LATENCY_REPEATS,
        "passage_count": len(passages),
        "query_count": len(queries),
        "development_query_count": int((queries["split"] == "dev").sum()),
        "evaluated_split": split,
        "evaluated_query_count": len(evaluation_queries),
        "test_query_count": int((queries["split"] == "test").sum()),
        "test_queries_scored": split == "test",
        "shared_relevant_passage_ids_across_splits": 0,
        "author_passage_content_review": "completed_all_40",
        "author_relevance_review": "completed_all_60",
        "independent_reviewer_confirmation": "pending",
        "results_status": "preliminary_author_reviewed_labels",
    }
    (RESULTS_DIR / f"{split}_run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"{split.title()} experiment evaluation completed.")
    print(f"Passages: {len(passages)}")
    if split == "dev":
        print(f"Development queries scored: {len(evaluation_queries)}")
        print("Held-out test queries scored: 0")
    else:
        print("Development queries scored: 0")
        print(f"Held-out test queries scored: {len(evaluation_queries)}")
    print(summaries[
        ["Method", "Precision@5", "Recall@5", "MRR", "Hit@1",
         "Median Query Latency (ms)"]
    ].to_string(index=False))
    print(f"Results written to: {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval on one fixed data split.")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    args = parser.parse_args()
    main(args.split)
