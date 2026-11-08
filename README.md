# CurriculumRAG

An educational document retrieval research project comparing keyword-based, dense semantic, and hybrid retrieval for Data Structures and Algorithms study questions.

## Research question

How do TF-IDF keyword retrieval, dense semantic retrieval, and Reciprocal Rank Fusion (RRF) compare when retrieving supporting educational passages for student questions?

## Dataset

- 40 original pilot passages covering Data Structures and Algorithms.
- 60 evaluation questions with provisional passage relevance mappings.
- 15 development questions and 45 held-out test questions.
- Source files: `data/passages.csv`, `data/queries.csv`, and `data/relevance_audit.csv`.

## Split design

The development/test split is grouped by connected relevant-passage IDs. Questions whose relevance mappings share one or more passage IDs are kept in the same split. This prevents a labelled support passage from appearing in both development and test query groups.

Current split validation: 15 development questions, 45 test questions, and zero relevant-passage IDs shared across the two splits.

## Retrieval methods

1. TF-IDF with unigram and bigram features.
2. Dense semantic retrieval using `sentence-transformers/all-MiniLM-L6-v2`.
3. Hybrid retrieval using Reciprocal Rank Fusion (RRF), with rank constant 60.

## Annotation and research status

This is a small pilot benchmark, not a peer-reviewed or independently validated benchmark. AI assistance was used when preparing the initial passage/question material and provisional relevance mappings. The mappings received an AI-assisted consistency check against the passage text; this is not a substitute for independent human annotation.

The audit file records concise mapping statuses. Q056 is marked `CORRECTED_MATCH` and maps only to P018, which directly describes adjacency-matrix rows, columns, and entries. Q038 is marked `WORDING_CHECK` because P033 describes binary search on sorted collections but could explain more explicitly why the ordering is required.

**Independent human confirmation of all 60 relevance mappings is pending.** Until a person checks and confirms the mappings, development metrics are preliminary and must not be described as human-validated findings.

## Results and evaluation boundary

The current development result files under `results/` were regenerated using the passage-grouped split. Only the 15 development questions are scored. The 45 test questions remain held out and have not been scored. Because the development set is small and the relevance mappings are awaiting human confirmation, the metrics should be treated as preliminary rather than generalizable evidence.

## Technologies

- Python
- Google Colab
- Scikit-learn
- Sentence Transformers
- Hugging Face
- Gradio
