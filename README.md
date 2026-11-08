# CurriculumRAG

An educational document retrieval research project comparing keyword-based, dense semantic, and hybrid retrieval for Data Structures and Algorithms study questions.

## Research question

How do TF-IDF keyword retrieval, dense semantic retrieval, and Reciprocal Rank Fusion (RRF) compare when retrieving supporting educational passages for student questions?

## Dataset

- 40 original pilot passages covering Data Structures and Algorithms.
- 60 evaluation questions with passage relevance labels.
- 15 development questions and 45 held-out test questions.
- Source files: `data/passages.csv`, `data/queries.csv`, and `data/relevance_audit.csv`.

## Split design

The development/test split is grouped by connected relevant-passage IDs. Questions whose relevance labels share one or more passage IDs are kept in the same split. This prevents the same labelled support passage from appearing in both development and test query groups.

Current split validation: 15 development questions, 45 test questions, and zero relevant-passage IDs shared across the two splits.

## Retrieval methods

1. TF-IDF with unigram and bigram features.
2. Dense semantic retrieval using `sentence-transformers/all-MiniLM-L6-v2`.
3. Hybrid retrieval using Reciprocal Rank Fusion (RRF), with rank constant 60.

## Research status and limitations

This is a small pilot benchmark, not a peer-reviewed or validated benchmark. Passage/question content and relevance labels were AI-assisted and reviewed against the current passage text. One relevance mapping (Q056) was corrected to P018 because P018 directly describes adjacency-matrix rows, columns, and entries.

**Independent human confirmation of all 60 relevance labels is still pending.** Until that is completed, evaluation results are preliminary and should not be presented as validated findings.

The CSV files currently under `results/` were generated using the earlier question-level split. They are retained for traceability but must be regenerated from the current passage-grouped split before interpreting or reporting development results.

## Technologies

- Python
- Google Colab
- Scikit-learn
- Sentence Transformers
- Hugging Face
- Gradio
