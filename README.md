# CurriculumRAG

An educational document retrieval research project comparing keyword-based, dense semantic, and hybrid retrieval for Data Structures and Algorithms study questions.

## Research question

How do TF-IDF keyword retrieval, dense semantic retrieval, and Reciprocal Rank Fusion (RRF) compare when retrieving supporting educational passages for student questions?

## Dataset

- 40 original pilot passages covering Data Structures and Algorithms.
- 60 evaluation questions with passage relevance mappings.
- 15 development questions and 45 held-out test questions.
- Source files: `data/passages.csv`, `data/queries.csv`, and `data/relevance_audit.csv`.

## Split design

The development/test split is grouped by connected relevant-passage IDs. Questions whose relevance mappings share one or more passage IDs are kept in the same split. This prevents a labelled support passage from appearing in both development and test query groups.

Current split validation: 15 development questions, 45 test questions, and zero relevant-passage IDs shared across the two splits.

## Retrieval methods

1. TF-IDF with unigram and bigram features.
2. Dense semantic retrieval using `sentence-transformers/all-MiniLM-L6-v2`.
3. Hybrid retrieval using Reciprocal Rank Fusion (RRF), with rank constant 60.

## Annotation and review status

This is a small pilot benchmark, not a peer-reviewed or independently validated benchmark. AI assistance was used when preparing the initial passage/question material and provisional relevance mappings.

The project author has reviewed all 60 relevance mappings against the passage text: **all 60 are marked author-confirmed**. During review, passage P033 (Binary Search) was expanded to explain why sorted order allows binary search to eliminate one half of the remaining collection. The updated passage text is reflected consistently in the corpus and in audit entries Q038, Q039, and Q052.

Author review is not independent annotation. **Independent reviewer confirmation has not yet been completed**, and the audit keeps that status separate from author review.

## Results and evaluation boundary

The development results are rebuilt from the current passage corpus and grouped split. Only the 15 development questions are scored. The 45 test questions remain held out and are not scored by the rebuild workflow. This is a small pilot dataset, so metrics should be interpreted as preliminary rather than generalizable evidence.

## Technologies

- Python
- Google Colab
- Scikit-learn
- Sentence Transformers
- Hugging Face
- Gradio
