# CurriculumRAG

An educational document retrieval research project comparing keyword-based, dense semantic, and hybrid retrieval for Data Structures and Algorithms study questions.

## Research question

How do TF-IDF keyword retrieval, dense semantic retrieval, and Reciprocal Rank Fusion (RRF) compare when retrieving supporting educational passages for student questions?

## Dataset

- 40 Data Structures and Algorithms pilot passages, reviewed and verified by the project author.
- 60 evaluation questions with author-reviewed passage relevance mappings.
- 15 development questions and 45 held-out test questions.
- Source files: `data/passages.csv`, `data/queries.csv`, and `data/relevance_audit.csv`.

## Split design

The development/test split is grouped by connected relevant-passage IDs. Questions whose relevance mappings share one or more passage IDs are kept in the same split. This prevents a labelled support passage from appearing in both development and test query groups.

Current split validation: 15 development questions, 45 test questions, and zero relevant-passage IDs shared across the two splits.

## Retrieval methods

1. TF-IDF with unigram and bigram features.
2. Dense semantic retrieval using `sentence-transformers/all-MiniLM-L6-v2`.
3. Hybrid retrieval using Reciprocal Rank Fusion (RRF), with rank constant 60.

## Data review and provenance

AI assistance was used when preparing the initial passage/question material and provisional relevance mappings. The project author has reviewed all 40 passage texts and marked them as revised and verified in the `source` field of `data/passages.csv`. The project author has also reviewed all 60 relevance mappings against the passage text; all 60 are marked author-confirmed in the audit.

These are author reviews, not independent annotations. **Independent reviewer confirmation has not yet been completed.** The benchmark is a small pilot dataset and is not peer-reviewed or independently validated.

## Evaluation protocol

- Development evaluation uses only the 15 development questions.
- Final held-out evaluation uses the 45 test questions separately. In GitHub Actions, open **Actions → Run retrieval evaluation → Run workflow**, select `test`, and start the workflow.
- The script writes split-specific predictions, metrics, comparison, error analysis, and run metadata. Test outputs use the `test_*` filenames and do not overwrite development outputs.
- Do not use test metrics to tune retrieval settings. If you change the corpus, labels, model, or retrieval configuration after seeing test results, treat the 45 questions as no longer an untouched test set and create a new holdout set for another final evaluation.
- Results should be interpreted as preliminary given the small corpus and pending independent review.

## Technologies

- Python
- Google Colab
- Scikit-learn
- Sentence Transformers
- Hugging Face
- Gradio
