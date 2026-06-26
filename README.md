# rag-implicit-mis

This repository contains data, code and results for our project.  
We propose a Retrieval Augmented Generation (RAG) framework for detecting and explaining implicit misogyny in Italian and English, leveraging three complementary knowledge bases Wikipedia, a slang lexicon and feminist literature. Human evaluation shows that RAG-based explanations are preferred over no-RAG explanations in both languages.

## Repo Structure

```text
├── data/
│   ├── implicITMis.xlsx                  # Italian dataset (ImplicIT-Mis)
│   ├── sbic_cleaned.csv                  # English dataset (SBIC+)
│   ├── full_results_ita.xlsx             # Italian results with retrieval, outputs and annotations
│   ├── full_results_eng.xlsx             # English results with retrieval, outputs and annotations
│   ├── annotation_ita_by_kb_final.xlsx   # Final Italian annotation file (faithfulness + human preference)
│   └── annotation_eng_by_kb_final.xlsx   # Final English annotation file (faithfulness + human preference)
├── code/
│   ├── generation/
│   │   ├── run.py                        # LLM inference (few-shot and zero-shot, RAG and no-RAG)
│   │   ├── job_fewshot.sh                # SLURM job script for few-shot inference
│   │   └── job_zeroshot.sh               # SLURM job script for zero-shot inference
│   ├── slang_pipeline.py                 # Slang KB construction and retrieval via Wiktionary
│   ├── compression.py                    # Semantic compression of retrieved passages
│   ├── merge.py                          # Merge retrieval outputs into unified datasets
│   ├── parser_rag.py                     # Parse LLM JSON outputs, extract labels and assumptions
│   ├── build_results.py                  # Build full results files with annotations
│   └── annotation_analysis.py            # Annotation statistics and inter-annotator agreement
└── README.md
```

## Data

### Datasets

**implicITMis.xlsx**: Italian dataset of Facebook comments with associated news headlines (1,120 instances)  
**sbic_cleaned.csv**: English dataset of social media posts (2,344 instances)

### Result files

full_results_ita.xlsx and full_results_eng.xlsx contain one row per instance with the following columns:

| Column | Description |
|---|---|
| id | Instance identifier |
| text | Input text |
| Contesto | News headline context (Italian only) |
| baseline_assumption | Implied assumption generated without retrieval |
| baseline_label | Misogyny label without retrieval (YES/NO or SÌ/NO) |
| baseline_raw | Full raw JSON output of the baseline model |
| wiki_retrieval | Retrieved Wikipedia passages |
| wiki_assumption | Implied assumption generated with Wikipedia retrieval |
| wiki_label | Misogyny label with Wikipedia retrieval |
| wiki_raw | Full raw JSON output of the Wikipedia RAG model |
| slang_retrieval | Retrieved slang definitions |
| slang_assumption | Implied assumption generated with slang retrieval |
| slang_label | Misogyny label with slang retrieval |
| slang_raw | Full raw JSON output of the slang RAG model |
| lit_retrieval | Retrieved feminist literature passages |
| lit_assumption | Implied assumption generated with literature retrieval |
| lit_label | Misogyny label with literature retrieval |
| lit_raw | Full raw JSON output of the literature RAG model |

### Annotation files

annotation_ita_by_kb_final.xlsx and annotation_eng_by_kb_final.xlsx contain the human evaluation for the 100 annotated instances per language:

| Column | Description |
|---|---|
| *_faithfulness_annotator1 | Faithfulness score by annotator1 (0=not grounded, 1=partial, 2=fully grounded) |
| *_faithfulness_annotator2 | Faithfulness score by annotator2 |
| human_preference_annotator1 | Preferred explanation according to annotator1 |
| human_preference_annotator2 | Preferred explanation according to annotator2 |

Human preference values: WIKI, SLANG, LITERATURE, BASELINE, NEITHER, TIE, TIE_X_Y (e.g. TIE_LITERATURE_WIKI).

The 100 annotated instances per language were sampled to maximise retrieval coverage (instances with more knowledge sources available were prioritised).

## Knowledge bases

### Wikipedia

Domain-filtered index built from Wikidata, retaining only human entities and pages related to feminism, femicide, and violence against women. Complemented at retrieval time by the Wikipedia Search API. Retrieved passages are compressed via semantic similarity (threshold = 0.3).

| Language | Coverage |
|---|---|
| Italian | 979 / 1,120 (87.4%) |
| English | 79 / 2,344 (3.4%) |

### Slang

Built from Wiktionary. For Italian: ~530 entries from the Italian_slang category (English Wiktionary). For English: ~22,000 entries combining English_slang, English_internet_slang, and English_incel_slang. Definitions tagged as vulgar, derogatory, or offensive are prioritised. Matching is performed via token-level lookup followed by semantic reranking (cosine similarity ≥ 0.3).

| Language | Coverage |
|---|---|
| Italian | 109 / 1,120 (9.7%) |
| English | 1,339 / 2,344 (57.1%) |

### feminist Literature

Built from two sources: Down Girl: The Logic of Misogyny (Manne, 2018) and Gender Hate Online: Understanding the New Anti-Feminism (Ging, 2019). Retrieval is based on semantic similarity between the input text and two-sentence units from the books (cosine similarity ≥ 0.6 for retrieval, ≥ 0.3 for sentence-level filtering).

| Language | Coverage |
|---|---|
| Italian | 331 / 1,120 (29.6%) |
| English | 956 / 2,344 (40.8%) |

## Annotation

Two annotators independently rated each of the 100 instances per language along two dimensions.

### Faithfulness (scale 0–2)

Measures how well the generated implied assumption is grounded in the input text and retrieved evidence.

| Score | Description |
|---|---|
| 2 — Fully grounded | The explanation accurately identifies the implicit misogyny and is clearly supported by the input and/or retrieved context |
| 1 — Partially grounded | The explanation captures part of the meaning but is incomplete or only weakly connected to the evidence |
| 0 — Not grounded | The explanation is unsupported, contradicts the text, or invents content |

### Human Preference

Which explanation the annotator considers best overall, considering correctness, clarity, and usefulness.

### Inter-Annotator Agreement

| Dimension | Italian κ | English κ |
|---|---:|---:|
| Faithfulness (aggregate) | 0.528 | 0.520 |
| Human Preference | 0.842 | 0.838 |

## Model

All experiments use meta-llama/Llama-3.1-8B-Instruct with temperature 0.3 and bfloat16 precision.

## Usage

### Retrieval

Slang pipeline (builds the Wiktionary KB and runs retrieval):

```bash
python code/slang_pipeline.py \
  --input data.csv \
  --output output.csv \
  --lang it \
  --cache slang_cache_it.json
```

Compression (compresses Wikipedia retrieved passages):

```bash
python code/compression.py \
  --input data.csv \
  --output output.csv \
  --retrieval-col retrieval_text_RAG \
  --threshold 0.3
```

### Inference

```bash
python code/generation/run.py \
  --input-csv data.csv \
  --out-csv results.csv \
  --mode fewshot \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --dtype bfloat16
```

To resume an interrupted job from row 500:

```bash
python code/generation/run.py \
  --input-csv data.csv \
  --out-csv results.csv \
  --mode fewshot \
  --batch-start 500
```

Via SLURM:

```bash
sbatch code/generation/job_fewshot.sh
sbatch code/generation/job_zeroshot.sh
```

## Analysis

Merge retrieval outputs:

```bash
python code/merge.py --lang it --data-dir ita/ --output merged_ita.csv --sample
```

Parse LLM outputs:

```bash
python code/parser_rag.py process \
  --input-dir results/ \
  --output-dir processed/ \
  --lang it
```

Build full results file:

```bash
python code/build_results.py \
  --lang it \
  --data-dir ita/ \
  --ann data/annotation_ita_by_kb_final.xlsx \
  --output data/full_results_ita.xlsx
```

Annotation statistics:

```bash
python code/annotation_analysis.py \
  --ann-ann1 annotation_ann1.xlsx \
  --ann-ann2 annotation_ann2.xlsx \
  --output-disagreements disagreements.csv \
  --output-stats stats.csv
```
