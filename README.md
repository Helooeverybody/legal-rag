# RAG for Legal Case Reasoning and Criminal Judgment Prediction

A Retrieval-Augmented Generation (RAG) system, integrated with a **Reason-and-Act
(ReAct)** loop, that predicts criminal charges and penalty durations for Vietnamese
first-instance criminal cases under the Penal Code (*Bộ luật Hình sự* — BLHS).



---

## 1. What this project does

Given the facts of a criminal case, the system predicts, **per defendant**:

- **Tội danh** — the offence / charge,
- **Applied law clauses** — the exact statutory references (Điều / Khoản / Điểm) of
  the Penal Code,
- **Phạt tù** — the imprisonment term,
- and where applicable, fines (*Phạt tiền*), civil liability (*Trách nhiệm dân sự*),
  and handling of physical evidence (*Xử lý vật chứng*).

It does so by combining two kinds of retrieval:

1. **Exact statutory retrieval** — deterministic lookup of Penal Code articles by
   signature (e.g. `174-4-a` → Điều 174, Khoản 4, Điểm a).
2. **Analogous-case retrieval** — semantic search over a vector database of past
   judgments to calibrate sentencing using similar mitigating/aggravating factors.

A multi-step ReAct loop then synthesizes statutory law with precedent to produce a
structured, validated prediction.

### Pipeline at a glance

```
 Scanned court PDFs
        │  (Marker / Surya OCR)
        ▼
   Markdown text  ──► OCR grammar fix ──► section extraction ──► LLM template fill
        │                                                              │
        ▼                                                              ▼
                          Structured case JSON  ◄── synthetic first-person summaries
        │
        ├──────────────► train/test split
        │
        ▼
   Embedding (BGE-M3) ──► ChromaDB vector store
        │                         │
        │                         ├── similar past cases
        │                         └── sentencing-calibration cases
        ▼
   ReAct reasoning loop (Gemma / OpenAI / OpenRouter)
        │   step 1: extract facts + candidate offences
        │   step 2: retrieve exact law + supporting articles → legal analysis
        │   step 3: retrieve precedents → final per-defendant verdict
        ▼
   Structured prediction ──► evaluation (law-clause P/R/F1, sentence RMSE)
```

---

## 2. Repository layout

```
.
├── rag/                       # Core Python package (the installable library)
│   ├── config.py              # Centralized defaults (models, fields, paths)
│   ├── parse_penalty.py       # Parse Vietnamese sentence text → month ranges
│   ├── core/                  # Retrieval & legal-reference building blocks
│   │   ├── embeddings.py      # Chunk JSON, embed with SentenceTransformer, store/query ChromaDB
│   │   ├── law_embeddings.py  # Hierarchy-aware chunking/embedding of the Penal Code
│   │   ├── law_retriever.py   # Exact clause lookup by signature (Điều/Khoản/Điểm)
│   │   ├── sentencing.py      # Extract imprisonment months from verdict text
│   │   └── verdict_labels.py  # Extract law-clause labels from ground-truth verdicts
│   ├── llm/
│   │   └── providers.py       # Unified structured-output interface + provider fallback
│   ├── runtime/
│   │   └── retrieval.py       # Stateful runtime that caches model + collection
│   ├── generation/
│   │   ├── schemas.py         # Pydantic schemas for generation I/O
│   │   ├── reasoning_act.py   # The ReAct judgment-prediction pipeline
│   │   └── verdict_from_eval.py  # Generate verdicts from precomputed retrieval outputs
│   └── evaluation/
│       ├── eval_utils.py             # Shared scoring/aggregation helpers
│       ├── reasoning_act_eval.py     # Evaluate the ReAct pipeline
│       ├── reasoning_one_call_eval.py# Single-call baseline (no retrieval)
│       ├── reasoning_past_eval.py    # Precedent-only baseline
│       └── recalculate_saved_metrics.py  # Re-score a saved report without re-running the LLM
├── data_create/              # Dataset construction pipeline (PDF → structured JSON)
│   ├── scraper.ipynb         # Scrape judgments from the Supreme People's Court portal
│   ├── ocr_marker_surya.py   # OCR scanned PDFs → markdown (Marker / Surya)
│   ├── fix_grammar_ocr.py    # Correct OCR diacritic/spelling errors
│   ├── extract_sections.py   # Split judgment markdown into sections
│   ├── extract_sections_2.py # Rule-based field extraction
│   ├── fill_template_aistudio.py  # LLM-based structuring into the case schema
│   └── schemas.py            # Pydantic schemas for the data pipeline
├── scripts/                  # Operational / one-off command-line tools
│   ├── generate_synthetic_summary_aistudio.py  # Build synthetic first-person summaries
│   ├── splitter.py           # Train/test document-level split
│   ├── check_llm_provider_availability.py      # Verify API keys / connectivity
│   ├── remove_case_eval.py   # Drop a doc_id from a saved results file
│   └── one_time_remove_filtered_cases.py       # Bulk-remove quality-filtered cases
├── notebooks/                # Exploratory analysis & metric reports
├── tests/                    # Unit tests (run with `pytest` / `unittest`)
├── output/                   # Generated artifacts (vector DBs, eval results, figures)
├── pyproject.toml            # Project metadata, dependencies, console scripts
├── requirements.txt          # Pinned dependency list for `pip install`
└── law_RAG.pdf               # Full project report
```

> **Note on data:** case JSON files, the Penal Code JSON (`raw_law.json` /
> `law_doc.json`), the `chunk/` corpus, and evaluation result files are **not**
> version-controlled (see `.gitignore`). They are produced by the
> `data_create/` pipeline or supplied locally.

---

## 3. Installation

The project targets **Python ≥ 3.13**. Use a dedicated [conda](https://docs.conda.io/)
environment and install the dependencies with `pip`.

```bash
# 1. Create and activate an isolated environment
conda create -n rag-luat python=3.13 -y
conda activate rag-luat

# 2. Install the runtime dependencies
pip install -r requirements.txt

# 3. Install the project itself (registers the `rag-*` console commands)
pip install -e .

# 4. (Optional) verify the install
python -c "import rag; print('ok')"
```

> `pip install -e .` performs an editable install so the `rag-embed`,
> `rag-evaluate-reasoning-act`, … console scripts (declared in `pyproject.toml`)
> become available on your `PATH`. If you skip it, run the equivalents with
> `python -m`, e.g. `python -m rag.core.embeddings`.

Key dependencies: `chromadb`, `sentence-transformers` (BGE-M3 embeddings),
`google-genai`, `openai`, `pydantic`, `python-dotenv`, `scipy`.

A CUDA-capable GPU is recommended for embedding (`--device cuda`); pass
`--device cpu` to run on CPU.

---

## 4. Configuration

Copy the template and fill in the API keys for the provider(s) you intend to use:

```bash
cp .env.example .env
```

| Variable | Used by | Notes |
|---|---|---|
| `GOOGLE_API_KEY`, `GOOGLE_API_KEY_2`, `GOOGLE_API_KEY_3` | `aistudio` provider | Up to three keys; the client rotates on quota/error. |
| `OPENROUTER_API_KEY` | `openrouter` provider | Used for the free/paid fallback tiers. |
| `OPENAI_API_KEY` | `openai` provider | |
| `LLM_REQUEST_TIMEOUT` | AI Studio calls | Seconds; default `90`. |
| `HF_HOME` | embeddings | HuggingFace cache dir; defaults to `~/.cache/huggingface`. |

Provider defaults (see `rag/config.py`):

- `aistudio` → `gemma-4-31b-it`
- `openrouter` → `google/gemma-4-31b-it:free`
- `openai` → `gpt-5.4-nano`

Verify your setup before a long run:

```bash
python scripts/check_llm_provider_availability.py
```

---

## 5. Building the dataset (`data_create/`)

This stage turns raw court PDFs into structured case JSON. It is run once to build
the corpus and is the most environment-heavy part (OCR models, GPU).

1. **Scrape** judgments — `data_create/scraper.ipynb`.
2. **OCR** scanned PDFs → markdown — `data_create/ocr_marker_surya.py`
   (Marker pipeline on Surya OCR; designed for a 16 GB GPU).
3. **Fix OCR errors** in headers — `data_create/fix_grammar_ocr.py`.
4. **Extract sections** — `data_create/extract_sections.py`,
   then field extraction with `extract_sections_2.py`.
5. **Structure with an LLM** into the case schema —
   `data_create/fill_template_aistudio.py`.
6. **Synthetic summaries** (first-person case stories used as retrieval queries) —
   `scripts/generate_synthetic_summary_aistudio.py`.
7. **Split** into train/test at the document level — `scripts/splitter.py`.

---

## 6. Building the vector store

Chunk case JSON files and embed them into a persistent ChromaDB collection:

```bash
# Embed case summaries from a folder of JSON files
rag-embed run \
    --input_dir   chunk/train \
    --db_dir      output/chroma_db_train \
    --content_fields Summary \
    --model_name  BAAI/bge-m3 \
    --device      cuda \
    --max_chunk_chars 1500

# Query the collection
rag-embed query \
    --db_dir output/chroma_db_train \
    --text   "tình tiết giảm nhẹ trách nhiệm hình sự" \
    --top_k  5
```

Look up exact Penal Code clauses by signature:

```bash
rag-law-retrieve --law_doc raw_law.json --clauses 174-4-a 51-2 51
```

---

## 7. Running the reasoning pipelines

All generation/evaluation entry points read `.env` automatically and accept
`--provider {aistudio,openrouter,openai}`.

### 7.1 ReAct judgment prediction (main system)

Evaluates the full multi-step pipeline over `chunk/test`, embedding `chunk/train`
on first run. Results stream to `--results-out` and the run is **resumable**.

```bash
rag-evaluate-reasoning-act \
    --test-dir  chunk/test \
    --train-dir chunk/train \
    --law-json  raw_law.json \
    --provider  openrouter \
    --results-out output/reasoning_act_eval/results.json \
    --first-n 10            # optional: limit to the first N cases
```

Useful flags: `--skip-embedding` (reuse an existing case DB),
`--top-k-case` / `--broad-top-k-case` (precedent retrieval breadth),
`--max-additional-law-rounds` (extra legal-analysis rounds),
`--disable-provider-fallback`, `--include-non-blhs`.

### 7.2 Baselines

```bash
# Single LLM call, no retrieval (non-agentic baseline)
rag-evaluate-reasoning-one-call --test-dir chunk/test --provider openrouter

# Precedent-only baseline (similar past cases, no statutory ReAct loop)
rag-evaluate-reasoning-past --test-dir chunk/test --train-dir chunk/train
```

### 7.3 Verdict generation from precomputed retrieval

```bash
rag-generate-verdict \
    --test-dir chunk/test \
    --eval-results eval_results.json \
    --law-doc raw_law.json \
    --output-dir output/generated_verdict_from_eval \
    --provider aistudio
```

### 7.4 Re-score a saved report (no LLM calls)

```bash
rag-recalculate-metrics output/reasoning_act_eval/results.json
```

---

## 8. Evaluation & metrics

Predictions are compared against the ground-truth verdict fields. Reports are saved
as JSON containing the run `config`, an aggregate `summary`, and `per_doc` details.
The headline metrics (see `rag/evaluation/eval_utils.py`) are:

- **Law-clause set Precision / Recall / F1 (macro)** — per-defendant set agreement
  between predicted and ground-truth Penal Code articles (Điều level).
- **Phạt tù RMSE (months)** — root-mean-squared error of the predicted imprisonment
  term, parsed from Vietnamese sentence text into months
  (`rag/core/sentencing.py`, `rag/parse_penalty.py`).

Only BLHS (*Bộ luật Hình sự*) clauses are scored by default; pass `--include-non-blhs`
to score all cited legal sources.

---

## 9. Testing

```bash
python -m pytest tests/ -q
# or, without pytest:
python -m unittest discover -s tests -v
```

The suite covers sentence parsing (`test_sentencing.py`), the ReAct helpers
(`test_reasoning_act.py`), and the one-call baseline (`test_reasoning_one_call_eval.py`).

---

## 10. LLM provider abstraction

`rag/llm/providers.py` exposes a single `generate_structured_output(...)` function
that validates responses against a Pydantic model, plus
`generate_structured_output_with_fallback(...)` which tries providers/models in
order (e.g. OpenRouter free → AI Studio → OpenRouter paid) for fault tolerance.
Switching providers requires only the `--provider` flag — no code changes.

---

## 11. License

See [`LICENSE`](LICENSE).
