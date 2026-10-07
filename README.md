# Codexa API

Find Bible verses from a plain-English description, and ask questions that are answered only from the
passages found, with citations.

```
GET /search?q=a woman looks back at a burning city and turns into a pillar of salt
→ Genesis 19:26  "But his wife looked back from behind him, and she became a pillar of salt."
```

FastAPI, NumPy, onnxruntime and Gemini; evaluated and traced with Langfuse; deployed to Cloud Run with
Terraform and GitHub Actions.

**Live:** https://codexa-api-p7wdcrydma-ew.a.run.app (API reference at [/docs](https://codexa-api-p7wdcrydma-ew.a.run.app/docs))

## Endpoints

| | |
|---|---|
| `GET /search` | Search 19 works (English translations, the Greek and Hebrew texts, commentaries, reference works) by keyword, by meaning, or both |
| `GET /ask` | An answer written by Gemini from the passages found, citing them as [1], [2], … (needs an `X-API-Key`) |
| `GET /health` | What's loaded |

## How it works

- **Packs.** Each work is one SQLite file: its texts, an FTS5 keyword index, and every text's
  embedding stored as Int8.
- **Embeddings.** Queries are embedded locally with EmbeddingGemma-300m (int8 ONNX, onnxruntime),
  truncated to 256 dimensions.
- **Search.** Semantic search is one matrix multiply per pack. Hybrid search merges the keyword and
  vector rankings with Reciprocal Rank Fusion, since BM25 scores from separate indexes can't be compared.
- **Answers.** The model sees only the retrieved passages, numbered. The question and passages are
  fenced as data so instructions hidden in them are ignored, and every [n] in the answer is checked
  against the passages.

| File | |
|---|---|
| `codexa_api/search.py` | Keyword, semantic and hybrid search |
| `codexa_api/assistant.py` | Prompt, Gemini call, citation check, Langfuse trace |
| `codexa_api/main.py` | Routes and start-up |
| `evals/retrieval_experiment.py` | The Langfuse experiment below |

## Evaluation

The test set is 59 questions that describe a verse without quoting it, each with the verse it should
find, for example *"the verse where kids make fun of a bald man and get mauled by bears"* → 2 Kings 2:23.

`evals/retrieval_experiment.py` loads them into Langfuse as a dataset, runs each search mode over it
as an experiment, and scores every question: is a correct verse in the top 1, 5 or 10 results, how
high (reciprocal rank), and how fast.

| Search | Top 1 | Top 5 | Top 10 | MRR |
|---|---|---|---|---|
| Keyword | 58% | 76% | 80% | 0.661 |
| Semantic | 75% | 95% | 95% | 0.834 |
| **Hybrid** (default) | **80%** | **92%** | **95%** | **0.854** |

Measured on the same questions:

- **Model choice.** EmbeddingGemma-300m found the right verse in the top 5 for 92% of questions,
  against 81% for bge-small-en-v1.5 and 63% for multilingual-e5-small. Gemini's embedding model scored
  100% but needs a network call per query (about 0.5 s against 0.1 s) and can't run offline.
- **Int8 storage.** Int8 vectors ranked as well as Float32 (95% top 5 either way) at a quarter of the size.

Each `/ask` is also traced in Langfuse: the retrieval, the Gemini call with its token counts, and a
`citations_valid` score.

## Run locally

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # GEMINI_API_KEY for /ask; Langfuse keys for tracing
uvicorn codexa_api.main:app --port 8790
```

It needs the packs in `data/` and the embedding model in `models/embeddinggemma-300m-ONNX`:

```bash
hf download onnx-community/embeddinggemma-300m-ONNX tokenizer.json \
  onnx/model_quantized.onnx onnx/model_quantized.onnx_data \
  --local-dir models/embeddinggemma-300m-ONNX
```

## Tests

```bash
pytest
```

Unit and API tests build two tiny packs on the fly and use a fake embedder and LLM, so they need no
model, network or keys. Real-data tests run when the packs and model are present.

## Deploy

Cloud Run in `europe-west1`, scaled to zero (nothing runs or is billed between requests), at most one
instance.

- **`terraform/`** creates the service, an image registry that keeps the two newest images, a private
  bucket for the packs and model, Secret Manager entries for the keys, keyless deploys from GitHub
  Actions (Workload Identity Federation), and a £5 monthly budget alert.
- **Every push** runs ruff and pytest. On `main`, it then builds the image and deploys it.

## Licence

MIT
