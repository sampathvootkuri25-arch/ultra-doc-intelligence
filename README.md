# Ultra Doc-Intelligence
The system accepts a single logistics document, indexes it for retrieval, answers grounded questions with evidence and confidence, and extracts a fixed shipment JSON schema.

## Scope

This repository is aligned to the problem statement in `Skill Test - AI Engineer.docx`:

- upload a logistics document in `PDF`, `DOCX`, or `TXT`
- parse and chunk the document
- create embeddings and store them in a vector index
- answer questions only from document context
- return answer, supporting source text, and confidence
- apply guardrails for missing or weak evidence
- extract structured shipment fields with `null` for missing data
- provide a lightweight reviewer UI and the required API endpoints

## Architecture

The implementation is intentionally compact:

- `app/main.py`
  - FastAPI entrypoint exposing `POST /upload`, `POST /ask`, and `POST /extract`
- `app/document_parsers.py`
  - document parsing, markdown normalization, block creation, and field extraction
- `app/pipeline.py`
  - orchestration for upload, chunking, embeddings, indexing, retrieval, grounded QA, confidence, and extraction
- `app/schemas.py`
  - API contracts and structured extraction schema
- `app/ui/streamlit_app.py`
  - minimal reviewer UI for upload, QA, source display, confidence display, and extraction

Runtime flow:

1. A document is uploaded through the API or Streamlit UI.
2. The parser reads page text and markdown-like structure.
3. The pipeline extracts high-value shipment fields and also builds broader text blocks.
4. Field chunks and text chunks are embedded with `BAAI/bge-m3`.
5. Chunks are stored in local Qdrant.
6. Questions are answered either by direct field grounding or vector retrieval plus grounded fallback synthesis.
7. Structured extraction returns the required shipment JSON with evidence and missing fields.

## Chunking Strategy

The chunking approach is designed for semi-structured logistics documents rather than generic prose alone.

- Field-first chunks
  - Each extracted field becomes its own chunk.
  - This improves precision for factual questions such as rate, shipper, consignee, pickup time, and shipment ID.
- Block-based text chunks
  - Parsed markdown is converted into paragraph blocks and table-row blocks.
  - Table rows are preserved as `header: value` pairs so retrieval works on logistics layouts.
- Token chunking for long text
  - Long blocks are split with `tiktoken` using:
    - `MAX_CHUNK_TOKENS = 500`
    - `CHUNK_OVERLAP_TOKENS = 60`

This hybrid chunking strategy is aligned to the skill test because it improves retrieval grounding on tables, labels, and short factual fields instead of treating the whole file as undifferentiated text.

## Retrieval Method

The retrieval stack is:

- embeddings: `BAAI/bge-m3`
- vector store: local Qdrant
- retrieval: cosine similarity over field and text chunks
- reranking: lightweight heuristic reranking based on term overlap, alias overlap, and a small bonus for field chunks

For non-LLM mode:

- if a question maps to an extracted field, the answer is returned directly from that field with evidence
- otherwise the system retrieves top chunks, reranks them, and selects the best grounded span from the evidence

For LLM-enabled mode:

- Azure OpenAI is optional
- the model receives only retrieved context
- the system prompt requires returning `Not found in document` when the answer is absent

## Guardrails Approach

The system includes explicit hallucination guardrails:

- answer only from document context
- return `Not found in document` when no retrieval results are available
- return `Not found in document` when retrieval confidence is below threshold
- for non-LLM fallback, only return spans that are directly present in the retrieved source text
- structured extraction uses `null` for missing fields instead of guessing

Returned QA statuses are:

- `answered`
- `not_found`
- `refused_low_confidence`

This aligns directly with the skill test requirement to include at least one hallucination guardrail.

## Confidence Scoring Method

Confidence is a transparent heuristic in `app/pipeline.py`.

Signals used:

- top retrieval score
- mean of the top retrieved scores
- lexical overlap between the question and the best source chunk
- bonus when the top result is a field chunk
- agreement across the top retrieved chunks

Current weighting:

- `0.30 * top_score`
- `0.20 * mean_top3`
- `0.25 * exact_support`
- `0.15 * field_bonus`
- `0.10 * agreement`

If confidence is below `LOW_CONFIDENCE_THRESHOLD = 0.55`, the system refuses with `Not found in document`.

## Structured Extraction

The extraction schema matches the problem statement:

- `shipment_id`
- `shipper`
- `consignee`
- `pickup_datetime`
- `delivery_datetime`
- `equipment_type`
- `mode`
- `rate`
- `currency`
- `weight`
- `carrier_name`

Extraction behavior:

- deterministic regex and labeled-field parsing first
- optional Azure OpenAI refinement second
- missing values remain `null`
- evidence snippets are returned for populated fields

## API

Required endpoints:

- `POST /upload`
- `POST /ask`
- `POST /extract`

Example local startup is documented in [RUN_STEPS.md](./RUN_STEPS.md).

## Minimal UI

The Streamlit reviewer UI supports:

- document upload
- question input
- answer display
- source snippet display
- confidence display
- structured extraction display

This is intentionally lightweight because the skill test grades usability rather than design polish.

## Tests

The test suite covers:

- end-to-end submission flow on the provided sample PDFs
- end-to-end upload, QA, extraction, and guardrail behavior on generated `DOCX` and `TXT` logistics files

Run:

```bash
uv run pytest tests/test_submission_flow.py
```

## Failure Cases

Known limitations:

- extraction is strongest on logistics documents with clear labels, tables, or standard rate confirmation structure
- uploaded parsed documents are currently kept in memory for the active process, so `ask` and `extract` depend on the process that handled the upload
- broader open-ended questions are improved but still less reliable than direct field questions
- scanned PDFs with weak OCR can reduce extraction and retrieval quality
- the system is single-document scoped and does not support cross-document reasoning

## Improvement Ideas

- persist parsed document metadata so questions continue to work after restart
- add OCR quality scoring and parser fallback telemetry
- add hybrid retrieval with sparse plus dense retrieval
- add a dedicated reranker model instead of heuristic reranking
- return source span offsets instead of snippet-only evidence
- add hosted deployment for the FastAPI API and Streamlit UI
- expand extraction coverage for invoices and shipment instructions with more diverse fixtures

## Local Run

Use the commands in [RUN_STEPS.md](./RUN_STEPS.md).

