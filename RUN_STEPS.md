# Run Steps

## 1. Install `uv`

If `uv` is not already installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Create the environment and install dependencies

From the repository root:

```bash
uv sync
```

This repo is pinned to Python `3.11` through `.python-version`, so `uv` will create the right environment automatically.

## 3. Create a local `.env` file

The app works without hosted model credentials because it has deterministic field extraction and retrieval built in. The `.env` file is only needed when you want hosted answer generation and hosted extraction refinement.

Copy the example file:

```bash
cp .env.example .env
```

Then update `.env` with your values:

```env
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_DEPLOYMENT=your_deployment_name
```

## 4. Start the FastAPI backend

```bash
uv run uvicorn app.main:app --reload
```

The backend will start on:

```text
http://127.0.0.1:8000
```

## 5. Start the Streamlit UI

Open a second terminal in the same repository and run:

```bash
uv run streamlit run app/ui/streamlit_app.py
```

The UI will open in the browser, usually at:

```text
http://localhost:8501
```

## 6. Try the sample documents

Use any file from:

```text
input-documents/
```

Suggested questions:

- `What is the carrier rate?`
- `What is the agreed amount?`
- `Who is the consignee?`
- `What is the weight?`
- `When is pickup scheduled?`

Then run structured extraction from the UI.

## 7. Run the end-to-end test

```bash
uv run pytest tests/test_submission_flow.py
```

This test uploads the provided sample PDFs, asks the expected submission questions, and verifies extraction output.

## 8. Optional API checks

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Notes

- The first run may take time because Docling and the embedding model may need to download their assets.
- PDF parsing uses `Docling` first and falls back to `unstructured` if Docling does not produce usable output.
- If the `.env` values are not set, the app still starts and still answers the sample questions using deterministic extraction plus retrieval.
- Uploaded files are stored in `data/uploads/`.
- The local vector index is stored in `data/qdrant/`.
