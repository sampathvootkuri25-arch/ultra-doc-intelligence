from fastapi import FastAPI, File, HTTPException, UploadFile

from app.pipeline import get_pipeline
from app.schemas import AskRequest, AskResponse, ExtractRequest, ExtractResponse, UploadResponse


app = FastAPI(title="Ultra Doc-Intelligence", version="0.1.0")
pipeline = get_pipeline()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    if file.filename.rsplit(".", 1)[-1].lower() not in {"pdf", "docx", "txt"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    return pipeline.upload_document(file.filename, await file.read())


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        return pipeline.ask(request.document_id, request.question)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/extract", response_model=ExtractResponse)
def extract(request: ExtractRequest) -> ExtractResponse:
    try:
        return pipeline.extract(request.document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
