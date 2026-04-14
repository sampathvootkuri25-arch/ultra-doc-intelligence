from typing import Literal

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    doc_type: str
    pages: int
    chunk_count: int
    status: str


class AskRequest(BaseModel):
    document_id: str
    question: str = Field(min_length=3)


class SourceSnippet(BaseModel):
    page_number: int
    chunk_id: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    confidence: float
    status: Literal["answered", "not_found", "refused_low_confidence"]
    sources: list[SourceSnippet]


class ShipmentExtraction(BaseModel):
    shipment_id: str | None = None
    shipper: str | None = None
    consignee: str | None = None
    pickup_datetime: str | None = None
    delivery_datetime: str | None = None
    equipment_type: str | None = None
    mode: str | None = None
    rate: float | None = None
    currency: str | None = None
    weight: str | None = None
    carrier_name: str | None = None


class ExtractionSource(BaseModel):
    field: str
    page_number: int
    text: str


class ExtractRequest(BaseModel):
    document_id: str


class ExtractResponse(BaseModel):
    data: ShipmentExtraction
    sources: list[ExtractionSource]
    missing_fields: list[str]

