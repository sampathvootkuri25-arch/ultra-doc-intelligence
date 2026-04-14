from __future__ import annotations

import atexit
import json
import re
import uuid
from pathlib import Path

from openai import AzureOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer
import tiktoken

from app.config import (
    ANSWER_TOP_K,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    CHUNK_OVERLAP_TOKENS,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LOW_CONFIDENCE_THRESHOLD,
    MAX_CHUNK_TOKENS,
    QDRANT_API_KEY,
    QDRANT_MODE,
    QDRANT_PATH,
    QDRANT_URL,
    RETRIEVAL_TOP_K,
    UPLOAD_DIR,
)
from app.document_parsers import FieldValue, ParsedBlock, ParsedDocument, parse_document
from app.schemas import AskResponse, ExtractResponse, ExtractionSource, ShipmentExtraction, SourceSnippet, UploadResponse


QUESTION_FIELD_MAP = [
    ({"consignee", "receiver"}, "consignee", "The consignee is {value}."),
    ({"shipper", "consignor"}, "shipper", "The shipper is {value}."),
    ({"carrier rate", "carrier pay"}, "rate", "The carrier rate is {value}."),
    ({"agreed amount", "customer rate", "rate"}, "rate", "The agreed amount is {value}."),
    ({"weight"}, "weight", "The weight is {value}."),
    ({"equipment", "equipment type"}, "equipment_type", "The equipment type is {value}."),
    ({"pickup scheduled", "pickup", "shipping date"}, "pickup_datetime", "The pickup is scheduled for {value}."),
    ({"delivery", "drop", "delivery date"}, "delivery_datetime", "The delivery is scheduled for {value}."),
    ({"carrier name", "carrier"}, "carrier_name", "The carrier name is {value}."),
    ({"shipment id", "reference id", "load id"}, "shipment_id", "The shipment ID is {value}."),
    ({"mode", "load type"}, "mode", "The mode is {value}."),
]

FIELD_RESPONSE_TEMPLATES = {
    "consignee": "The consignee is {value}.",
    "shipper": "The shipper is {value}.",
    "weight": "The weight is {value}.",
    "equipment_type": "The equipment type is {value}.",
    "pickup_datetime": "The pickup is scheduled for {value}.",
    "delivery_datetime": "The delivery is scheduled for {value}.",
    "carrier_name": "The carrier name is {value}.",
    "shipment_id": "The shipment ID is {value}.",
    "mode": "The mode is {value}.",
    "currency": "The currency is {value}.",
}


class AppPipeline:
    def __init__(self, qdrant_path: str | None = None, use_llm: bool = True) -> None:
        self._embedder: SentenceTransformer | None = None
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self._qdrant_path = qdrant_path
        self.qdrant = self._build_qdrant_client()
        self._documents: dict[str, ParsedDocument] = {}
        self._llm = self._build_llm_client() if use_llm else None

    def _build_qdrant_client(self) -> QdrantClient:
        if QDRANT_MODE == "server" and QDRANT_URL:
            return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        return QdrantClient(path=self._qdrant_path or QDRANT_PATH)

    def _build_llm_client(self):
        required = [
            AZURE_OPENAI_API_KEY,
            AZURE_OPENAI_ENDPOINT,
            AZURE_OPENAI_API_VERSION,
            AZURE_OPENAI_DEPLOYMENT,
        ]
        if not all(required):
            return None
        return AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
        )

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedder

    def _ensure_collection(self) -> None:
        if self.qdrant.collection_exists(COLLECTION_NAME):
            return
        dim = len(self.embedder.encode(["dimension probe"], normalize_embeddings=True)[0])
        self.qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        )

    def upload_document(self, filename: str, content: bytes) -> UploadResponse:
        self._ensure_collection()
        document_id = str(uuid.uuid4())
        stored_path = Path(UPLOAD_DIR) / f"{document_id}-{filename}"
        stored_path.write_bytes(content)

        parsed = parse_document(stored_path)
        chunks = self._build_chunks(document_id, parsed)
        self._index_chunks(document_id, filename, parsed.doc_type, chunks)
        self._documents[document_id] = parsed

        return UploadResponse(
            document_id=document_id,
            filename=filename,
            doc_type=parsed.doc_type,
            pages=len(parsed.pages),
            chunk_count=len(chunks),
            status="indexed",
        )

    def ask(self, document_id: str, question: str) -> AskResponse:
        parsed = self._documents.get(document_id)
        if not parsed:
            raise KeyError("Unknown document_id")

        direct = self._answer_from_fields(parsed, question)
        if direct is not None:
            return direct

        results = self._search(document_id, question)
        if not results:
            return AskResponse(answer="Not found in document", confidence=0.0, status="not_found", sources=[])

        sources = [
            SourceSnippet(
                page_number=result.payload["page_number"],
                chunk_id=result.payload["chunk_id"],
                score=round(float(result.score), 4),
                text=result.payload["source_text"][:700],
            )
            for result in results[:ANSWER_TOP_K]
        ]
        confidence = self._confidence(question, results)
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            return AskResponse(
                answer="Not found in document",
                confidence=round(confidence, 2),
                status="refused_low_confidence",
                sources=sources,
            )

        if self._llm:
            answer = self._llm_answer(question, sources)
        else:
            answer = self._extractive_answer(question, parsed, sources)

        status = "answered" if answer != "Not found in document" else "not_found"
        return AskResponse(answer=answer, confidence=round(confidence, 2), status=status, sources=sources)

    def extract(self, document_id: str) -> ExtractResponse:
        parsed = self._documents.get(document_id)
        if not parsed:
            raise KeyError("Unknown document_id")

        raw = {
            "shipment_id": self._field_value(parsed.fields.get("shipment_id")),
            "shipper": self._field_value(parsed.fields.get("shipper")),
            "consignee": self._field_value(parsed.fields.get("consignee")),
            "pickup_datetime": self._field_value(parsed.fields.get("pickup_datetime")),
            "delivery_datetime": self._field_value(parsed.fields.get("delivery_datetime")),
            "equipment_type": self._field_value(parsed.fields.get("equipment_type")),
            "mode": self._field_value(parsed.fields.get("mode")),
            "rate": self._float_or_none(self._field_value(parsed.fields.get("rate"))),
            "currency": self._field_value(parsed.fields.get("currency")),
            "weight": self._field_value(parsed.fields.get("weight")),
            "carrier_name": self._field_value(parsed.fields.get("carrier_name")),
        }

        if self._llm:
            raw = self._llm_extract(parsed, raw)

        data = ShipmentExtraction(**raw)
        sources = []
        for field_name, value in data.model_dump().items():
            field = parsed.fields.get(field_name)
            if field and value is not None:
                sources.append(
                    ExtractionSource(
                        field=field_name,
                        page_number=field.page_number,
                        text=field.source_text[:500],
                    )
                )
        missing_fields = [field for field, value in data.model_dump().items() if value is None]
        return ExtractResponse(data=data, sources=sources, missing_fields=missing_fields)

    def _build_chunks(self, document_id: str, parsed: ParsedDocument) -> list[dict]:
        chunks: list[dict] = []
        for field_name, field in parsed.fields.items():
            chunks.append(
                {
                    "chunk_id": f"field-{field_name}",
                    "page_number": field.page_number,
                    "section_name": field_name,
                    "chunk_type": "field",
                    "source_text": field.source_text,
                    "semantic_text": self._field_semantic_text(field_name, field),
                    "aliases": field.aliases,
                }
            )

        for block in parsed.blocks:
            for piece_index, piece in enumerate(self._chunk_text(block.text), start=1):
                chunks.append(
                    {
                        "chunk_id": f"{block.block_id}-c{piece_index}",
                        "page_number": block.page_number,
                        "section_name": block.section_name,
                        "chunk_type": block.chunk_type,
                        "source_text": piece,
                        "semantic_text": piece,
                        "aliases": block.aliases,
                    }
                )
        return chunks

    def _field_semantic_text(self, field_name: str, field: FieldValue) -> str:
        label = field_name.replace("_", " ")
        aliases = ", ".join(field.aliases)
        return f"{label}: {field.value}. aliases: {aliases}. source: {field.source_text}"

    def _chunk_text(self, text: str) -> list[str]:
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= MAX_CHUNK_TOKENS:
            return [text]
        parts: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + MAX_CHUNK_TOKENS, len(tokens))
            part = self._tokenizer.decode(tokens[start:end]).strip()
            if part:
                parts.append(part)
            if end == len(tokens):
                break
            start = max(0, end - CHUNK_OVERLAP_TOKENS)
        return parts

    def _index_chunks(self, document_id: str, filename: str, doc_type: str, chunks: list[dict]) -> None:
        self.qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[qmodels.FieldCondition(key="document_id", match=qmodels.MatchValue(value=document_id))]
                )
            ),
        )
        vectors = self.embedder.encode([chunk["semantic_text"] for chunk in chunks], normalize_embeddings=True)
        self.qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                qmodels.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector.tolist(),
                    payload={
                        "document_id": document_id,
                        "filename": filename,
                        "doc_type": doc_type,
                        "page_number": chunk["page_number"],
                        "section_name": chunk["section_name"],
                        "chunk_id": chunk["chunk_id"],
                        "chunk_type": chunk["chunk_type"],
                        "source_text": chunk["source_text"],
                        "semantic_text": chunk["semantic_text"],
                        "aliases": chunk["aliases"],
                    },
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )

    def _answer_from_fields(self, parsed: ParsedDocument, question: str) -> AskResponse | None:
        normalized = " ".join(re.findall(r"[a-z0-9]+", question.lower()))
        for terms, field_name, template in QUESTION_FIELD_MAP:
            if any(term in normalized for term in terms):
                field = parsed.fields.get(field_name)
                if not field:
                    return None
                answer = self._format_field_answer(field_name, field, parsed, question, template)
                return AskResponse(
                    answer=answer,
                    confidence=0.96,
                    status="answered",
                    sources=[
                        SourceSnippet(
                            page_number=field.page_number,
                            chunk_id=f"field-{field_name}",
                            score=0.99,
                            text=field.source_text[:700],
                        )
                    ],
                )
        return None

    def _search(self, document_id: str, question: str):
        query = self.embedder.encode(question, normalize_embeddings=True).tolist()
        results = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query,
            limit=RETRIEVAL_TOP_K,
            query_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="document_id", match=qmodels.MatchValue(value=document_id))]
            ),
        ).points
        return self._rerank(question, results)

    def _rerank(self, question: str, results):
        question_terms = set(re.findall(r"[a-z0-9]+", question.lower()))
        ranked = []
        for result in results:
            payload = result.payload
            text = f"{payload['section_name']} {payload['source_text']} {' '.join(payload.get('aliases', []))}".lower()
            overlap = len(question_terms & set(re.findall(r"[a-z0-9]+", text)))
            alias_bonus = sum(1 for alias in payload.get("aliases", []) if alias and alias in question.lower())
            type_bonus = 0.08 if payload.get("chunk_type") == "field" else 0.0
            result.score = float(result.score) + overlap * 0.015 + alias_bonus * 0.05 + type_bonus
            ranked.append(result)
        return sorted(ranked, key=lambda item: item.score, reverse=True)

    def _confidence(self, question: str, results) -> float:
        if not results:
            return 0.0
        question_terms = set(re.findall(r"[a-z0-9]+", question.lower()))
        top = results[0]
        scores = [max(0.0, min(1.0, float(item.score))) for item in results[:3]]
        mean3 = sum(scores) / len(scores)
        overlap = len(question_terms & set(re.findall(r"[a-z0-9]+", top.payload["source_text"].lower())))
        exact_support = min(1.0, overlap / max(1, len(question_terms)))
        field_bonus = 1.0 if top.payload.get("chunk_type") == "field" else 0.75
        agreement = 1.0 if len(scores) == 1 or max(scores) - min(scores) < 0.25 else 0.7
        return 0.30 * scores[0] + 0.20 * mean3 + 0.25 * exact_support + 0.15 * field_bonus + 0.10 * agreement

    def _llm_answer(self, question: str, sources: list[SourceSnippet]) -> str:
        context = "\n\n".join(f"[page {source.page_number}] {source.text}" for source in sources)
        response = self._llm.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer only from the provided context. "
                        "Return JSON with the single key answer. "
                        "If not present, answer must be Not found in document."
                    ),
                },
                {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
            ],
        )
        data = json.loads(response.choices[0].message.content)
        return data.get("answer", "Not found in document")

    def _extractive_answer(self, question: str, parsed: ParsedDocument, sources: list[SourceSnippet]) -> str:
        if not sources:
            return "Not found in document"

        top = sources[0]
        if top.chunk_id.startswith("field-"):
            field_name = top.chunk_id.replace("field-", "", 1)
            field = parsed.fields.get(field_name)
            if field:
                return self._format_field_answer(field_name, field, parsed, question)

        candidates = self._candidate_spans(sources)
        if not candidates:
            return "Not found in document"

        scored = sorted(
            ((self._score_candidate(question, candidate), candidate) for candidate in candidates),
            key=lambda item: item[0],
            reverse=True,
        )
        best_score, best_candidate = scored[0]
        if best_score < 0.18:
            return "Not found in document"

        answer = self._normalize_candidate_answer(question, best_candidate)
        if not answer:
            return "Not found in document"
        if not any(answer.lower() in source.text.lower() for source in sources):
            return "Not found in document"
        return answer

    def _llm_extract(self, parsed: ParsedDocument, raw: dict) -> dict:
        context = "\n\n".join(
            f"[page {page.page_number}] {page.text[:2500]}"
            for page in parsed.pages
        )
        response = self._llm.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Fill the provided shipment JSON from the document using null for missing fields. "
                        "Do not guess."
                    ),
                },
                {"role": "user", "content": json.dumps({"current": raw, "document": context})},
            ],
        )
        data = json.loads(response.choices[0].message.content)
        return {**raw, **data}

    def _candidate_spans(self, sources: list[SourceSnippet]) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()
        for source in sources[:3]:
            raw_lines = [line.strip(" -\t") for line in source.text.splitlines() if line.strip()]
            for line in raw_lines:
                for span in self._split_candidate_line(line):
                    normalized = span.strip()
                    key = normalized.lower()
                    if len(normalized) < 4 or key in seen:
                        continue
                    seen.add(key)
                    candidates.append(normalized)
        return candidates

    @staticmethod
    def _split_candidate_line(line: str) -> list[str]:
        parts = [line]
        sentence_parts = [
            piece.strip()
            for piece in re.split(r"(?<=[.!?])\s+| \| ", line)
            if piece.strip()
        ]
        parts.extend(sentence_parts)
        return parts

    def _score_candidate(self, question: str, candidate: str) -> float:
        q_terms = set(re.findall(r"[a-z0-9]+", question.lower()))
        c_terms = set(re.findall(r"[a-z0-9]+", candidate.lower()))
        if not q_terms or not c_terms:
            return 0.0
        overlap = len(q_terms & c_terms) / max(1, len(q_terms))
        score = overlap
        question_lower = question.lower()
        candidate_lower = candidate.lower()
        if ":" in candidate:
            score += 0.08
        if "when" in question_lower and self._extract_datetime(candidate):
            score += 0.25
        if any(term in question_lower for term in ["rate", "amount", "price", "cost"]) and self._extract_money(candidate):
            score += 0.25
        if "weight" in question_lower and self._extract_weight(candidate):
            score += 0.25
        if "who" in question_lower and any(label in candidate_lower for label in ["shipper", "consignee", "carrier", "pickup", "drop"]):
            score += 0.18
        return score

    def _normalize_candidate_answer(self, question: str, candidate: str) -> str | None:
        question_lower = question.lower()
        if "when" in question_lower:
            return self._extract_datetime(candidate)
        if any(term in question_lower for term in ["rate", "amount", "price", "cost"]):
            money = self._extract_money(candidate)
            if money:
                return money
        if "weight" in question_lower:
            weight = self._extract_weight(candidate)
            if weight:
                return weight
        if "who" in question_lower:
            label_match = re.search(r"^[A-Za-z ]+:\s*(.+)$", candidate)
            if label_match:
                return label_match.group(1).strip()
        return candidate.strip()

    @staticmethod
    def _field_value(field: FieldValue | None) -> str | None:
        return field.value if field else None

    @staticmethod
    def _float_or_none(value: str | None) -> float | None:
        if value is None:
            return None
        cleaned = re.sub(r"[^0-9.]", "", value)
        return float(cleaned) if cleaned else None

    @staticmethod
    def _format_currency(value: str, currency_field: FieldValue | None) -> str:
        currency = currency_field.value if currency_field else "USD"
        cleaned = value.strip()
        if not cleaned.startswith("$"):
            cleaned = f"${cleaned}"
        return f"{cleaned} {currency}".strip()

    def _format_field_answer(
        self,
        field_name: str,
        field: FieldValue,
        parsed: ParsedDocument,
        question: str,
        template: str | None = None,
    ) -> str:
        normalized = " ".join(re.findall(r"[a-z0-9]+", question.lower()))
        if field_name == "rate" and parsed.doc_type == "carrier_rate_confirmation" and "carrier" in normalized:
            return f"The carrier rate is {self._format_currency(field.value, parsed.fields.get('currency'))}."
        if field_name == "rate":
            return f"The agreed amount is {self._format_currency(field.value, parsed.fields.get('currency'))}."
        answer_template = template or FIELD_RESPONSE_TEMPLATES.get(field_name)
        if answer_template:
            return answer_template.format(value=field.value)
        return field.value

    @staticmethod
    def _extract_datetime(text: str) -> str | None:
        match = re.search(
            r"(?:\b\d{2}-\d{2}-\d{4}\b|\b\d{4}-\d{2}-\d{2}\b)(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?",
            text,
        )
        return match.group(0) if match else None

    @staticmethod
    def _extract_money(text: str) -> str | None:
        match = re.search(r"\$[0-9,]+(?:\.[0-9]{2})?(?:\s*[A-Z]{3})?", text)
        if match:
            return match.group(0).strip()
        match = re.search(r"\b([0-9,]+(?:\.[0-9]{2})?)\s*([A-Z]{3})\b", text)
        if match:
            return f"${match.group(1)} {match.group(2)}"
        return None

    @staticmethod
    def _extract_weight(text: str) -> str | None:
        match = re.search(r"\b[0-9][0-9,.\s]*lbs\b", text, flags=re.IGNORECASE)
        return match.group(0).strip() if match else None

    @staticmethod
    def _best_named_entity(text: str) -> str | None:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.match(r"(?:Shipper|Consignee|Carrier|Customer):\s*(.+)", line, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def close(self) -> None:
        try:
            self.qdrant.close()
        except Exception:
            pass


_pipeline_instance: AppPipeline | None = None


def get_pipeline() -> AppPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = AppPipeline()
        atexit.register(_pipeline_instance.close)
    return _pipeline_instance
