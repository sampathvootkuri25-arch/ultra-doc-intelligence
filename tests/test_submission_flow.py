from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from docx import Document

from app.pipeline import AppPipeline


def test_sample_documents_submission_flow():
    questions = {
        "BOL53657_billoflading.pdf": {
            "Who is the consignee?": "xyz",
            "Who is the shipper?": "AAA",
            "What is the weight?": "56000 lbs",
            "When is pickup scheduled?": "02-08-2026 09:00",
        },
        "LD53657-Carrier-RC.pdf": {
            "What is the carrier rate?": "$400.00 USD",
            "Who is the shipper?": "AAA",
            "Who is the consignee?": "xyz",
            "When is pickup scheduled?": "02-08-2026 15:00",
        },
        "LD53657-Shipper-RC.pdf": {
            "What is the agreed amount?": "$1000.00 USD",
            "Who is the consignee?": "xyz",
            "What is the weight?": "56000.00 lbs",
            "When is pickup scheduled?": "02-08-2026 15:00",
        },
    }

    with TemporaryDirectory() as temp_qdrant:
        pipeline = AppPipeline(qdrant_path=temp_qdrant, use_llm=False)
        try:
            for name, checks in questions.items():
                path = Path("input-documents") / name
                upload = pipeline.upload_document(path.name, path.read_bytes())
                assert upload.status == "indexed"
                assert upload.chunk_count > 0

                for question, expected in checks.items():
                    response = pipeline.ask(upload.document_id, question)
                    assert response.status == "answered"
                    assert expected.lower() in response.answer.lower()

                extraction = pipeline.extract(upload.document_id)
                assert extraction.data.shipment_id == "LD53657"
                assert extraction.data.weight is not None
        finally:
            pipeline.close()


TEST_DOC_CONTENT = """Carrier Rate Confirmation
Shipment ID: LD-DOCX-TXT-001
Shipper: AAA Manufacturing
Consignee: XYZ Retail
Pickup Date: 2026-02-08 15:00
Delivery Date: 2026-02-09 09:30
Equipment Type: Reefer
Mode: FTL
Carrier Name: Blue Transport
Carrier Rate: $400.00 USD
Currency: USD
Weight: 56000 lbs
Special Instructions: Call before arrival.
"""


def _build_test_file(tmp_path: Path, suffix: str, content: str) -> Path:
    path = tmp_path / f"generated-logistics{suffix}"
    if suffix == ".txt":
        path.write_text(content, encoding="utf-8")
        return path
    if suffix == ".docx":
        document = Document()
        for line in content.strip().splitlines():
            document.add_paragraph(line)
        document.save(path)
        return path
    raise ValueError(f"Unsupported suffix {suffix}")


@pytest.mark.parametrize("suffix", [".txt", ".docx"])
def test_docx_and_txt_end_to_end_submission_requirements(tmp_path: Path, suffix: str):
    path = _build_test_file(tmp_path, suffix, TEST_DOC_CONTENT)

    with TemporaryDirectory() as temp_qdrant:
        pipeline = AppPipeline(qdrant_path=temp_qdrant, use_llm=False)
        try:
            upload = pipeline.upload_document(path.name, path.read_bytes())
            assert upload.status == "indexed"
            assert upload.doc_type == "carrier_rate_confirmation"

            rate = pipeline.ask(upload.document_id, "What is the carrier rate?")
            assert rate.status == "answered"
            assert "$400.00 USD" in rate.answer
            assert rate.sources

            consignee = pipeline.ask(upload.document_id, "Who is the consignee?")
            assert consignee.status == "answered"
            assert "XYZ Retail" in consignee.answer

            pickup = pipeline.ask(upload.document_id, "When is pickup scheduled?")
            assert pickup.status == "answered"
            assert "2026-02-08 15:00" in pickup.answer

            instructions = pipeline.ask(upload.document_id, "What are the special instructions?")
            assert instructions.status == "answered"
            assert "Call before arrival" in instructions.answer

            unknown = pipeline.ask(upload.document_id, "What is the invoice number?")
            assert unknown.answer == "Not found in document"
            assert unknown.status in {"not_found", "refused_low_confidence"}

            extraction = pipeline.extract(upload.document_id)
            assert extraction.data.shipment_id == "LD-DOCX-TXT-001"
            assert extraction.data.shipper == "AAA Manufacturing"
            assert extraction.data.consignee == "XYZ Retail"
            assert extraction.data.carrier_name == "Blue Transport"
            assert extraction.data.rate == 400.0
            assert extraction.data.weight == "56000 lbs"
        finally:
            pipeline.close()
