import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Ultra Doc-Intelligence", layout="wide")
st.title("Ultra Doc-Intelligence")
st.caption("Submission-focused reviewer UI for upload, QA, and structured extraction.")

if "document_id" not in st.session_state:
    st.session_state.document_id = None
if "upload_meta" not in st.session_state:
    st.session_state.upload_meta = None

uploaded_file = st.file_uploader("Upload a logistics document", type=["pdf", "docx", "txt"])
if uploaded_file and st.button("Upload document", use_container_width=True):
    response = requests.post(
        f"{API_URL}/upload",
        files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")},
        timeout=120,
    )
    if response.ok:
        st.session_state.upload_meta = response.json()
        st.session_state.document_id = st.session_state.upload_meta["document_id"]
        st.success("Document indexed successfully.")
    else:
        st.error(response.text)

if st.session_state.upload_meta:
    st.subheader("Current document")
    st.json(st.session_state.upload_meta)

question = st.text_input("Ask a question about the current document")
if st.button("Ask", use_container_width=True, disabled=not st.session_state.document_id or not question):
    response = requests.post(
        f"{API_URL}/ask",
        json={"document_id": st.session_state.document_id, "question": question},
        timeout=120,
    )
    if response.ok:
        result = response.json()
        st.subheader("Answer")
        st.write(result["answer"])
        st.write(f"Confidence: {result['confidence']}")
        st.write(f"Status: {result['status']}")
        st.subheader("Sources")
        for source in result["sources"]:
            st.markdown(f"**Page {source['page_number']} | {source['chunk_id']} | score={source['score']}**")
            st.code(source["text"])
    else:
        st.error(response.text)

if st.button("Run structured extraction", use_container_width=True, disabled=not st.session_state.document_id):
    response = requests.post(
        f"{API_URL}/extract",
        json={"document_id": st.session_state.document_id},
        timeout=120,
    )
    if response.ok:
        result = response.json()
        st.subheader("Structured extraction")
        st.json(result["data"])
        st.write("Missing fields:", ", ".join(result["missing_fields"]) or "None")
        if result["sources"]:
            st.subheader("Evidence")
            for source in result["sources"]:
                st.markdown(f"**{source['field']} | page {source['page_number']}**")
                st.code(source["text"])
    else:
        st.error(response.text)

