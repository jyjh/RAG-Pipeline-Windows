from __future__ import annotations

import os


def main() -> None:
    import requests
    import streamlit as st

    api_url = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
    api_token = os.environ.get("RAG_API_TOKEN", "")
    headers = {"X-API-Token": api_token} if api_token else {}

    st.set_page_config(page_title="FSAE RAG", layout="wide")
    st.title("FSAE RAG")

    with st.sidebar:
        st.caption(api_url)
        mode = st.selectbox("Retrieval", ["hybrid", "vector", "bm25"], index=0)
        top_k = st.slider("Sources", 3, 15, 8)
        if st.button("Refresh documents"):
            st.session_state.pop("documents", None)
        if st.button("Rebuild index"):
            with st.spinner("Indexing"):
                response = requests.post(f"{api_url}/index", headers=headers, timeout=600)
                st.json(response.json())

        if "documents" not in st.session_state:
            try:
                st.session_state.documents = requests.get(
                    f"{api_url}/documents",
                    headers=headers,
                    timeout=30,
                ).json()
            except Exception:
                st.session_state.documents = []
        st.subheader("Documents")
        for doc in st.session_state.documents:
            st.write(doc.get("title", doc.get("doc_id", "")))

    question = st.chat_input("Ask a question from the local knowledge base")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating"):
                response = requests.post(
                    f"{api_url}/query",
                    headers=headers,
                    json={"question": question, "mode": mode, "top_k": top_k},
                    timeout=600,
                )
                response.raise_for_status()
                payload = response.json()
            st.markdown(payload.get("answer", ""))
            st.caption(f"timings: {payload.get('timings', {})}")

            citations = payload.get("citations", [])
            if citations:
                st.subheader("Citations")
                for idx, citation in enumerate(citations, start=1):
                    label = (
                        f"C{idx}: {citation['document_title']} p.{citation['page']} "
                        f"{citation['modality']} {citation['block_id']}"
                    )
                    with st.expander(label):
                        st.write(citation.get("snippet", ""))
                        if citation.get("asset_url"):
                            st.image(f"{api_url}{citation['asset_url']}")

            blocks = payload.get("retrieved_blocks", [])
            if blocks:
                st.subheader("Retrieved Blocks")
                for block in blocks:
                    with st.expander(f"{block['document_title']} p.{block['page']} {block['modality']}"):
                        st.write(block.get("text", ""))


if __name__ == "__main__":
    main()

