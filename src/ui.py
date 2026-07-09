from __future__ import annotations

import os


def main() -> None:
    import requests
    import streamlit as st

    api_url = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
    token = os.environ.get("RAG_API_TOKEN", "")
    headers = {"X-API-Token": token} if token else {}
    st.set_page_config(page_title="FSAE RAG", layout="wide")
    st.title("FSAE RAG")
    with st.sidebar:
        st.caption(api_url)
        mode = st.selectbox("Retrieval", ["hybrid", "vector", "bm25"], index=0)
        top_k = st.slider("Sources", 3, 15, 8)
        if st.button("Rebuild index"):
            st.json(requests.post(f"{api_url}/index", headers=headers, timeout=600).json())
        st.subheader("Documents")
        try:
            for doc in requests.get(f"{api_url}/documents", headers=headers, timeout=30).json():
                st.write(doc.get("title", doc.get("doc_id", "")))
        except Exception:
            st.write("API unavailable")
    question = st.chat_input("Ask a question from the local knowledge base")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            payload = requests.post(f"{api_url}/query", headers=headers,
                                    json={"question": question, "mode": mode, "top_k": top_k}, timeout=600).json()
            st.markdown(payload.get("answer", ""))
            for idx, citation in enumerate(payload.get("citations", []), start=1):
                with st.expander(f"C{idx}: {citation['document_title']} p.{citation['page']} {citation['modality']}"):
                    st.write(citation.get("snippet", ""))
                    if citation.get("asset_url"):
                        st.image(f"{api_url}{citation['asset_url']}")


if __name__ == "__main__":
    main()

