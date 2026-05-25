import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from pdf_loader import (
    split_documents,
    EmbeddingManager,
    VectorStore,
    RAGRetriever,
    GroqLLM
)

from langchain_community.document_loaders import PyMuPDFLoader

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Banking Compliance System",
    layout="wide"
)

st.title("🏦 AI Banking Compliance & Loan Risk Advisory")

# =========================
# INITIALIZE
# =========================

@st.cache_resource
def initialize():

    embedding_manager = EmbeddingManager()

    vectorstore = VectorStore(
        collection_name="banking_docs",
        persist_directory="./vector_store"
    )

    retriever = RAGRetriever(
        vectorstore,
        embedding_manager
    )

    llm = GroqLLM(
        model_name="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    return embedding_manager, vectorstore, retriever, llm

embedding_manager, vectorstore, rag_retriever, groq_llm = initialize()

# =========================
# PDF UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

# =========================
# PROCESS PDF
# =========================

if uploaded_file:

    if st.button("Process PDF"):

        with st.spinner("Processing PDF..."):

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:

                tmp_file.write(uploaded_file.read())

                temp_pdf_path = tmp_file.name

            loader = PyMuPDFLoader(temp_pdf_path)

            documents = loader.load()

            st.write(f"Loaded Pages: {len(documents)}")

            chunks = split_documents(
                documents,
                chunk_size=500,
                chunk_overlap=100
            )

            st.write(f"Generated Chunks: {len(chunks)}")

            texts = [doc.page_content for doc in chunks]

            embeddings = embedding_manager.generate_embeddings(texts)

            vectorstore.add_documents(
                chunks,
                embeddings
            )

            st.success("PDF Processed Successfully")

# =========================
# QUESTION
# =========================

query = st.text_input(
    "Ask Question"
)

# =========================
# ANALYZE
# =========================

if st.button("Analyze"):

    with st.spinner("Retrieving Context..."):

        result = rag_retriever.retrieve(
            query=query,
            top_k=10,
            score_threshold=-1
        )

        st.write("Retrieved Docs:", len(result))

        if len(result) == 0:

            st.error("No context found")

        else:

            context = "\n\n".join(
                [doc["content"] for doc in result]
            )

            answer = groq_llm.generate_response(
                query=query,
                context=context
            )

            st.subheader("AI Analysis")

            st.write(answer)

            with st.expander("Retrieved Chunks"):

                for idx, doc in enumerate(result):

                    st.markdown(f"### Chunk {idx+1}")

                    st.write(
                        f"Score: {doc['similarity_score']}"
                    )

                    st.write(doc["content"])

                    st.divider()
