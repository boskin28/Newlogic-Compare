import streamlit as st
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain_pinecone import PineconeVectorStore
import hmac
import uuid
import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# --- Authentication (unchanged) ---
def check_password():
    def login_form():
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)
    def password_entered():
        if st.session_state["username"] in st.secrets["passwords"] and hmac.compare_digest(
            st.session_state["password"], st.secrets.passwords[st.session_state["username"]]):
            st.session_state["password_correct"] = True
            del st.session_state["username"]
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
    if st.session_state.get("password_correct", False):
        return True
    login_form()
    if "password_correct" in st.session_state:
        st.error("Username or password incorrect")
    return False

if not check_password():
    st.stop()

# --- Load secrets & init clients ---
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
PINECONE_API_KEY = st.secrets['PINECONE_API_KEY']
index_name = st.secrets['INDEX_NAME']
host = st.secrets['HOST']
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=host)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model="gpt-4")
chain = load_qa_chain(llm, chain_type="stuff")

# --- Helpers ---
def get_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    return "".join(page.extract_text() or "" for page in reader.pages)

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.split_text(text)

def embed_full_doc(text):
    chunks = get_text_chunks(text)
    embs = embeddings.embed_documents(chunks)
    return np.mean(np.array(embs), axis=0)

def upsert_chunks(text_chunks, namespace):
    batch_size = 200
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i+batch_size]
        texts = [f"{namespace}: {c}" for c in batch]
        embs = embeddings.embed_documents(texts)
        vectors = [(str(uuid.uuid4()), emb, {"source": namespace}) for emb in embs]
        index.upsert(vectors=vectors, namespace=namespace)

# --- UI layout with tabs ---
tab1, tab2 = st.tabs(["Q&A", "Compare Docs"])

# Keep uploaded files in session
if 'files' not in st.session_state:
    st.session_state.files = {}

# File uploader shared by both tabs
uploaded = st.file_uploader("Upload PDF files", accept_multiple_files=True, type='pdf')
for f in uploaded:
    st.session_state.files[f.name] = f

# Tab 1: Q&A
with tab1:
    st.header("Ask Questions")
    if 'vs' not in st.session_state and st.session_state.files:
        # build vectorstore from all docs
        all_texts = []
        for name, file in st.session_state.files.items():
            text = get_pdf_text(file)
            chunks = get_text_chunks(text)
            upsert_chunks(chunks, namespace=name)
        st.session_state.vs = PineconeVectorStore.from_existing_index(
            embedding=embeddings, index_name=index_name)
    if 'vs' in st.session_state:
        if prompt := st.text_input("Enter your question about the documents..."):
            docs = st.session_state.vs.similarity_search(prompt)
            answer = chain.run(input_documents=docs, question=prompt)
            st.write(answer)
    else:
        st.info("Upload at least one PDF to enable Q&A.")

# Tab 2: Compare
with tab2:
    st.header("Compare Two Documents")
    if len(st.session_state.files) < 2:
        st.info("Upload two or more PDFs to compare.")
    else:
        names = list(st.session_state.files.keys())
        doc_a = st.selectbox("First document", names, key="a")
        doc_b = st.selectbox("Second document", names, key="b")
        if st.button("Compare") and doc_a != doc_b:
            text_a = get_pdf_text(st.session_state.files[doc_a])
            text_b = get_pdf_text(st.session_state.files[doc_b])
            emb_a = embed_full_doc(text_a)
            emb_b = embed_full_doc(text_b)
            sim = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a)*np.linalg.norm(emb_b))
            st.metric("Cosine Similarity", f"{sim:.3f}")
            # summary
            prompt = f"Compare titles '{doc_a}' vs '{doc_b}'. What are their main similarities and differences?"
            docs_for_chain = [type('D',( ),{'page_content': text_a})(), type('D',(),{'page_content': text_b})()]
            summary = chain.run(input_documents=docs_for_chain, question=prompt)
            st.subheader("Summary of Comparison")
            st.write(summary)
