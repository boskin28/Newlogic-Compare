import streamlit as st
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain_pinecone import PineconeVectorStore
import hmac

# Authentication

def check_password():
    def login_form():
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        if (
            st.session_state["username"] in st.secrets["passwords"]
            and hmac.compare_digest(
                st.session_state["password"],
                st.secrets.passwords[st.session_state["username"]],
            )
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
            del st.session_state["username"]
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

# Load secrets
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
PINECONE_API_KEY = st.secrets['PINECONE_API_KEY']
PINECONE_API_ENV = st.secrets['ENVIRONMENT']
index_name = st.secrets['INDEX_NAME']
page_title = st.secrets.get('PAGE_TITLE', 'Q&A App')

# Initialize Pinecone client & OpenAI embeddings
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=st.secrets['HOST'])
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

def get_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    return "".join(page.extract_text() or "" for page in reader.pages)


def get_text_chunks(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.split_text(text)


def get_vectorstore(text_chunks, namespace="default"):
    texts = [chunk for chunk in text_chunks]
    metadatas = [{"source": namespace} for _ in texts]
    # Use from_texts to create or connect and upsert
    vs = PineconeVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        index_name=index_name,
        namespace=namespace,
    )
    return vs

# Set up Q&A chain
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model="gpt-4")
chain = load_qa_chain(llm, chain_type="stuff")

st.title(page_title)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything about your uploads..."):
    # 1) Echo the user query in the chat
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) Record it in history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3) Ensure vectorstore is ready
    if "vs" not in st.session_state:
        st.session_state.vs = get_vectorstore(
            [get_pdf_text(f) for f in st.session_state.get('uploaded_files', [])],
            namespace="docs"
        )
    vs = st.session_state.vs

    # 4) Run similarity search & QA chain
    docs = vs.similarity_search(prompt)
    answer = chain.run(input_documents=docs, question=prompt)

    # 5) Append and display the assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
