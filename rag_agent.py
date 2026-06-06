from dotenv import load_dotenv
load_dotenv()

import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

# ── Session state init ────────────────────────────────────────────────────────
for key, default in {
    "document_uploaded": False,
    "agent": None,
    "vector_store": None,
    "messages": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Document processing ───────────────────────────────────────────────────────
def process_document(path: str):
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents=docs)

    # ✅ FREE: HuggingFace embeddings run locally, no API key needed
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_db = InMemoryVectorStore.from_documents(   # ✅ Fixed typo (was InMemoryVectoryStore)
        documents=docs,
        embedding=embeddings,
    )
    st.session_state.vector_store = vector_db

    llm = ChatGroq(model="llama-3.3-70b-versatile")  # ✅ Free Groq API

    @tool
    def retrieve_context(query: str) -> str:
        """Retrieve documents relevant to a query from the knowledge base."""
        results = vector_db.similarity_search(query=query, k=3)
        return "\n\n".join(doc.page_content for doc in results)

    system_prompt = (
        "You are a helpful assistant that answers questions using retrieved context. "
        "Your knowledge base consists of details from the uploaded document. "
        "ALWAYS use the `retrieve_context` tool for questions requiring external knowledge."
    )

    memory = MemorySaver()  # ✅ Fixed: was InMemorySaver (wrong class name)

    agent = create_react_agent(   # ✅ Fixed: was create_agent (doesn't exist)
        model=llm,
        tools=[retrieve_context],
        prompt=system_prompt,
        checkpointer=memory,
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📄 RAG Chatbot (100% Free)")

if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(
        label="Select PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded:
        with st.spinner("Processing documents... (downloading embedding model on first run)"):
            save_dir = "doc_files"
            os.makedirs(save_dir, exist_ok=True)  # ✅ Fixed: create folder if missing

            for file in uploaded:
                with open(os.path.join(save_dir, file.name), "wb") as f:  # ✅ Fixed path join
                    f.write(file.getvalue())

            process_document(save_dir)
            st.rerun()

if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    query = st.chat_input("Ask anything related to the uploaded documents...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)

        response = st.session_state.agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": {"thread_id": "1"}},  # ✅ Fixed: thread_id must be a string
        )

        answer = response["messages"][-1].content
        st.chat_message("assistant").markdown(answer)  # ✅ Fixed: "ai" → "assistant"
        st.session_state.messages.append({"role": "assistant", "content": answer})