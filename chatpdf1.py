import streamlit as st
from PyPDF2 import PdfReader
import os
import time
from dotenv import load_dotenv

# LangChain Imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Google Gemini
import google.generativeai as genai

# ---------------------------
# Environment Setup
# ---------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ---------------------------
# PDF Text Extraction
# ---------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text


# ---------------------------
# Text Chunking
# ---------------------------
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    return text_splitter.split_text(text)


# ---------------------------
# Vector Store Creation
# ---------------------------
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    if os.path.exists("faiss_index"):
        vector_store = FAISS.load_local("faiss_index", embeddings)
        vector_store.add_texts(text_chunks)
    else:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    vector_store.save_local("faiss_index")


# ---------------------------
# Load Vector Store
# ---------------------------
def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    return FAISS.load_local("faiss_index", embeddings)


# ---------------------------
# Build Conversational RAG Chain
# ---------------------------
def get_conversational_chain(vector_store):
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )

    return chain


# ---------------------------
# Handle User Query
# ---------------------------
def user_input(user_question):
    vector_store = load_vector_store()
    chain = get_conversational_chain(vector_store)

    start_time = time.time()

    response = chain({"question": user_question})

    end_time = time.time()
    latency = round(end_time - start_time, 2)

    st.subheader("Answer")
    st.write(response["answer"])

    st.subheader("Response Time")
    st.write(f"{latency} seconds")

    st.subheader("Sources Used")
    for doc in response["source_documents"]:
        st.write(doc.page_content[:400])
        st.write("---")


# ---------------------------
# Streamlit App
# ---------------------------
def main():
    st.set_page_config(page_title="Chat with PDF - RAG System", layout="wide")
    st.title("📄 RAG-Based PDF Chat System (Gemini + FAISS)")
    st.write("Upload documents and ask contextual questions.")

    user_question = st.text_input("Ask a question from your uploaded documents:")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.header("📂 Upload Documents")
        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True
        )

        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Documents indexed successfully!")
            else:
                st.warning("Please upload at least one PDF file.")


if __name__ == "__main__":
    main()