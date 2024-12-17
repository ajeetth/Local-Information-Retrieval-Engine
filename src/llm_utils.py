import os
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import (WebBaseLoader, PyPDFLoader, Docx2txtLoader, TextLoader)
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate

# Indexing phase

def load_file_data_to_db():
    DB_DOCS_LIMIT = 10
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = []
        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_docs:
                if len(doc_file) < DB_DOCS_LIMIT:
                    os.makedirs("source_files", exist_ok=True)
                    file_path = f"./source_files/{doc_file.name}"
                    with open(file_path, 'wb') as file:
                        file.write(doc_file.read())
                    try:
                        if doc_file.type == 'application/pdf':
                            loader = PyPDFLoader(file_path=file_path)
                        elif doc_file.name.endswith(".docx"):
                            loader = Docx2txtLoader(file_path=file_path)
                        elif doc_file.type in ["text/plain", "text/markdown"]:
                            loader = TextLoader(file_path=file_path)
                        else:
                            st.warning(f"Document type {doc_file.type} not supported!")
                            continue

                        docs.extend(loader.load())
                        st.session_state.rag_sources.append(doc_file.name)

                    except Exception as e:
                        st.toast(f"error loading document {doc_file.name} : {e}")
                        print(f"Error loading document {doc_file.name} : {e}")
                    finally:
                        os.remove(file_path)
                else:
                    st.error(f"Maximum number of documents reached ({DB_DOCS_LIMIT}).")    
            if docs:
                doc_split_chunker(docs)
                loaded_files = [doc_file.name for doc_file in st.session_state.rag_docs]
                st.toast(f"Document loaded sucessfully : {', '.join(loaded_files)}.", icon="✅")


def load_url_data_to_db():
    DB_DOCS_LIMIT = 10
    if "rag_url" in st.session_state and st.session_state.rag_url:
        url = st.session_state.rag_url
        docs = []
        if url not in st.session_state.rag_sources:
            if len(st.session_state.rag) < DB_DOCS_LIMIT:
                try:
                    loader = WebBaseLoader(url)
                    docs.extend(loader.load())
                    st.session_state.rag_sources.append(url)
                except Exception as e:
                    st.error(f"Error loading documents from {url}: {e}")
                if docs:
                    doc_split_chunker(docs)
                    st.toast(f"Document from URL *{url}* loaded successfully", icon='✅')


def doc_split_chunker(docs):
    text_splitter = RecursiveCharacterTextSplitter(
                    seperator='\n',
                    chunk_size=5000,
                    chunk_overlap=100
    )
    doc_chunks = text_splitter.split_documents(docs)
    return doc_chunks

def setup_vectorstore(doc_chunks):
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    vectorStore = Chroma.from_documents(doc_chunks, 
                                         embedding=embeddings)
    return vectorStore

def prompting():
    pass
    
def create_chain(vectorStore):
    llm = ChatOllama(model='llama3.2',
                     temperature=0)
    retriever = vectorStore.as_retriever()
    memory = ConversationBufferMemory(
        llm = llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    chain = retriever | llm | memory
    return chain

# RAG system logics
