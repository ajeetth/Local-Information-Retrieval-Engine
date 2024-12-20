import os
import uuid
import logging
import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import (WebBaseLoader, PyPDFLoader, Docx2txtLoader, TextLoader)
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser   

# Indexing phase
DB_DOCS_LIMIT = 10

def load_file_data_to_db():
    TEMP_DIR = "source_files"
    if "rag_docs" not in st.session_state or not st.session_state.rag_docs:
        return
    docs = []
    st.session_state.setdefault("rag_sources", [])

    def save_temp_file(doc_file) -> str:
        """Save a file temporarily and return its path"""
        os.makedirs(TEMP_DIR, exist_ok=True)
        file_path = os.path.join(TEMP_DIR, doc_file.name)
        with open(file_path, "wb") as file:
            file.write(doc_file.read())
        return file_path
    
    for doc_file in st.session_state.rag_docs:
        # skip duplicate files
        if doc_file.name in st.session_state.rag_sources:
            continue
        # check document limit
        if len(st.session_state.rag_sources) >= DB_DOCS_LIMIT:
            st.error(f"Maximum number of documents reached {DB_DOCS_LIMIT}")
            break
       
        if doc_file.name not in st.session_state.rag_docs:
            try:
                file_path = save_temp_file(doc_file)
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
                st.error(f"failed to process {doc_file.name}")
            finally:
                # cleanup temporary file
                if os.path.exists(file_path):
                    os.remove(file_path)

        if docs:
            doc_split_chunker(docs)
            loaded_files = [doc_file.name for doc_file in st.session_state.rag_docs]
            st.toast(f"Document loaded sucessfully : {', '.join(loaded_files)}.", icon="✅")

def load_url_data_to_db():
    if "rag_url" in st.session_state and st.session_state.rag_url:
        url = st.session_state.rag_url
        docs = []
        if url not in st.session_state.rag_sources:
            if len(st.session_state.rag_sources) >= DB_DOCS_LIMIT:
                st.error(f"Maximum document limit reached ({DB_DOCS_LIMIT}). Cannot load more documents.")
            try:
                loader = WebBaseLoader(url)
                docs.extend(loader.load())
                st.session_state.rag_sources.append(url)
                if docs:
                    doc_split_chunker(docs)
                    st.toast(f"Document from URL *{url}* loaded successfully", icon='✅')
            except Exception as e:
                st.error(f"Error loading documents from {url}: {e}")

def doc_split_chunker(docs):
    text_splitter = RecursiveCharacterTextSplitter(
                    seperator='\n',
                    chunk_size=5000,
                    chunk_overlap=100
    )
    doc_chunks = text_splitter.split_documents(docs)
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = setup_vectorstore(doc_chunks)
    else:
        st.session_state.vector_db.add_documents(doc_chunks)

def setup_vectorstore(doc_chunks):
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    vectorDB = Chroma.from_documents(
        doc_chunks,
        embedding=embeddings,
        collection_name= f"{uuid.uuid4()}_{st.session_state['session_id']}"
    )
    # Manage the number of collections in memory, keeping only the last 20
    MAX_COLLECTIONS = 20
    chroma_client = vectorDB._client
    try:
        collection_names = sorted([collection.name for collection in chroma_client.list_collections()])
        logging.info(f"Number of collections: {len(collection_names)}")
        while len(collection_names) > MAX_COLLECTIONS:
            collection_to_delete = collection_names.pop(0)
            try:
                chroma_client.delete_collection(collection_to_delete)
                logging.info(f"Deleted collection: {collection_to_delete}")
            except Exception as e:
                logging.error(f"Failed to delete collection {collection_to_delete} : {e}")
    except Exception as e:
        logging.error(f"Failed to retrieve or process collections: {e}")
    return vectorDB
    
# RAG system logics

def conversational_llm_rag_chain():
    """
    Creates a conversational Retrieval-Augmented Generation (RAG) chain.
    Args:
        vector_db: A vector database with a retriever method.
    Returns:
        rag_chain: A LangChain pipeline for conversational question answering.
    """
    SYSTEM_PROMPT_TEMPLATE = """ 
        You are an intelligent assistant tasked with answering questions based strictly on the provided context. 
    The context comes from a database of documents or URLs and is highly relevant to the query.
    **Guidelines:**
    1. Your response must only use the provided context.
    2. If the context does not contain enough information to answer the query, respond with: 
        "I'm sorry, I cannot answer this question based on the provided context."
    3. Do not make assumptions or provide information that is not explicitly mentioned in the context.
    """
    HUMAN_PROMPT_TEMPLATE = """ 
        This is the question : {question}

        This is the context : {context}
    """
    system_prompt =  SystemMessage(content=SYSTEM_PROMPT_TEMPLATE)
    human_prompt = HumanMessagePromptTemplate.from_template(template=HUMAN_PROMPT_TEMPLATE)
    chat_template = ChatPromptTemplate([system_prompt, human_prompt])   

    vector_db = st.session_state.get("vector_db")
    if vector_db is None:
        st.error('Vector DB not found in session state, please load documents first')
        return None
    
    retriever = vector_db.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOllama(model='llama3.2',temperature=0)

    rag_chain = ({"context":retriever, "question": RunnablePassthrough()} 
                 | chat_template
                 | llm 
                 | memory
                 | StrOutputParser())
    return rag_chain

def stream_llm_response(user_query):
    """
    Streams the response from the LLM RAG chain for the given question.
    Args:
        vector_db: The vector database object with a `as_retriever()` method.
        question: The input question to query the LLM.
    Returns:
        Generator that yields the response in chunks.
    """
    chain = conversational_llm_rag_chain()
    response = chain.stream({"question": user_query})
    yield response
