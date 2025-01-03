import os
import uuid
import time
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
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


# Indexing phase
DB_DOCS_LIMIT = 10

def load_file_data_to_db():
    TEMP_DIR = "embedded_files"
    st.session_state.setdefault("rag_sources", [])

    if not st.session_state.rag_docs:
        st.warning("No documents to load. Please upload files.")
        return None
    
    docs = []

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
            st.info(f"File {doc_file.name} is already processed. Skipping...")
            continue
        # check document limit
        if len(st.session_state.rag_sources) >= DB_DOCS_LIMIT:
            st.error(f"Maximum number of documents reached {DB_DOCS_LIMIT}")
            break
       
        #if doc_file.name not in st.session_state.rag_docs:
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


# def load_url_data_to_db(): # will be fixed in future
#     if "rag_url" not in st.session_state and st.session_state.rag_url:
#         url = st.session_state.rag_url
#         docs = []
#         if url not in st.session_state.rag_sources:
#             if len(st.session_state.rag_sources) >= DB_DOCS_LIMIT:
#                 st.error(f"Maximum document limit reached ({DB_DOCS_LIMIT}). Cannot load more documents.")
#             try:
#                 loader = WebBaseLoader(url)
#                 docs.extend(loader.load())
#                 st.session_state.rag_sources.append(url)
#                 if docs:
#                     doc_split_chunker(docs)
#                     st.toast(f"Document from URL *{url}* loaded successfully", icon='✅')
#             except Exception as e:
#                 st.error(f"Error loading documents from {url}: {e}")

def doc_split_chunker(docs):
    text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
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
        collection_name=f"{str(time.time()).replace('.', '')[:14]}_" + st.session_state['session_id']
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
        This is the question : {input}

        This is the context : {context}
    """
    system_prompt =  SystemMessage(content=SYSTEM_PROMPT_TEMPLATE)
    human_prompt = HumanMessagePromptTemplate.from_template(template=HUMAN_PROMPT_TEMPLATE)
    chat_template = ChatPromptTemplate([system_prompt, human_prompt])   

    vector_db = st.session_state.get("vector_db")
    if vector_db is None:
        return None
    
    retriever = vector_db.as_retriever()

    llm = ChatOllama(model='llama3.2',temperature=0)

    combine_documents_chain = create_stuff_documents_chain(llm=llm, prompt=chat_template)

    retrieval_chain = create_retrieval_chain(
                    retriever=retriever,
                    combine_docs_chain=combine_documents_chain
    )
    return retrieval_chain

def stream_llm_response(user_query):
    """
    Streams the response from the LLM RAG chain for the given question.
    Args:
        user_query: The input question to query the LLM.
    Returns:
        response in chunks, or a warning message.
    """
    chain = conversational_llm_rag_chain()
    # Check if the chain is None (e.g., missing context)
    if chain is None:
        yield "Please upload a file to provide context before asking a question."
        return
    try:
        response = chain.invoke({"input": user_query})
        if response is None or 'answer' not in response:
            yield "No response received from the model. Please check your inputs or context."
            return
        yield response['answer']
    except Exception as e:
        yield f"An error occurred: {str(e)}"
