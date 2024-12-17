import streamlit as st
from llm_utils import (load_file_data_to_db, load_url_data_to_db)

st.set_page_config(
    page_title='Retrival Engine',
    page_icon = '📂',
    layout='centered',
    initial_sidebar_state='expanded'
)

st.title("Local Information Retrieval Engine 🤖💬")

with st.sidebar:
    st.header("RAG Sources:")
    # file input for RAG with documents
    st.file_uploader(
        "Upload a document file 📄",
        type = ['pdf', 'txt', 'docx'],
        accept_multiple_files=True,
        on_change= load_file_data_to_db, # function which will load the uploaded doc to db
        key="rag_docs"
    )
    # URL input for RAG with websites
    st.text_input(
        "Paste a document URL 🔗",
        placeholder='https://example.com',
        on_change=load_url_data_to_db,
        key="rag_url"
    )