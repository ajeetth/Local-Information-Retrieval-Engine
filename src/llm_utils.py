from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import (UnstructuredPDFLoader, WebBaseLoader, PyPDFLoader, Docx2txtLoader)
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import conversational_retrieval
from langchain.memory import ConversationBufferMemory

def load_document(docs):
    loader = UnstructuredPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
                    seperator='\n',
                    chunk_size=5000,
                    chunk_overlap=100
    )
    doc_chunks = text_splitter.split_documents(documents)
    return doc_chunks

def setup_vectorstore(doc_chunks):
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    vectorStore = Chroma.from_documents(doc_chunks, 
                                         embedding=embeddings)
    return vectorStore
    
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

