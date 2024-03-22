from langchain.vectorstores import DocArrayInMemorySearch
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)


def chroma_vector_store(documents, api_key):

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    #embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    embedding_function = OpenAIEmbeddings(api_key=api_key)
    # split into smaller chunks
    docs = text_splitter.split_documents(documents)
    print("docs", docs[0])
    db = Chroma.from_documents(docs, embedding_function)
    return db

""" FAISS (Facebook ai similarity search) vector store """
def FAISS_vector_store(documents, api_key):

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    #embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    embedding_function = OpenAIEmbeddings(api_key=api_key)
    # split into smaller chunks
    docs = text_splitter.split_documents(documents)
    #db = Chroma.from_documents(docs, embedding_function)

    db = FAISS.from_documents(docs, embedding_function)
    return db

""" The most basic vector store, which stores all documents in memory """
def doc_array_in_memory_store():
    return DocArrayInMemorySearch()
