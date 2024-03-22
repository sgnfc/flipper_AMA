import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.indexes import VectorstoreIndexCreator

from langchain_openai import OpenAIEmbeddings
"""
    Retrievers return documents given an unstructured query.
    It is more general than a vector store. A retriever does not need to be able to store documents, only to return (or retrieve) them.
"""

def get_tavily_retriever():

    tavily_api_key = os.environ.get('TAVILY_API_KEY', '')
    if not tavily_api_key:
        print('You need to set the TAVILY_API_KEY environment variable to use this script.')

    tavily_retriever = TavilySearchResults()
    return tavily_retriever

def vector_store_retriever_from_loaders(loaders:list, api_key:str, vectorstore):

    vectore_store_index = VectorstoreIndexCreator(
    vectorstore_cls=vectorstore,
    embedding=OpenAIEmbeddings(api_key=api_key),
    ).from_loaders(loaders)

    return vectore_store_index.vectorstore.as_retriever()
