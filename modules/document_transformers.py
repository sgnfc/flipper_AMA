from langchain_community.document_transformers.openai_functions import (
    create_metadata_tagger,
)
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
"""
"multiball": {"type": "string", 
                      "description": "A 20 word summary of the steps involved to get a multi-ball in the game"},
"""
flipper_schema = {
    "properties": {
        "pinball_title": {"type": "string",
                          "description": "The title of the pinball game"},
        "possible_shots" : {"type": "string",
                   "description": "A list of all possible shots in the game"},
        "manufacturer": {"type": "string",
                     "description": "The manufacturer of the pinball machine"},
    },
    "required": ["pinball_title", "multiball"],
}

def add_metadata_to_documents(documents, schema=flipper_schema):
    #gpt-4-turbo-preview	
    #gpt-3.5-turbo-0125
    document_transformer = create_metadata_tagger(metadata_schema=schema, llm=ChatOpenAI(temperature=0, model="gpt-4-turbo-preview"))

    enhanced_documents = document_transformer.transform_documents(documents, verbose=True)

    return enhanced_documents

