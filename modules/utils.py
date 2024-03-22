from  langchain.schema import Document
from typing import Iterable
from bs4 import BeautifulSoup as Soup
import requests
import json

""" Scrapes the pinball.org website for all the sub-urls """
def get_sub_urls(url):
    html_content = requests.get(url).text

    soup = Soup(html_content, "html.parser")
    links = soup.find_all('a')  # Find all anchor tags
    urls = [link.get('href') for link in links if link.get('href') is not None]
    return urls


def save_docs_to_jsonl(array:Iterable[Document], file_path:str)->None:
    with open(file_path, 'w') as jsonl_file:
        for doc in array:
            jsonl_file.write(doc.json() + '\n')

def load_docs_from_jsonl(file_path)->Iterable[Document]:
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array