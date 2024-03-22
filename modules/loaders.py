from langchain_community.document_loaders import RedditPostsLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import WikipediaLoader
from .utils import get_sub_urls

""" Scrapes reddit posts from the subreddit r/pinball"""
def get_reddit_loader():
    
    reddit_loader = RedditPostsLoader(
        client_id="dBUWlJtE3mfzcruyAY1j4Q",
        client_secret="fRFhEWtFLmT6Hmhrn6dln3DpHQUdXA",
        user_agent="extractor by u/Master_Ocelot8179",
        categories=["new", "hot"],  # List of categories to load posts from
        mode="subreddit",
            search_queries=[
                "pinball",
            ],  # List of subreddits to load posts from
        number_posts=100,  # Default value is 10
    )
    return reddit_loader

def get_custom_flipper_loader():

    pinball_url = "http://www.pinball.org/rules/index-old.html"
    pinball_sub_urls = get_sub_urls(pinball_url)
    pinball_sub_urls = ['http://www.pinball.org/rules/' + url for url in pinball_sub_urls if not url.startswith('http')]

    loader = WebBaseLoader(
        pinball_sub_urls,
        continue_on_failure=True
    )
    return loader

def get_wiki_loader():
    wiki_loader = WikipediaLoader()
    return wiki_loader

