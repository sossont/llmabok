import bs4
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://m.sports.naver.com/wfootball/article/076/0004354285",
                      bs_kwargs=dict(parse_only=bs4.SoupStrainer("article")))
docs = loader.load()
print(docs[0].page_content)