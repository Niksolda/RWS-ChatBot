from django.shortcuts import render
from django.http import HttpResponse
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from django.http import JsonResponse

import praw
import chromadb

reddit = praw.Reddit(
    client_id="Your client id here",
    client_secret="Your secret client here",
    password="Reddit password here",
    user_agent="user agent name here",
    username="Your reddit account name here",
)
inv = reddit.subreddit("bapcsalescanada").hot(limit=100)

persistent_client = chromadb.PersistentClient()
collection = persistent_client.get_or_create_collection("RedditInv")
for posts in inv:
    collection.add(
        documents=[posts.title], metadatas=[{"source": posts.url}], ids=[posts.id]
    )
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
langchain_chroma = Chroma(
    client=persistent_client,
    collection_name="RedditInv",
    embedding_function=embedding_function,
)

sourceList = []


## Cite sources
def process_llm_response(llm_response):
    print(llm_response["result"])
    print("\n\nSources:")
    for source in llm_response["source_documents"]:
        print(source.metadata["source"])
        sourceList.append(source.metadata["source"])


retriever = langchain_chroma.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key="Your open ai key here"),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)


def AskLLM(request):
    if request.method == "POST":
        userInput = request.POST["user_input"]
        llm_response = qa_chain(userInput)
        sourceList.clear()
        process_llm_response(llm_response)
        return JsonResponse(
            {"botResponse": llm_response["result"], "botSources": sourceList}
        )
    return render(request, "chat.html")
