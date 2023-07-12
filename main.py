import os
import requests
import pandas as pd

from urllib.parse import quote
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

os.environ["OPENAI_API_KEY"] = ""
def call_api(url):
    headers = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response
    else:
        print("Failed to retrieve issues.")

def search_github_issue_commands(issue_no):
    url = f"https://api.github.com/repos/quartz-scheduler/quartz/issues/{issue_no}/comments"
    response_json = call_api(url).json()
    res = []
    for command in response_json:
        item = {
            "command_content": command["body"]
        }
        res.append(item)
    return res

def search_github_issues(keyword):
    url = f"https://api.github.com/repos/quartz-scheduler/quartz/issues?q={quote(keyword)}&per_page=100"
    print("url", url)
    response_json = call_api(url).json()
    res = list()
    for issue in response_json:
        item = {
            "issue_title": issue["title"] or "",
            "issue_description": issue["body"] or "",
            "issue_link": issue["html_url"],
            "issue_state": issue["state"],
            # "issue_commands": search_github_issue_commands(issue["number"])
        }
        res.append(item)
    return res


issues = search_github_issues("is:close")
print("issue count: ", len(issues))

df = pd.DataFrame(issues)
loader = DataFrameLoader(df, page_content_column="issue_description")
docs = loader.load()

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embeddings)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0.6, model_name="gpt-3.5-turbo"),
    vectorstore.as_retriever(max_tokens_limit=4097),
    memory=memory,
    verbose=True
)

q_1 = "Find issues related to service restart and summarize their root causes."
result = qa({"question": q_1})
print(result)