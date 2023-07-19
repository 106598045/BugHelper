import os

import requests
import json
import pandas as pd

from urllib.parse import quote
from langchain.document_loaders import DataFrameLoader
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

os.environ["OPENAI_API_KEY"] = ""

def call_api(url):
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response
    else:
        print("Failed to retrieve issues.")

def parse_jira_comments(json):
    res = list()
    for comment in json["comments"]:
        item = {
            "body": comment["body"],
            "author": comment["author"]["displayName"]
        }
        res.append(item)
    return res

def get_all_jira_bug_issue(startAt, maxResultCount):
    url = f"https://jira.mongodb.org/rest/api/2/search?fields=comment%2csummary%2cdescription&startAt={startAt}&maxResults={maxResultCount}&jql=project%20%3D%20%22Java%20Driver%22%20AND%20type%20%3D%20Bug%20"
    print("url", url)
    response_json = call_api(url).json()
    res = list()
    for issue in response_json["issues"]:
        item = {
            "issue_summary": issue["fields"]["summary"] or "",
            "issue_description": issue["fields"]["description"] or "",
            "issue_link": "https://jira.mongodb.org/browse/" + issue["key"],
            # "issue_state": issue["state"],
            "issue_comments": parse_jira_comments(issue["fields"]["comment"])
        }
        res.append(item)
    return res

def search_jira_issue(keyword):
    # fields => summary, comment, description
    # jql => project = Java Driver And type = Bug
    url = f"https://jira.mongodb.org/rest/api/2/search?fields=comment%2csummary%2cdescription&jql=project%20%3D%20%22Java%20Driver%22%20AND%20type%20%3D%20Bug%20AND{quote(keyword)}"
    print("url", url)
    response_json = call_api(url).json()
    res = list()
    for issue in response_json["issues"]:
        item = {
            "issue_summary": issue["fields"]["summary"] or "",
            "issue_description": issue["fields"]["description"] or "",
            "issue_link": "https://jira.mongodb.org/browse/"+issue["key"],
            # "issue_state": issue["state"],
            "issue_comments": str(parse_jira_comments(issue["fields"]["comment"]))
            # "issue_comments": parse_jira_comments(issue["fields"]["comment"])
        }
        res.append(item)
    return res


issues = search_jira_issue(' (summary ~ "NullPointerException" OR description ~ "NullPointerException") AND resolution != Duplicate')

print("issue count: ", len(issues))
print(json.dumps(issues[0:3]))

df = pd.DataFrame(issues)
loader = DataFrameLoader(df, page_content_column="issue_description")
docs = loader.load()

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embeddings)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo-16k"),
    vectorstore.as_retriever(search_kwargs={"k": 10}),
    memory=memory,
    verbose=True
)

q_1 = "Summarize issues related to NPE and show their root causes."
result = qa({"question": q_1})
print(result)
