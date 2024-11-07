import os
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Form
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import uvicorn
from collections import defaultdict
from github_fetch import fetch_from_repos, vectorize_docs, store_in_faiss
from confluence_fetch import fetch_confluence_docs

# Initialize the SentenceTransformer model for semantic search
model = SentenceTransformer('all-MiniLM-L6-v2')
# Set up FastAPI
app = FastAPI()

# Load repositories and vectorize documents
repos = {
    "https://github.com/ciec-infra/labweek.git": "test-vector-labweek",
    "https://github.com/ciec-infra/labweek-test.git": "test-vector-labweek-test"
}

# Fetch and vectorize documents from GitHub
docs = fetch_from_repos(repos)
vectors = vectorize_docs(docs)
index, doc_mapping = store_in_faiss(vectors)

# Cache for search results
cache = defaultdict(dict)

# Request body for search
class QueryRequest(BaseModel):
    query: str
    page: int = Query(1, gt=0)
    size: int = Query(3, gt=0)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Vector API!"}

@app.post("/search/")
async def search_docs(query: QueryRequest):
    if query.query in cache and query.page in cache[query.query]:
        return cache[query.query][query.page]

    try:
        # GitHub document search
        query_vector = model.encode(query.query).astype('float32')
        D, I = index.search(np.array([query_vector]), k=query.size * query.page)
        github_results = []

        for idx, i in enumerate(I[0]):
            if idx < query.size * query.page:
                if i in doc_mapping:
                    relative_path, content, repo_url = doc_mapping[i]
                    clean_repo_url = repo_url.replace(".git", "")
                    github_url = f"{clean_repo_url}/blob/main/{relative_path}"
                    snippet_start = max(content.lower().find(query.query.lower()) - 50, 0)
                    snippet_end = snippet_start + 200
                    github_results.append({
                        "title": f"{relative_path}",
                        "url": github_url,
                        "snippet": content[snippet_start:snippet_end]
                    })

        # Confluence document search
        confluence_results = fetch_confluence_docs(query.query)

        # Combine results
        combined_results = {
            "github_results": github_results,
            "confluence_results": confluence_results
        }

        cache[query.query][query.page] = combined_results
        return combined_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Handle Slack commands
@app.post("/slack_command")
async def handle_slack_command(
    token: str = Form(...),
    team_id: str = Form(...),
    team_domain: str = Form(...),
    channel_id: str = Form(...),
    channel_name: str = Form(...),
    user_id: str = Form(...),
    user_name: str = Form(...),
    command: str = Form(...),
    text: str = Form(...),
    response_url: str = Form(...)
):
    try:
        # Prepare query request for search
        query_request = {"query": text, "page": 1, "size": 3}
        response = await search_docs(QueryRequest(**query_request))

        if response["github_results"] or response["confluence_results"]:
            # Separate GitHub and Confluence results
            github_results = response["github_results"]
            confluence_results = response["confluence_results"]

            # Build response text for GitHub docs
            github_text = "*Github Docs:*\n" if github_results else ""
            for idx, result in enumerate(github_results, start=1):
                file_path = result.get('url', result.get('file_path'))
                file_name = file_path.split('/')[-1]
                snippet = result['snippet']

                # Make file name a hyperlink
                github_text += f"{idx}. *<{file_path}|{file_name}>*\n"
                github_text += f">```\n{snippet}\n```\n\n"

            # Build response text for Confluence docs
            confluence_text = "*Confluence Docs:*\n" if confluence_results else ""
            for idx, result in enumerate(confluence_results, start=1):
                url = result.get('url')
                title = result.get('title')
                snippet = result['snippet']

                # Make Confluence title a hyperlink
                confluence_text += f"{idx}. *<{url}|{title}>*\n"
                confluence_text += f">```\n{snippet}\n```\n\n"

            # Combine the GitHub and Confluence results into the response text
            response_text = github_text + confluence_text if github_results or confluence_results else "No documents found for your query."

        else:
            response_text = "There is no document related to the search."

        return {
            "response_type": "in_channel",
            "text": response_text
        }

    except Exception as e:
        return {"text": f"Error during search: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8003)
