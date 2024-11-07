import os
import base64
import requests
from fastapi import HTTPException
from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize

# Confluence credentials and base URL
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL")
CONFLUENCE_BASE_URL_PAGE = os.getenv("CONFLUENCE_BASE_URL_PAGE")
CONFLUENCE_USERCODE = os.getenv("CONFLUENCE_USERCODE")

# Function to extract keywords, filtering out common filler words
def extract_keywords(sentence):
    pattern = r'\b(?:what|is|why|the|a|an|of|and|for|to|in|on|at|by|with|as|from)\b|[^\w\s]'
    keywords = re.sub(pattern, '', sentence, flags=re.IGNORECASE).split()
    return [word for word in keywords if word]

# Function to fetch documents from Confluence
def fetch_confluence_docs(query):
    keywords = extract_keywords(query)
    if keywords:
        keyword_query = " OR ".join([f'text ~ "{word}"' for word in keywords])
    else:
        return {"message": "No valid keywords found in query."}

    auth_string = f"{CONFLUENCE_USERNAME}:{CONFLUENCE_API_TOKEN}"
    encoded_auth_string = base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')
    
    headers = {
        "Authorization": f"Basic {encoded_auth_string}",
        "Content-Type": "application/json"
    }
    
    params = {
        "cql": keyword_query,
        "limit": 10
    }

    try:
        response = requests.get(f"{CONFLUENCE_BASE_URL}/content/search", headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        results = []
        for result in data.get("results", []):
            page_id = result["id"]
            title = result["title"]
            url = f"{CONFLUENCE_BASE_URL_PAGE}/spaces/~{CONFLUENCE_USERCODE}/pages/{page_id}"
            excerpt = result.get("excerpt", "")

            # If excerpt is empty, fetch the page content for a snippet
            if not excerpt:
                page_content_response = requests.get(
                    f"{CONFLUENCE_BASE_URL}/content/{page_id}?expand=body.view", headers=headers
                )
                if page_content_response.status_code == 200:
                    page_content_html = page_content_response.json().get("body", {}).get("view", {}).get("value", "")
                    if page_content_html:
                        soup = BeautifulSoup(page_content_html, "html.parser")
                        page_content_text = soup.get_text()
                        excerpt = page_content_text[:200].strip() + "..."  # Adjust the length as needed

            results.append({
                "title": title,
                "url": url,
                "snippet": excerpt
            })

        return results
    except requests.exceptions.RequestException as e:
        print(f"Error fetching from Confluence: {e}")
        return []
