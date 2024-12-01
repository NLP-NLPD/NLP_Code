import requests
from langchain.schema import Document
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
load_dotenv()

def extract_text_or_table(pdf_path="../ewha.pdf"):
    api_key = os.getenv("UPSTAGE_API_KEY")
    url = "https://api.upstage.ai/v1/document-ai/document-parse"
    headers = {"Authorization": f"Bearer {api_key}"}
    documents = []

    with open(pdf_path, "rb") as file:
        response = requests.post(url, headers=headers, files={"document": file})

    if response.status_code == 200:
        print("hello")
        data = response.json()
        html_content = data.get("content", {}).get("html", "")
        if not html_content:
            print("Error: No HTML content found in API response.")
            return []

        soup = BeautifulSoup(html_content, "html.parser")

        categories = {
            "table": "table",
            "figure": "figure",
            "chart": "img[data-category='chart']",
            "heading1": "h1",
            "header": "header",
            "footer": "footer",
            "caption": "caption",
            "paragraph": "p[data-category='paragraph']",
            "equation": "p[data-category='equation']",
            "list": "p[data-category='list']",
            "index": "p[data-category='index']",
            "footnote": "p[data-category='footnote']"
        }

        for category, selector in categories.items():
            elements = soup.select(selector)
            for element in elements:
                content = element.get_text(strip=True)
                metadata = {"category": category, "html": str(element)}
                cleaned_content = clean_extracted_text(content)
                documents.append(Document(page_content=content, metadata=metadata))

        if not documents:
            print("No sections were extracted.")
            return []
        return documents
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return []

# step 1.2 pdf parsing한 것을 cleaning text 
import re

def clean_extracted_text(text):
    # 문장 중간의 줄바꿈 제거
    cleaned_text = re.sub(r'(?<=[a-z,])\n(?=[a-z])', ' ', text)
    # 문장 끝 줄바꿈 유지
    cleaned_text = re.sub(r'(?<=[.?!])\s*\n', '\n', cleaned_text)
    
    return cleaned_text