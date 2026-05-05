import os
import time

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

pdfs_file = "data/" # path to PDFs folder
urls_file = "urls.txt" # txt file with one URL per line

def clean_text(text):
    return " ".join(text.split())
    
def load_pdfs(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    docs=loader.load()
    print("Length of pdf pages-", len(docs))
    return docs

def load_urls(urls_file):
    with open(urls_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]
    docs = []

    for url in urls:
        try:
            loader = WebBaseLoader(
                web_paths=[url],
                requests_kwargs={
                    "headers": {"User-Agent": "Mozilla/5.0"},
                    "timeout": 20
                }
            )
            data = loader.load()
            if data:
                docs.extend(data)
                print(f"Loaded-{url}")
            else:
                print(f"No data-{url}")

            time.sleep(2)  # avoid blocking

        except Exception as e:
            print(f"Skipped: {url}")

    print("Total pages loaded-", len(docs))
    return docs

# Initialize RecursiveCharacterTextSplitter
def create_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,      
        chunk_overlap=80
    )
    return splitter.split_documents(docs)

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
pdf_docs = load_pdfs(pdfs_file)
web_docs = load_urls(urls_file)

# Add metadata
for d in pdf_docs:
    d.metadata["source_type"] = "pdf"

for d in web_docs:
    d.metadata["source_type"] = "web"

all_docs = pdf_docs + web_docs

# Clean + filter empty
all_docs = [d for d in all_docs if d.page_content.strip()]

# Clean text Only
for d in all_docs:
    d.page_content = clean_text(d.page_content)

#Deduplicate (on clean text)
unique = set()
filtered = []

for d in all_docs:
    key = d.page_content[:300]   # partial match
    if key not in unique:
        unique.add(key)
        filtered.append(d)

all_docs = filtered

#Add source After deduplication
for d in all_docs:
    source = d.metadata.get("source", "")
    d.page_content = f"Source: {source}\n{d.page_content}"

#Now chunking
chunks = create_chunks(all_docs)
#remove weak chunks
chunks = [c for c in chunks if len(c.page_content) > 80]
print("Total chunks-", len(chunks))

embedding_model = get_embedding_model()

import shutil
DB_FAISS_PATH = "vectorstore/db_faiss"

if os.path.exists(DB_FAISS_PATH):
    shutil.rmtree(DB_FAISS_PATH)

os.makedirs(DB_FAISS_PATH, exist_ok=True)
db = FAISS.from_documents(
    chunks,
    embedding_model,
    normalize_L2=True
)

db.save_local(DB_FAISS_PATH)
print("FAISS index saved successfully")


