###1)
### Extract text from PDF
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import requests
import torch
import numpy as np
import PyPDF2
import re
import pdfplumber
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import faiss
from openai import AzureOpenAI
from dotenv import load_dotenv

def is_chinese(query):
    return any('\u4e00' <= char <= '\u9fff' for char in query)

def is_korean(query):
    return any('\uac00' <= char <= '\ud7af' for char in query)

### Predict mode function
def predict_mode(query):
    if is_chinese(query):
        return "Summarization Mode"
    elif is_korean(query):
        return "Summarization Mode"
    else:
        return "Retrieval Mode"

load_dotenv()

endpoint = os.getenv("ENDPOINT_URL")
deployment = os.getenv("DEPLOYMENT_NAME")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version="2025-01-01-preview"
)

from pathlib import Path

def find_pdf_on_desktop(filename="EJ333.pdf"):
    candidate_dirs = [
        Path.home() / "Desktop",
        Path.home() / "OneDrive" / "Desktop",
        Path.home() / "OneDrive" / "桌面",
        Path.home() / "桌面"
    ]
    for base_dir in candidate_dirs:
        for path in base_dir.rglob(filename):
            return str(path)
    raise FileNotFoundError(f"Could not locate {filename} on known desktop paths or subfolders.")
###

class PDFParser:
    def extract_text_from_pdf(self, pdf_path):
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text

class AdvancedPDFParser:
    def extract_text_from_pdf(self, pdf_path):
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

class QueryUnderstanding:
    def __init__(self, client, deployment):
        self.client = client
        self.deployment = deployment

    def extract_keywords_and_language(self, query):
        prompt = f"""
You are a multilingual NLP assistant.

Given this user query:
"{query}"

1. Extract the main search keywords (1~5 words).
2. Detect the language and provide its ISO 639-1 code (e.g., "en", "zh", "ko").

Respond only in the following JSON format:
{{
  "keywords": ["keyword1", "keyword2"],
  "language": "en"
}}"""

        messages = [
            {"role": "system", "content": "You extract structured metadata from user queries."},
            {"role": "user", "content": prompt}
        ]

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            max_tokens=300
        )

        import json
        try:
            return json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            print("Error parsing JSON:", e)
            return {"keywords": [], "language": "unknown"}
###main 1
def main_1():
    parser = PDFParser()
    pdf_path = find_pdf_on_desktop("EJ333.pdf")
    text = parser.extract_text_from_pdf(pdf_path)
    print(text)
###

###2)
### main 2 Embed sample sentence
def main_2():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    text = "This is a test sentence."
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
    print("Embedding shape:", embedding.shape)
###

###3)
### Create FAISS index and test search logic
def encode_text_to_vector(text):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def create_faiss_index(vectors):
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors, dtype=np.float32))
    return index

### main 3
def main_3():
    samples = ["This is the first item.", "This is the second one.", "Another unrelated sentence."]
    vectors = np.vstack([encode_text_to_vector(p) for p in samples])
    index = create_faiss_index(vectors)

    query = "first item"
    query_vec = encode_text_to_vector(query)
    _, top_indices = index.search(query_vec, k=2)
    print("Top 2 neighbors:", top_indices)

###

###4）
### Full pipeline - PDF parse + vector search + LLM answer
class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            cache_folder="./models",
            use_auth_token=False
        )

    def get_embedding(self, text):
        return self.model.encode(text, convert_to_numpy=True)

class VectorDatabase:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.paragraphs = []

    def add(self, vectors):
        self.index.add(vectors)

    def search(self, query_vector, top_k=3):
        query_vector = np.array(query_vector).reshape(1, -1)
        distances, indices = self.index.search(query_vector, top_k)
        return distances, indices

import requests
def generate_answer_with_azure(client, deployment, context, question):
    messages = [
        {"role": "system", "content": "You are an academic writing assistant."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    completion = client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_tokens=800,
        temperature=0.7,
        stream=False
    )

    return completion.choices[0].message.content.strip()


import re
def split_into_paragraphs(text):
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    paragraphs = []
    temp = []
    for i, sentence in enumerate(sentences):
        temp.append(sentence)
        if len(temp) >= 3:
            paragraphs.append(" ".join(temp))
            temp = []
    if temp:
        paragraphs.append(" ".join(temp))
    return paragraphs
###

import os
api_key = os.getenv("OPENROUTER_API_KEY")

def is_summarize_query(query):
    summarize_keywords = ["summarize", "overview", "key points", "main points", "summary", "overall", "conclusion"]
    normalized_query = query.strip().lower()
    return any(keyword in normalized_query for keyword in summarize_keywords)

def predict_mode(query):
    """
    Predict whether the query should trigger Summarization Mode or Retrieval Mode
    """
    if is_summarize_query(query):
        return "Summarization Mode"
    else:
        return "Retrieval Mode"

### main 4
def main_4():
    import sys
    from pathlib import Path
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import json
    import requests

    ### Select parser type from command line or default to advanced
    parser_choice = "advanced"
    if len(sys.argv) > 1:
        if sys.argv[1] in ["advanced", "default"]:
            parser_choice = sys.argv[1]
        else:
            print("Invalid parser option. Defaulting to 'advanced'.")

    ### Initialize PDF parser
    if parser_choice == "advanced":
        parser = AdvancedPDFParser()
        print("[Parser] Using AdvancedPDFParser")
    else:
        parser = PDFParser()
        print("[Parser] Using PDFParser")

    all_paragraphs = []
    paragraph_source_info = []

    ### Ask user to choose single PDF or folder mode
    mode = input("Select mode: (1) Single PDF Q&A, (2) Folder Q&A. Enter 1 or 2: ").strip()

    if mode == "1":
        pdf_path = input("Enter full PDF file path: ").strip()
        pdf_file = Path(pdf_path)
        if not pdf_file.exists() or not pdf_file.is_file():
            print(f"File '{pdf_path}' not found.")
            return

        print(f"Processing single PDF: {pdf_file.name}")
        try:
            text = parser.extract_text_from_pdf(str(pdf_file))
            if not text.strip():
                print(f"No text extracted from {pdf_file.name}")
                return
            paragraphs = split_into_paragraphs(text)
            all_paragraphs = paragraphs
            paragraph_source_info = [{"pdf_file": pdf_file.name, "para_id": i} for i in range(len(paragraphs))]
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")
            return

    elif mode == "2":
        folder_path = input("Enter folder path with PDFs: ").strip()
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            print(f"Folder '{folder_path}' invalid.")
            return

        pdf_files = list(folder.glob("*.pdf"))
        if not pdf_files:
            print("No PDFs found in folder.")
            return

        print(f"Found {len(pdf_files)} PDFs, processing...")

        def process_pdf(pdf_file):
            try:
                text = parser.extract_text_from_pdf(str(pdf_file))
                if not text.strip():
                    print(f"No text extracted from {pdf_file.name}")
                    return []
                paras = split_into_paragraphs(text)
                return [(pdf_file.name, p) for p in paras]
            except Exception as e:
                print(f"Error processing {pdf_file.name}: {e}")
                return []

        ### Concurrently process PDFs
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_pdf, pdf) for pdf in pdf_files]
            for future in as_completed(futures):
                for pdf_name, para in future.result():
                    paragraph_source_info.append({"pdf_file": pdf_name, "para_id": len(all_paragraphs)})
                    all_paragraphs.append(para)

    else:
        print("Invalid mode selected.")
        return

    if not all_paragraphs:
        print("No content extracted from PDFs.")
        return

    embedder = Embedder()

    ### Embed paragraphs in batches
    batch_size = 32
    para_vecs_list = []
    for i in range(0, len(all_paragraphs), batch_size):
        batch = all_paragraphs[i:i+batch_size]
        batch_vecs = np.vstack([embedder.get_embedding(p) for p in batch])
        para_vecs_list.append(batch_vecs)
    para_vecs = np.vstack(para_vecs_list)

    ### Create vector DB and add embeddings
    db = VectorDatabase(para_vecs.shape[1])
    db.paragraphs = all_paragraphs
    db.add(para_vecs)

    query_understanding = QueryUnderstanding(client, deployment)

    MAX_CALLS = 100
    MAX_TOKENS = 80000
    call_count = 0
    total_tokens_used = 0

    while True:
        try:
            query = input("\nEnter your question (or 'exit' to quit): ").strip()
            if not query:
                print("Please enter a question.")
                continue
            if query.lower() == "exit":
                print("Exiting.")
                break

            call_count += 1
            if call_count > MAX_CALLS:
                print("Reached max question limit.")
                break

            ### Extract keywords and language from query
            metadata = query_understanding.extract_keywords_and_language(query)
            print(f"\n[Metadata] Keywords: {metadata.get('keywords', [])}")
            print(f"[Metadata] Language: {metadata.get('language', 'unknown')}")

            ### Embed query and search top paragraphs
            query_vec = embedder.get_embedding(query)
            _, indices = db.search(query_vec, top_k=5)
            top_paras = [db.paragraphs[idx] for idx in indices[0] if idx != -1]

            ### Display top paragraphs and sources
            print("\nTop 5 matched paragraphs:")
            for rank, idx in enumerate(indices[0]):
                if idx == -1: continue
                print(f"Rank {rank+1} - File: {paragraph_source_info[idx]['pdf_file']}\nParagraph: {db.paragraphs[idx]}\n{'-'*60}")

            ### Prepare context for LLM
            context = "\n\n".join(top_paras)[:1600]

            try:
                answer = generate_answer_with_azure(client, deployment, context, query)
            except Exception as e:
                print(f"Error generating answer: {e}")
                answer = "Sorry, unable to generate answer."

            if total_tokens_used > MAX_TOKENS:
                print("Reached max token usage. Exiting.")
                break

            print(f"\nLLM Answer:\n{answer}\n")

        except KeyboardInterrupt:
            print("User exited.")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            continue
###

### Run all parts
if __name__ == "__main__":
    # main_1()
    # main_2()
    # main_3()
    main_4()


