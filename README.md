# PDF AI Companion

A lightweight AI-powered companion that answers questions about PDF documents using Azure OpenAI GPT-4.5.  
It supports both `PyPDF2` and `pdfplumber` for flexible text extraction, and uses FAISS for semantic search and retrieval.  
Users can interactively select the parser and target PDF at runtime, with support for summarization-style queries and multilingual input (English, Chinese, Korean).

---

### Features

- Dual PDF parser support (`PyPDF2` and `pdfplumber`)  
- CLI-based dynamic PDF and parser selection  
- Multilingual query handling (EN / 中文 / 한국어)  
- Summarization mode for long-document questions  
- FAISS-based vector similarity search  
- Azure OpenAI GPT-4.5 integration  

---

### Getting Started

```bash
# 1. Clone the repository:
git clone https://github.com/MIN-R78/Google-Jun.git
cd Google-Jun

# 2. Install dependencies:
pip install -r requirements.txt

# 3. Create a `.env` file based on `.env.example` and add your Azure OpenAI API credentials:
# (edit the file manually, or use echo commands here)
echo "AZURE_OPENAI_API_KEY=your_api_key_here" >> 
echo "AZURE_OPENAI_ENDPOINT=your_endpoint_here" >> 
echo "AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name_here" >> 

# 4. Run the app (choose parser mode: 'advanced' or 'default'):
python Google-4.py advanced  
