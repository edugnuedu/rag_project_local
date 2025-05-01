
Repo : https://github.com/edugnuedu/rag_project_local

RAG Application for Press Release Analysis

Overview

This project implements a Retrieval-Augmented Generation (RAG) application designed to process and query press release documents. It uses Weaviate (a vector database) to store and retrieve document embeddings, Sentence Transformers to generate embeddings, OpenAI's GPT-3.5-turbo for answer generation, and FastAPI to provide a web interface for querying. The application is built to run locally, connecting to a Weaviate Cloud cluster for vector storage and retrieval.
The project ingests text files from a data directory, embeds them using a pre-trained Sentence Transformer model (all-MiniLM-L6-v2), stores them in Weaviate, and allows users to query the documents via a web interface. 
Retrieved documents are used as context for GPT-3.5-turbo to generate accurate, contextually relevant answers.

rag_project_local/
├── main.py                # FastAPI app for the web interface
├── ingestion.py           # Script to ingest and embed documents into Weaviate
├── retrieval.py           # Script to retrieve documents from Weaviate
├── generation.py          # Script to generate answers using OpenAI API
├── templates/
│   └── index.html         # HTML template for the web interface
├── data/
│   ├── 0.txt             # Sample press release
│   ├── 1.txt             # Sample press release
│   └── ...               # Additional press release files
├── tests/
│   └── test_rag.py       # Pytest tests for ingestion and retrieval
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (API keys, paths)
├── .gitignore            # Git ignore file
└── README.md             # This file

Components

Ingestion (ingestion.py): Reads text files from the data/ directory, generates embeddings using sentence-transformers, and stores them in a Weaviate Cloud cluster with a Document schema.
Retrieval (retrieval.py): Encodes user queries into embeddings and retrieves the top-k most similar documents from Weaviate using vector search.
Generation (generation.py): Uses OpenAI's GPT-3.5-turbo to generate answers based on retrieved documents.
Web Interface (main.py, templates/index.html): A FastAPI application with a simple HTML form to submit queries and display answers.
Weaviate Cloud: A sandbox cluster stores document embeddings and supports vector search.
Environment Variables (.env): Stores sensitive information like API keys and configuration paths.

Prerequisites:

Python: Installed and added to PATH (download).
Weaviate Cloud Account and API Key:
Step 1: Sign up or log in to Weaviate Cloud Console.
Step 2: Create a sandbox cluster (if not already created)
Navigate to the dashboard and click Create Cluster.
Name: rag-test (or your choice).
Type: Sandbox (free for 14 days).
Region: Europe (e.g., europe-west3).
Confirm creation.

Step 3: Obtain the Weaviate API Key:
In the Weaviate Cloud Console, go to your cluster’s Details or API Keys section.
Copy the API key (a long alphanumeric string).


OpenAI API Key:
Step 1: Sign up or log in to OpenAI Platform.
Step 2: Create an API key:
Navigate to API Keys in the dashboard.
Click Create New Key.
Copy the key (starts with sk-).

Data Files:
Place press release text files in data/ (e.g., 0.txt, 1.txt).
Sample files:
0.txt: Deutsche Telekom launches 5G campus network for RTL Deutschland to support ultra-low latency broadcasting.
1.txt: Deutsche Telekom partners with Siemens to deploy IoT solutions for smart cities using NB-IoT technology.

Setup Instructions:

Follow these steps to set up and run the project locally:

1. Clone 
https://github.com/edugnuedu/rag_project_local


git clone https://github.com/your-username/rag-project.git
cd rag_project

Or Set Up the Project

If not already set up, create the directory structure and add the necessary files (see Project Structure).

2. Create and Activate Virtual Environment
cd D:\rag_project_local
python -m venv venv
.\venv\Scripts\activate


Prompt should show (venv).

3. Install Dependencies

Ensure requirements.txt contains:
fastapi==0.115.4
uvicorn==0.32.0
jinja2==3.1.4
sentence-transformers==3.0.1
weaviate-client==4.8.1
openai==0.28.1
huggingface_hub==0.23.4
python-dotenv==1.0.1
python-multipart==0.0.20
pytest==8.3.3

Install dependencies:
pip install -r requirements.txt


4. Configure Environment Variables

Create or update D:\rag_project_local\.env:

OPENAI_API_KEY=your_openai_api_key
WEAVIATE_URL=your_weaviate_url
WEAVIATE_API_KEY=your_weaviate_api_key
DATA_DIR= your_project_path\data 

Replace your_openai_api_key with the key from OpenAI (starts with sk-).
Replace your_weaviate_api_key with the key from Weaviate Cloud Console.
Replace your_weaviate_url with the URL from Weaviate Cloud Console.
Replace your_project_path with the path for the repo.

4. Running the Application

Ingest Documents:
python ingestion.py

Expected:
Found X files to ingest.
Ingestion completed successfully.

This reads text files from data/, generates embeddings, and stores them in Weaviate.

Start the Web App:
uvicorn main:app --host 0.0.0.0 --port 8000
or 
python main.py

Expected:
Uvicorn running on http://0.0.0.0:8000

Access at http://localhost:8000.

Use the Web Interface:

Open http://localhost:8000 in a browser.
Enter a question (e.g., “What is the purpose of the 5G campus network for RTL Deutschland?”).
Expected: Displays the question and a relevant answer from GPT-3.5-turbo.


5.Automated Tests
pytest tests/test_rag.py -v

Tests:

Tests:

test_weaviate_connection: Verifies the Weaviate API key and connection to the cluster.
test_openai_connection: Verifies the OpenAI API key and a simple API call.
test_ingestion: Verifies documents are ingested into Weaviate with correct properties and vectors.
test_retrieval: Checks that querying returns relevant documents.


Notes

Weaviate Cloud: The sandbox cluster (created 4/30/2025) expires after 14 days. Create a new sandbox or upgrade if needed.
API Key Security: Never share OPENAI_API_KEY or WEAVIATE_API_KEY. Keep them in .env.
Performance: Weaviate Cloud handles large datasets efficiently, but ingestion time depends on file count and network speed.

