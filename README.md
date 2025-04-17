# RAG-Enabled Chat Assistant

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-brightgreen)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A **Retrieval-Augmented Generation (RAG)** chatbot that allows you to upload documents, create embeddings using **SentenceTransformers**, store them in a **FAISS vector database**, and query them through a **Streamlit UI** connected to **Azure OpenAI**.

---

## Features

* **Upload PDFs or text files** and automatically chunk them into manageable text segments.
* **Generate embeddings** with SentenceTransformers and store in FAISS for fast similarity search.
* **RAG chain with Azure OpenAI** for contextual question answering.
* **Streamlit web app UI** for document upload and chatbot interaction.
* **Command-line chatbot** for quick testing.

---

## Setup Instructions

### Clone the Repository

```bash
git clone https://github.com/harshal1711/RAG-Chat-Assistant.git
cd RAG-Chat-Assistant
```

### Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure API Keys

Create a `.env` file (based on `.env.example`) and add your keys:

```env
AZURE_OPENAI_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
```

---

## Usage

### **Run Command-Line Chatbot**

```bash
python cli_chat.py
```

### **Run Streamlit Web App**

```bash
streamlit run app.py
```

Then open the provided local URL in your browser.

---

## Security

* Never commit your `.env` file (API keys).
* Use `.gitignore` to exclude environment-specific files.