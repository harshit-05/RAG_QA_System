# RAG_QA_System

Welcome to **RAG_QA_System** – a flexible, general-purpose Retrieval-Augmented Generation (RAG) chatbot that answers questions based on your provided files. This project is built to be adaptable and scalable, leveraging modern retrieval techniques and powerful open-source models for robust, context-aware responses.

---

## 🚀 Project Goal

RAG_QA_System is designed to create chatbots that give accurate answers based on the files you provide. It currently supports text, PDF, DOCX, and images as knowledge sources. The system retrieves relevant information from your uploaded documents and uses the Qwen2-7B model (served via Ollama) to generate contextually rich responses.

---

## ✨ Features

- **General-Purpose Chatbot:** Answers are grounded in your uploaded files (text, DOCX, PDFs, images)
- **Ollama & Qwen2-7B Integration:** Uses open-source large language models for generation
- **Modular & Extensible:** Easily add new data sources or swap models
- **Strong Retrieval:** FAISS-based vector store for efficient semantic search
- **Language Support:** Currently supports English (more languages planned)
- **CLI & API Ready:** Designed for easy experimentation and integration
- **Future-Proof:** Planned features include learning from handwritten and distorted texts, anti-hallucination, and multilingual capabilities

---

## 🏗️ Project Structure

```
.
├── docs/                         # Documentation and reports
├── v1/                           # First version of the QA pipeline
│   ├── config.py                 # Configuration settings
│   ├── docx_processor.py         # DOCX file processor
│   ├── main.py                   # Main QA pipeline entry
│   └── requirements.txt          # Python dependencies
├── v2/                           # Second version, modular pipeline
│   ├── config.yaml               # YAML configuration for v2
│   ├── evaluate.py               # Evaluation scripts and metrics
│   ├── file_processor.py         # Unified file ingestion logic
│   ├── main2.py                  # Main script for v2 pipeline
│   └── pipeline_builder.py       # Orchestration and pipeline management
├── vectorstore/db_faiss/         # FAISS vector store for document embeddings
│   ├── index.faiss
│   └── index.pkl
├── README.md                     # This file
├── Screencast from ...           # Demo screencasts (can be updated with newer demo)
└── __pycache__/                  # Python cache files
```

---

## 📦 Installation

```bash
git clone https://github.com/harshit-05/RAG_QA_System.git
cd RAG_QA_System
pip install -r v1/requirements.txt
# or for v2, ensure dependencies in v2/config.yaml are installed
```

---

## 💡 Usage

1. **Prepare your data:** Put your source files (text, DOCX, PDFs, images) in an accessible directory.
2. **Configure settings:** Edit `v1/config.py` or `v2/config.yaml` for model and retrieval parameters.
3. **Run QA pipeline:**  
   - For v1: `python v1/main.py`
   - For v2: `python v2/main2.py`
4. **Experiment and evaluate:** Use scripts in `v2/evaluate.py` to measure performance.

---

## 🧪 Testing

Run tests (if implemented):

```bash
pytest v1/
pytest v2/
```

---

## 🤖 Technologies Used

- Python
- Ollama (Qwen2-7B)
- LangChain
- FAISS (vector database)
- PyTorch / TensorFlow
- Streamlit / Gradio / CLI
- DOCX/PDF/Image file support

---

## Planned & Upcoming Features

- Learning from handwritten and distorted texts
- Anti-hallucination strategies for factual accuracy
- Multilingual support (currently English only)
- Enhanced document/image processing
- Public demo (coming soon)

---

## 👫 Contributing

We welcome contributions! Please open issues or submit pull requests to help improve this project.

---

## 📄 License

Distributed under the MIT License.

---

> **Note:** This README will be updated as new features and demos are added.  
> For questions or collaboration, open a GitHub issue.
