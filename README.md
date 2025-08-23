# RAG_QA_System

Welcome to **RAG_QA_System** – a flexible, general-purpose Retrieval-Augmented Generation (RAG) chatbot that answers questions based on your provided files. Designed to be adaptable and scalable, this project combines powerful open-source models and modern retrieval techniques for robust, context-aware responses.

## 🚀 Overview

RAG_QA_System is engineered to create chatbots that answer user queries by first retrieving relevant content from your uploaded documents and then generating accurate, contextually informed responses. It supports a wide range of file types, including text, PDF, DOC, and images.

## ✨ Features

- **General-Purpose Chatbot:** Answers based on the files you provide (text, docs, PDFs, images)
- **Ollama Integration:** Uses ollama to pull and serve the Qwen2-7B model for generation
- **Modular & Extensible:** Easily integrate new data sources and models
- **Language Support:** English (with plans to add support for other languages)
- **Future-Proof:** Planned features include learning from handwritten and distorted texts, anti-hallucination strategies, and multi-language capabilities
- **CLI-based & Experiment-Friendly:** Simple to run, experiment, and evaluate

## 🏗️ Project Structure

```
.
├── data/               # Uploaded files and knowledge base
├── src/                # Core QA system code
├── configs/            # Model, retrieval, and system configs
├── notebooks/          # Jupyter notebooks for experiments
├── tests/              # Unit and integration tests
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## 📦 Installation

```bash
git clone https://github.com/harshit-05/RAG_QA_System.git
cd RAG_QA_System
pip install -r requirements.txt
```

## 💡 Usage

1. **Prepare your data:** Place source documents (text, PDF, DOC, images) in the `data/` folder
2. **Configure models:** Edit `configs/` to set retrieval and generation parameters (currently uses Qwen2-7B via Ollama)
3. **Run QA pipeline:** Execute main scripts in the `src/` directory or use provided notebooks

## 🧪 Testing

Run the test suite to validate your setup:
```bash
pytest tests/
```

## 🤖 Technologies Used

- Python
- Ollama (Qwen2-7B)
- LangChain
- FAISS / ElasticSearch (for retrieval)
- PyTorch / TensorFlow (for deep learning)
- Streamlit / Gradio / CLI (for UI)

## 📚 Documentation

Comprehensive documentation for setup, customization, and API usage is coming soon!

## 👫 Contributing

We welcome contributions! Please open issues or submit pull requests to help improve this project.

## 📄 License

Distributed under the MIT License.

---

> **Note:** Upcoming features include support for handwritten/distorted texts, anti-hallucination, and multilingual capabilities. Demo and advanced showcases will be added soon!