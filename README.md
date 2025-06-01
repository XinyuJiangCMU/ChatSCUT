# ChatSCUT

**ChatSCUT** is a campus Q&A system designed for South China University of Technology (SCUT). It supports document upload, text-based question answering, web crawling, and some research assistant tools.

## 📁 Project Structure

```

ChatSCUT/
├── app/                          # Web interface and HTML page
│   ├── app.py
│   └── templates/index.html
│
├── agent/                        # Research assistant tools
│   ├── L2_research_write_article.ipynb
│   ├── config.json
│   └── config35.json
│
├── common/                       # Core functions and tools
│   ├── func.py                   # Main logic
│   ├── retriever.py              # Document retriever
│   ├── embedding.py              # Text embedding
│   ├── docx2json.py              # Convert DOCX to JSON
│   └── …                       # Other utility files
│
├── web_crawler/                  # Crawlers for SCUT and forums
│   ├── web_crawler_scut/
│   └── web_crawler_tieba/
│
├── chatgpt_api_server.py        # ChatGPT backend API
├── chatglm_api_server.py        # ChatGLM backend API
├── chat_stream_api.py           # Streaming response API
├── data_ingestion.py            # Load and clean data
├── docx_uploader.py             # Upload DOCX files
├── file_transfer.py             # File copy/move tool
├── glm_gpt.py                   # LLM selection logic
├── main_app.py                  # Main app logic
├── multimodal.py                # Multimodal support
├── query_api.py                 # Simple QA API
├── rag_server.py                # RAG-based Q&A service
├── utilities.ipynb              # Miscellaneous notebook
└── README.md

```

## 💡 Main Features

- **Campus Q&A**: Ask about scholarships, teachers, courses, etc.
- **Document QA**: Upload `.docx` files and ask questions about them.
- **Research Assistant**: Help with writing and article drafting.
- **Web Crawlers**: Collect public data from SCUT websites and forums.
- **LLM Support**: Work with ChatGLM or ChatGPT as backend models.