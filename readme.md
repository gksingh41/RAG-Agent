# 🧠 The AI Architect: Local RAG Agent

## 🚀 Product Overview
This is a retrieval-augmented generation (RAG) agent designed to answer questions about AI Architecture using a curated knowledge base.

Unlike standard LLMs which hallucinate, this agent uses a **Hybrid Local/Cloud Architecture** to ground its answers in specific technical documentation (PDFs & Text).

## 🛠 Tech Stack
*   **Orchestration:** Agno (Phidata)
*   **Frontend:** Streamlit
*   **LLM:** Google Gemini 2.5 Flash
*   **Vector Database:** LanceDB (Embedded/Local)
*   **Embeddings:** Sentence Transformers (all-MiniLM-L6-v2)

## 🏗 Architecture
1.  **Ingestion:** Local ETL pipeline chunks PDFs/Text and embeds them using a local model (384 dimensions).
2.  **Storage:** Vectors are stored in LanceDB (Serverless).
3.  **Retrieval:** Uses Semantic Search (ANN) to retrieve top-k context chunks.
4.  **Generation:** Gemini 2.5 synthesizes the answer with strict adherence to the retrieved context.

## ⚙️ How to Run Locally

1. **Clone the repo**
   ```bash
   git clone [YOUR_REPO_LINK_HERE]