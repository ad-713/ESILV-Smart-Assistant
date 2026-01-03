# ESILV Smart Assistant

An intelligent chatbot dedicated to ESILV, capable of answering questions about programs and admissions using RAG (Retrieval-Augmented Generation) and coordinating agents for student enrollment.

## Features
-   **RAG System**: Answers questions based on uploaded PDF documents (e.g., brochures, course catalogs).
-   **Multi-Agent System**: Uses CrewAI to coordinate an Information Specialist and an Enrollment Assistant.
-   **Streamlit UI**: A friendly chat interface to interact with the bot and upload documents.

## Prerequisites
-   Python 3.10+
-   [Ollama](https://ollama.com/) installed and running locally.

## Setup

1.  **Clone the repository** (if applicable)

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Ollama**
    Make sure Ollama is running and pull the `llama3.1:8b` model (for chat) and `qwen3-embedding:0.6b` (for embeddings):
    ```bash
    ollama serve
    ollama pull llama3.1:8b
    ollama pull qwen3-embedding:0.6b
    ```

4.  **Run the Application**
    ```bash
    streamlit run src/app.py
    ```

## Usage
1.  **Upload Documents**: Use the sidebar to upload PDF documents (e.g., "ESILV Brochure.pdf"). Click "Process Documents" to build the knowledge base.
2.  **Chat**: Ask questions in the main chat window.
    *   *Example*: "What are the majors available at ESILV?"
    *   *Example*: "How do I apply for the engineering program?"
3.  **Enrollment**: If you express interest (e.g., "I want to apply"), the bot may ask for your details.

## Project Structure
-   `src/app.py`: Main Streamlit application.
-   `src/agents.py`: CrewAI agent definitions and orchestration.
-   `src/rag.py`: Logic for loading PDFs, chunking, and ChromaDB operations.
-   `data/`: Stores raw files and the ChromaDB vector store.
