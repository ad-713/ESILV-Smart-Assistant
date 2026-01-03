# ESILV Smart Assistant

An intelligent chatbot dedicated to ESILV, capable of answering questions about programs and admissions using RAG (Retrieval-Augmented Generation) and coordinating agents for student enrollment.

## Authors

- **Gabriel GERMAIN**
- **Alexandre HERVE**
- **Adrien GREVET**

## Features

-   **RAG System**: Answers questions based on uploaded PDF documents and crawled web content.
-   **Multi-Agent System**: Uses CrewAI to coordinate an Information Specialist and an Enrollment Assistant.
-   **Web Crawler**: Integrated [`Crawl4AI`](src/crawler.py) to scrap information directly from the ESILV website with content filtering for relevance.
-   **Lead Management**: Automatically captures and stores student lead information (name, email, interest) during chat interactions.
-   **Streamlit UI**: A tabbed interface separating the user chat experience from administrative management tools.

## Prerequisites

-   Python 3.10+
-   [Ollama](https://ollama.com/) installed and running locally.
-   [Playwright](https://playwright.dev/) dependencies (for the web crawler).

## Setup

1.  **Clone the repository** (if applicable)

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    playwright install
    ```

3.  **Setup Ollama**
    Make sure Ollama is running and pull the required model for the chat:
    ```bash
    ollama serve
    ollama pull llama3.1:8b
    ```

4.  **Embeddings Model**
    The RAG system uses `nomic-ai/nomic-embed-text-v1` from HuggingFace for generating embeddings. This model is automatically downloaded on the first run.

5.  **Run the Application**
    ```bash
    streamlit run src/app.py
    ```

## Usage

The application is divided into two main tabs:

### User Tab
-   **Chat**: Ask questions in the chat window about ESILV programs, admissions, or campus life.
-   **Enrollment**: If you express interest, the assistant will capture your details to save as a lead.

### Admin Tab
-   **Document Upload**: Upload PDF brochures or catalogs to the RAG knowledge base.
-   **Web Crawler**: Enter an ESILV URL to crawl and index web content. Includes a relevance filter to ensure high-quality data.
-   **Lead Management**: View, download (as CSV), or clear captured student leads.
-   **Maintenance**: Clear the knowledge base to reset the RAG system.

## Project Structure

-   [`src/app.py`](src/app.py): Main Streamlit application with tabbed UI.
-   [`src/agents.py`](src/agents.py): CrewAI agent definitions and enrollment orchestration logic.
-   [`src/rag.py`](src/rag.py): Logic for document processing and ChromaDB vector store operations.
-   [`src/crawler.py`](src/crawler.py): Web crawling implementation using `Crawl4AI`.
-   [`src/leads_manager.py`](src/leads_manager.py): Lead storage and retrieval logic.
-   `data/`: Stores raw files, the ChromaDB vector store, and captured leads (`leads.json`).

## User Interface

### User use case example

![User Tab](/screenshots/user-tab-exemple.png)

### Admin use case example

![Admin Tab](/screenshots/admin-tab-exemple-1.png)
![](/screenshots/admin-tab-exemple-2.png)

