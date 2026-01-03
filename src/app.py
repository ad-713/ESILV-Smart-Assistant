import streamlit as st
import os
import time
import asyncio
import sys
from rag import load_documents, split_documents, add_to_chroma, clear_database
from agents import run_crew
from crawler import crawl_esilv
from leads_manager import get_leads_dataframe, clear_leads

# Fix for Windows asyncio loop with Playwright
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

st.set_page_config(page_title="ESILV Smart Assistant", page_icon="🎓")

st.title("🎓 ESILV Smart Assistant")

# Create Tabs
tab_user, tab_admin = st.tabs(["User", "Admin"])

# --- USER TAB (Chatbot) ---
with tab_user:
    st.header("💬 Chat Assistant")
    
    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Ask a question about ESILV..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = run_crew(prompt)
                    st.markdown(response)
                    # Add assistant response to history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.info("Make sure Ollama is running (`ollama serve`) and you have pulled the model (`ollama pull llama3.1:8b`). Also ensure `qwen3-embedding:0.6b` is pulled for embeddings.")

# --- ADMIN TAB (Knowledge Base & Crawler) ---
with tab_admin:
    st.header("📚 Knowledge Base Management")
    
    # Document Upload Section
    st.subheader("📄 Document Upload")
    uploaded_files = st.file_uploader("Upload PDF Documents", accept_multiple_files=True, type=['pdf'])
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                # Save uploaded files temporarily
                temp_dir = "data/temp"
                os.makedirs(temp_dir, exist_ok=True)
                file_paths = []
                
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)
                
                # RAG Pipeline
                documents = load_documents(file_paths)
                chunks = split_documents(documents)
                add_to_chroma(chunks)
                
                st.success(f"Successfully processed {len(file_paths)} files!")
                
                # Cleanup
                # for file_path in file_paths:
                #     os.remove(file_path)
        else:
            st.warning("Please upload files first.")

    st.divider()

    # Web Crawler Section
    st.subheader("🌐 Web Crawler")
    start_url = st.text_input("Start URL", value="https://www.esilv.fr/en/")
    max_depth = st.slider("Max Depth", min_value=1, max_value=3, value=1)
    
    if st.button("Crawl Website"):
        with st.spinner(f"Crawling {start_url} (Depth: {max_depth})..."):
            try:
                # Run the async crawler
                # Streamlit runs in a loop, so we need to be careful.
                # crawl4ai uses Playwright which requires the main loop on Windows sometimes or Proactor.
                
                documents = asyncio.run(crawl_esilv(start_url, max_depth))
                
                if documents:
                    st.info(f"Crawled {len(documents)} pages. Processing...")
                    chunks = split_documents(documents)
                    add_to_chroma(chunks)
                    st.success(f"Successfully added content from {len(documents)} pages!")
                else:
                    st.warning("No content found to crawl.")
                    
            except Exception as e:
                st.error(f"Crawling failed: {e}")
                
    st.divider()
    
    # Maintenance Section
    st.subheader("🛠️ Maintenance")
    if st.button("Clear Knowledge Base"):
        clear_database()
        st.success("Database cleared!")

    st.divider()

    # Leads Section
    st.subheader("📋 Interested Users (Leads)")
    
    # Reload leads
    leads_df = get_leads_dataframe()
    
    if not leads_df.empty:
        st.dataframe(leads_df, use_container_width=True)
        
        # CSV Download
        csv = leads_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Leads as CSV",
            data=csv,
            file_name='esilv_leads.csv',
            mime='text/csv',
        )
        
        if st.button("🗑️ Clear Leads Data", type="primary"):
            if clear_leads():
                st.success("Leads data cleared successfully!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Failed to clear leads data.")
    else:
        st.info("No leads captured yet.")
