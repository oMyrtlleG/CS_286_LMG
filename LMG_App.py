"""
LEARNING MATERIAL GENERATOR FROM PDFs
Presented by Myrtlle Gem L. Oraño

Project Overview: To develop a system that takes multiple PDF documents on a specific topic as input, synthesizes their information, and generates a structured, detailed summary formatted as educational content (e.g., a study guide).

"""

import streamlit as st
from LMG_Logic import MaterialProcessor
import uuid, re

# ==============================================================================
#                                STREAMLIT APP LAYOUT
# ==============================================================================

# --- SESSION MANAGEMENT ---
def _get_session_id():
    """
    Ensures a per-run unique session_id is available in Streamlit state.
    """
    sid = st.session_state.get("session_id")
    if not sid:
        sid = str(uuid.uuid4())
        st.session_state["session_id"] = sid
    return sid

# --- MAIN FUNCTION ---
def main():
    st.set_page_config(layout="wide")
    st.title("📚 Learning Material Generator from PDFs (RAG-Powered)")
    st.markdown("Upload your study materials, provide an instructional prompt, and the app will generate a custom learning material.")
    st.markdown("---")

    # STEP 1. UPLOAD SECTION
    st.header("1. Upload and Define Scope")
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_files = st.file_uploader(
            "Select your PDF files (Word-based or Image-based)",
            type=['pdf'],
            accept_multiple_files=True
        )

    # --- USER PROMPT FOR RAG ---
    with col2:
        user_prompt = st.text_area(
            "Enter the specific 'Topic' for the learning material:",
            placeholder="E.g., Database Normalization, OSI Model, or Python Loops",
            height=150
        )
        
    st.markdown("---")
    
    session_id = _get_session_id

    # --- EXECUTION BUTTON ---
    if uploaded_files and user_prompt and st.button("Generate Learning Material", type="primary"):
        processor = MaterialProcessor()
        
        # STEP 2. EXTRACTION AND OCR 
        with st.spinner('1. Extracting text from all pages...'):
            raw_combined_text = processor._extract_text_from_pdfs(uploaded_files)
        
        if not raw_combined_text.strip():
            st.error("No text could be extracted from the uploaded files. Check if the PDFs are corrupted or truly empty.")
            return
        
        # STEP 3. CLEARING AND STRUCTURING (Chunking for RAG)
        with st.spinner('2. Cleaning text and creating RAG chunks...'):
            rag_chunks, document_id = processor._structure_and_clean_text(
                raw_text = raw_combined_text,
                session_id = session_id,
                user_prompt = user_prompt
            )
            
        if not rag_chunks:
            st.error("Text was extracted but could not be structured into RAG chunks. Aborting.")
            return

        # STEP 4. RAG INGESTION: Indexing into D: Drive Vector Store
        vector_collection, embedding_model = processor._create_and_ingest_vectors(rag_chunks, session_id, document_id)
            
        if not vector_collection or not embedding_model:
            st.error("Failed to initialize or ingest data into the vector store. Check your D: drive permissions and ChromaDB setup.")
            return
            
        # STEP 5. RAG RETRIEVAL: Finding the most relevant context
        retrieved_context = processor._retrieve_relevant_chunks(
            collection = vector_collection,
            embedding_model = embedding_model,
            query_text = user_prompt, 
            k = 7, 
            session_id = session_id,
            document_id =  document_id,
            topic = user_prompt
        )
            
        if "No context retrieved" in retrieved_context:
            st.error("Retrieval failed or the user prompt did not match any content in the PDFs.")
            return

        # STEP 6. TEXT GENERATION: Utilizing the retrieved context
        result = processor._generate_learning_material(retrieved_context)
        
        if result:
            summary_title, final_summary = result
            
            st.header("5. Final Learning Material")
            st.subheader(f"Generated Title: {summary_title}")
            st.text_area("Final Content Preview", final_summary, height=400)
            
            # STEP 7. PDF PRODUCTION
            with st.spinner('Generating final PDF file...'):
                pdf_data = processor._generate_pdf(final_summary)
                
            if pdf_data:
                try:
                    st.download_button(
                        label="⬇️ Download Learning Material PDF",
                        data=pdf_data,
                        file_name=f"{summary_title}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"An error occurred while preparing the download button: {e}")

# RUN THE APP
if __name__ == "__main__":
    main()