import streamlit as st
import fitz
from PIL import Image
import io
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import re
import time
import uuid

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

from assets.Extract_Mod import _reconstruct_text_from_blocks, _preprocess_image_for_ocr
from assets.Restructure_Mod import _clean_raw_text, _remove_boilerplate, _group_by_extracted_tags
from assets.APICall_Mod import _call_gemini_api
from assets.Section_Mod import _dissect_learning_material
from assets.Wrap_Mod import check_formatting


class MaterialProcessor:
    # --- TEXT EXTRACTION ---
    def _extract_text_from_pdfs(self, uploaded_files):
        """
        Extracts text from multiple PDFs, prioritizing layout and structure.
        Applies OCR with preprocessing for image-based pages (backup).
        """
        combined_text = ""
        total_files = len(uploaded_files)
        st.info(f"Processing {total_files} PDF file(s)... This may take a moment.")

        progress_bar = st.progress(0)

        for i, file in enumerate(uploaded_files):
            current_text = []
            try:
                pdf_document = fitz.open(stream=file.read(), filetype="pdf")

                for page_num in range(len(pdf_document)):
                    page = pdf_document.load_page(page_num)

                    # ATTEMPT 1: Extract structured text
                    page_dict = page.get_text("dict")
                    extracted_text = _reconstruct_text_from_blocks(page_dict)

                    if extracted_text.strip():
                        current_text.append(extracted_text)

                        # Placeholder for detecting and describing Tables/Figures
                        if any(kw in extracted_text.lower() for kw in ["table", "figure", "entity", "attribute"]):
                            current_text.append("\n[-- TABLE/FIGURE DETECTED: Placeholder for structured data extraction --]\n")

                    else:
                        # ATTEMPT 2: OCR for image-based pages
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        img = Image.open(io.BytesIO(pix.tobytes("ppm")))

                        img_preprocessed = _preprocess_image_for_ocr(img)
                        ocr_text = pytesseract.image_to_string(img_preprocessed)

                        # Post-OCR cleaning
                        ocr_text = re.sub(r'[^\x00-\x7F]+', ' ', ocr_text).strip()
                        ocr_text = re.sub(r'\s+', ' ', ocr_text).strip()

                        if ocr_text:
                            current_text.append(f"\n[-- OCR TEXT, Page {page_num + 1} --]\n{ocr_text}\n")

                combined_text += f"\n--- Start of Document: {file.name} ---\n"
                combined_text += "\n".join(current_text)
                combined_text += f"\n--- End of Document: {file.name} ---\n\n"

            except Exception as e:
                if "tesseract is not installed" in str(e):
                    st.error("Tesseract OCR is not found. Please install the Tesseract engine on your system.")
                else:
                    st.warning(f"Error processing {file.name}. Skipping. Error: {e}")

            progress_bar.progress((i + 1) / total_files)

        progress_bar.empty()
        return combined_text

    # --- CLEANING AND STRUCTURING METHOD ---
    def _structure_and_clean_text(self, raw_text, max_chunk_chars=1200, session_id=None, user_prompt=None):
        """
        Cleans and structures raw text into chunks.
        Each chunk is tagged with session_id, document_id, and topic (user_prompt).
        """
        st.subheader("2. Structuring and Cleaning Data")

        # Create a unique document_id using topic + timestamp + UUID
        document_id = f"{user_prompt}_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        # STEP 1: Low-level character and whitespace cleaning
        clean_text = _clean_raw_text(raw_text)

        # STEP 2: Remove high-level noise like page numbers
        filtered_text = _remove_boilerplate(clean_text)

        # STEP 3: Group content based on the PyMuPDF-detected structural tags
        grouped_content = _group_by_extracted_tags(filtered_text)

        # STEP 4: Convert structured sections → text chunks
        rag_chunks = []
        chunk_id = 0

        for section in grouped_content:
            header = section.get("header", "").strip()
            header_md = f"## {header}" if header else "## Untitled Section"
            section_text = "\n".join(section.get("content", [])).strip()

            if not section_text:
                continue

            # Chunk section text by length
            start = 0
            while start < len(section_text):
                end = start + max_chunk_chars
                chunk_body = section_text[start:end]
                start = end

                chunk_text = f"{header_md}\n\n{chunk_body}"

                rag_chunks.append({
                    "chunk_id": chunk_id,
                    "header": header,
                    "text": chunk_text,
                    "metadata": {
                        "section": header or "",
                        "char_start": int(max(0, start - max_chunk_chars)),
                        "char_end": int(start),
                        "source": "PDF",
                        "session_id": str(session_id or ""),
                        "document_id": str(document_id),
                        "topic": str(user_prompt or "")
                    }
                })
                chunk_id += 1

        st.success("Information cleaned and structurally refined.")
        return rag_chunks, document_id

    # --- RAG INGESTION METHOD ---
    def _create_and_ingest_vectors(self, rag_chunks, session_id, document_id):
        """
        Ingests structured text chunks into a persistent ChromaDB collection.
        Each chunk carries session_id, document_id, and topic metadata.
        """
        st.subheader("3. Indexing Documents into Vector Database (ChromaDB)")

        VECTOR_DB_PATH = "D:/nlu_project_vector_store"

        try:
            # STEP 1: Initialize ChromaDB Client
            client = PersistentClient(path=VECTOR_DB_PATH)

            # STEP 2: Load the Embedding Model
            with st.spinner("Loading Sentence Transformer model for embeddings..."):
                embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

            # STEP 3: Prepare data for ChromaDB
            ids = [f"{document_id}_chunk_{c['chunk_id']}" for c in rag_chunks]
            documents = [c['text'] for c in rag_chunks]

            # Ensure metadata includes session_id and document_id
            for c in rag_chunks:
                c["metadata"]["session_id"] = str(session_id or "")
                c["metadata"]["document_id"] = str(document_id or "")

            metadatas = [c["metadata"] for c in rag_chunks]

            # STEP 4: Use ONE persistent collection
            collection = client.get_or_create_collection(name="learning_materials")

            # STEP 5: Generate Embeddings and Add to Collection
            with st.spinner(f"Generating embeddings and adding {len(rag_chunks)} chunks..."):
                embeddings = embedding_model.encode(documents, show_progress_bar=False).tolist()

                # Debug: print one metadata dict
                print("Sample metadata:", metadatas[0])

                collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings
                )

            st.success(f"Successfully indexed {len(rag_chunks)} chunks to persistent collection 'learning_materials'")
            return collection, embedding_model

        except Exception as e:
            st.error(f"Error during vector ingestion: {e}")
            return None, None

    # --- RAG RETRIEVAL METHOD ---
    def _retrieve_relevant_chunks(self, collection, embedding_model, query_text, k=5, session_id=None, document_id=None, topic=None):
        """
        Retrieves top 'k' relevant chunks from the persistent collection.
        Filters by session_id, document_id, or topic if provided.
        """
        st.subheader("4. Retrieving Relevant Context")

        if collection is None or embedding_model is None:
            st.error("Vector database or embedding model is not available for retrieval.")
            return "No context retrieved."

        try:
            query_embedding = embedding_model.encode([query_text]).tolist()

            # STEP 1: Build filters robustly
            filters = []
            if session_id:
                filters.append({"session_id": str(session_id)})
            if document_id:
                filters.append({"document_id": str(document_id)})
            if topic:
                filters.append({"topic": str(topic)})

            query_kwargs = {
                "query_embeddings": query_embedding,
                "n_results": k,
                "include": ["documents", "metadatas"]
            }

            # Chroma expects exactly one operator at the top level
            if len(filters) == 1:
                query_kwargs["where"] = filters[0]
            elif len(filters) > 1:
                query_kwargs["where"] = {"$and": filters}
            # else: no filter, global retrieval

            with st.spinner("Performing ChromaDB Query fetching..."):
                results = collection.query(**query_kwargs)

            #  STEP 2: Retrieving chunks
            retrieved_chunks = results["documents"][0] if results and results.get("documents") else []
            context = "\n\n---\n\n".join(retrieved_chunks)

            st.success(f"Retrieved {len(retrieved_chunks)} context chunks (filters applied: {query_kwargs.get('where', 'none')}).")
            return context if context else "No context retrieved."

        except Exception as e:
            st.error(f"Error during context retrieval: {e}")
            return "No context retrieved."

    # --- GENERATION METHOD (GEMINI) ---
    def _generate_learning_material(self, retrieved_text):
        """
        Creates a complete structured learning material using RAG + template.
        Loads template, fills metadata, and calls the Gemini API.
        Prompts strictly ground the model to use only the provided context.
        """
        st.subheader("5. Generating the Learning Material")

        # STEP 1: Retrieve Secrets and Config
        try:
            API_KEY = st.secrets["gemini_api_key"]
            MODEL_NAME = st.secrets["model_name"]
        except KeyError as e:
            st.error(f"Missing key in secrets.toml: {e}. Please ensure you have 'gemini_api_key' and 'model_name'.")
            return None

        # STEP 2: Load the template from file
        TEMPLATE_PATH = "assets/learning_material_template.mb"
        try:
            with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
                learning_template = f.read()
        except FileNotFoundError:
            st.error(f"Template file not found at {TEMPLATE_PATH}.")
            return None

        # STEP 3: Format the template (only retrieved_text is used)
        try:
            formatted_template = learning_template.format(
                retrieved_text=retrieved_text
            )
        except Exception as e:
            st.error(f"Template formatting error: {e}")
            return None

        # STEP 4: Add system and user prompts
        system_prompt = (
            """
            You are an expert educator AI that writes complete, accurate and highly structured learning materials following templates strictly.
            Use the provided TEMPLATE provided as primary source.
            If TEMPLATE is missing details, you may supplement with accurate general knowledge.
            The retrieved text is provided in the user message.
            """
        )

        user_prompt = (
            "TEMPLATE START\n"
            f"{formatted_template}\n"
            "TEMPLATE END\n\n"
            "Generate a complete structured learning material that cites or derives from CONTEXT."
            "Start with a cover page title and DO NOT include any AI disclaimers or commentary."
        )

        main_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        # STEP 5: Generate learning material
        with st.spinner(f"Generating learning material using Gemini model ({MODEL_NAME})..."):
            final_summary = _call_gemini_api(
                messages=main_messages,
                model_name=MODEL_NAME,
                api_key=API_KEY,
                max_tokens=5000
            )

        if final_summary is None:
            return None
        
        sections = _dissect_learning_material(final_summary)
        
        for header, content in sections.items():
            if header == "I. COVER PAGE":
                summary_title = content
                # Clean the title
                summary_title = re.sub(r'^\s*-\s*Title\s+of\s+the\s+Material:\s*', '', summary_title, flags=re.IGNORECASE)
                summary_title = summary_title.replace('"', "").replace("'", "").strip()
                summary_title = re.sub(r'[^A-Za-z0-9._-]', ' ', summary_title)
                sections["I. COVER PAGE"] = summary_title
                

        # If no title line was found, fall back
        if summary_title is None:
            summary_title = "Learning Material"

        st.success("Learning material generated using Gemini API (grounded to current context).")
        return summary_title, final_summary

    # --- PDF GENERATION METHOD ---
    def _generate_pdf(self, final_summary):
        """
        Creates a formatted PDF document from structured Markdown sections using PyMuPDF.
        - Bond paper (8.5 x 13 inches) with 1-inch margins.
        - Supports headings, lists, paragraphs, and cover page styling.
        - Uses wrap_text() for consistent width-based wrapping.
        """
        st.subheader("6. Creating Final PDF Learning Material")

        # STEP 1: Page setup
        width = 8.5 * 72  # 612 points
        height = 13 * 72  # 936 points
        margin = 72       # 1 inch margin
        max_width = width - 2 * margin

        doc = fitz.open()
        sections = _dissect_learning_material(final_summary)

        # STEP 2: Iterate sections
        for header, content in sections.items():
            page = doc.new_page(width=width, height=height)
            rect = page.rect
            y = margin

            # Section header
            page.insert_text((margin, y), header, fontsize=24, fontname="Helvetica-Bold")
            y += 50

            # Content lines
            for raw_line in content.split("\n"):
                line = raw_line.strip()
                if not line:
                    y += 12
                    if y > page.rect.height - margin:
                        page = doc.new_page(width=width, height=height)
                        rect = page.rect
                        y = margin
                    continue
                
                font_size, font_name, line, wrapped, y = check_formatting(header, line, max_width, y)

                # STEP 3: Render wrapped lines with bold re-application (if any)
                for wl in wrapped:
                    x = margin
                    # Only split if bold markers exist; otherwise render as a single part
                    parts = re.split(r'(\*\*.*?\*\*)', wl) if "**" in wl else [wl]
                    for part in parts:
                        if not part:
                            continue
                        if part.startswith("**") and part.endswith("**"):
                            text = part[2:-2]
                            page.insert_text((x, y), text, fontsize=font_size, fontname="Helvetica-Bold")
                            x += fitz.get_text_length(text, fontname="Helvetica-Bold", fontsize=font_size)
                        else:
                            page.insert_text((x, y), part, fontsize=font_size, fontname=font_name)
                            x += fitz.get_text_length(part, fontname=font_name, fontsize=font_size)

                    y += font_size + 6
                    if y > page.rect.height - margin:
                        page = doc.new_page(width=width, height=height)
                        rect = page.rect
                        y = margin

            # STEP 4: Footer (right-aligned via textbox)
            footer_rect = fitz.Rect(margin, page.rect.height - 50, page.rect.width - margin, page.rect.height - 20)
            page.insert_textbox(footer_rect, f"Page {doc.page_count}", fontsize=10, fontname="Helvetica", align=2)

        pdf_bytes = doc.tobytes()
        st.success("Final PDF generated with improved formatting!")
        return pdf_bytes
