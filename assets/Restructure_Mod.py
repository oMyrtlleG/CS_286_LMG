import re 

# --- HELPER 1: CHARACTER & WHITESPACE CLEANING ---
def _clean_raw_text(text):
    """
    Handles low-level character encoding, hyphenation, and whitespace normalization.
    """
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'[ \t]+', ' ', text)
    # Note: We keep [BLOCK_SEP] intact here for the next step.
    return text.strip()

# --- HELPER 2: BOILERPLATE REMOVAL ---
def _remove_boilerplate(text):
    """
    Removes common document noise like page numbers, document headers/footers.
    """
    text = re.sub(
        r'(?:^|\n)\s*(?:Page\s+\d+.*|(\d+)\s*(?:of\s*\d+)?)\s*(?:\n|$)', 
        '\n', 
        text, 
        flags=re.MULTILINE | re.IGNORECASE
    )
    return text.strip()

# --- HELPER 3: GROUP BY EXTRACTED TAGS ---
def _group_by_extracted_tags(tagged_text):
    """
    Groups content based on the [H1]/[H2] tags created during extraction,
    and converts the custom tags into standard Markdown for final processing/PDF gen.
    """
    grouped_content = []
    
    # Split the entire text by our custom block separator first.
    blocks = tagged_text.split('[BLOCK_SEP]')

    current_section = {"header": "Untitled Document Section", "content": []}

    # Regex to capture the tag and the content
    TAG_PATTERN = re.compile(r'\[(H[12]|P|B)\]\s*(.*?)\s*\[/\1\]', re.DOTALL)
    
    for block in blocks:
        block = block.strip()
        if not block: continue
        
        match = TAG_PATTERN.match(block)
        if match:
            tag = match.group(1)
            content = match.group(2).strip()

            if tag in ["H1", "H2"]:
                # End the previous section and start a new one
                if current_section["content"]:
                    grouped_content.append(current_section)
                
                # Use the header content
                current_section = {"header": content, "content": []}
            
            elif tag in ["P", "B"]:
                # Convert custom tags to Markdown for final output consistency
                if tag == "B":
                    content = f"**{content}**"
                
                # Append to the current section's content list
                current_section["content"].append(content)
        else:
            # Handle text blocks that didn't match the tag pattern (e.g., OCR output)
            if block.startswith('[-- OCR TEXT'):
                 current_section["content"].append(f"\n{block}\n")
            elif block.strip():
                 # Treat untagged text as a regular paragraph
                 current_section["content"].append(block.strip())

    # Add the last section
    if current_section["content"] or current_section["header"] != "Untitled Document Section":
        grouped_content.append(current_section)

    return grouped_content
