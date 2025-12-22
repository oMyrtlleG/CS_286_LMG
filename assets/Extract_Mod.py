from PIL import Image, ImageEnhance 

# --- HELPER 1: TEXT STRUCTURING BY BLOCK ---
def _reconstruct_text_from_blocks(page_dict):
    """
    Reconstructs text from PyMuPDF's 'dict' output.
    Uses font size and style to infer structure (Headers, Bold text).
    """
    text = []
    
    # 1. Determine a baseline font size (e.g., the most common or median size)
    #    For simplicity, we'll use a fixed threshold or the size of the first text block.
    #    In a production system, you'd calculate the median size for the document.
    
    # Simple Heuristic: Assume largest size is the main title size
    largest_font_size = 0
    for block in page_dict.get('blocks', []):
        if block.get('type') == 0:
            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    if span['size'] > largest_font_size:
                        largest_font_size = span['size']

    # Set thresholds relative to the largest found size
    # This makes the detection adaptive to the document's design
    if largest_font_size == 0:
        return ""
        
    H1_THRESHOLD = largest_font_size * 0.95
    H2_THRESHOLD = largest_font_size * 0.8
    BOLD_FLAG = 2  # PyMuPDF flag for bold

    for block in page_dict.get('blocks', []):
        if block.get('type') == 0: 
            block_text = []
            for line in block.get('lines', []):
                line_text = []
                # Find the properties for the first span (or all if consistent)
                first_span = next((s for s in line.get('spans', []) if s['text'].strip()), None)
                if not first_span: continue
                
                size = first_span['size']
                is_bold = (first_span['flags'] & BOLD_FLAG) == BOLD_FLAG

                prefix = ""
                suffix = ""
                
                # Use custom tags for clear structural markers
                if size >= H1_THRESHOLD:
                    prefix = "[H1] "  
                    suffix = " [/H1]" 
                elif size >= H2_THRESHOLD:
                    prefix = "[H2] " 
                    suffix = " [/H2]"
                elif is_bold:
                    prefix = "[B]"
                    suffix = "[/B]"
                else:
                    prefix = "[P]"
                    suffix = "[/P]"
                
                # Reconstruct the text for the line
                line_content = "".join([span['text'] for span in line.get('spans', [])]).strip()
                
                # For simplicity, apply the tag to the whole line's content
                block_text.append(f"{prefix}{line_content}{suffix}")

            # Separate blocks with a clear, single separator.
            text.append("\n".join(block_text).strip())
            
    # Use a clear separator for structural blocks.
    return "\n[SECTION_BREAK]\n".join(text)

# --- HELPER 2: OCR FOR IMAGE-TEXT EXTRACTION ---
def _preprocess_image_for_ocr(img):
    """
    Applies enhanced preprocessing for better Tesseract accuracy.
    """
    img = img.convert('L') 
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.8) 
    img = img.point(lambda x: 0 if x < 150 else 255, '1') 
    return img