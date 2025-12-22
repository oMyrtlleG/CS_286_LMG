import re

# --- HELPER 1: DISSECT GENERATED LEARNING MATERIAL ---
def _dissect_learning_material(final_summary: str):
    """
    Splits the generated learning material into sections based on the RAG template.
    Returns a dict mapping section names to their content.
    """
    # Define the section headers in order
    headers = [
        "I. COVER PAGE",
        "II. INTRODUCTION",
        "III. CORE CONTENT",
        "IV. SUPPLEMENTARY MATERIALS",
        "V. ASSESSMENT SECTION",
        "VI. SUMMARY / KEY TAKEAWAYS",
    ]
    
    # Build regex pattern to capture each header and its content
    pattern = r"(" + "|".join(re.escape(h) for h in headers) + r")"
    
    # Split by headers, keeping them in the result
    parts = re.split(pattern, final_summary)
    
    # Assemble into dictionary
    sections = {}
    current_header = None
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part in headers:
            current_header = part
            sections[current_header] = ""
        elif current_header:
            sections[current_header] += part + "\n"
    
    return sections