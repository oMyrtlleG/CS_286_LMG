import fitz, re

# --- HELPER 1: TEXT FORMATTING ---
def check_formatting(header, line, max_width, y):
    # Detect markdown categories and set style
    
    if header == "I. COVER PAGE":
        font_size, font_name = 28, "Helvetica-Bold"

    elif line.startswith("### "):
        font_size, font_name = 14, "Helvetica-Bold"
        line = line[4:].strip()
        y += 20

    elif line.startswith("## "):
        font_size, font_name = 16, "Helvetica-Bold"
        line = line[3:].strip()
        y += 25

    elif line.startswith("# "):
        font_size, font_name = 20, "Helvetica-Bold"
        line = line[2:].strip()
        y += 30

    elif line.startswith(("-", "* ", "o")):
        font_size, font_name = 12, "Helvetica"
        line = "• " + line[1:].strip()
        y += 10
        
    else:
        font_size, font_name = 12, "Helvetica"
    
    wrapped = wrap_text(line, font_name, font_size, max_width)
    return font_size, font_name, line, wrapped, y

# --- HELPER 2: TEXT WRAP ---
def wrap_text(text, fontname, fontsize, max_width):
    """
    Wraps text into multiple lines so each line fits within max_width.
    
    Args:
        text (str): Input text to wrap.
        fontname (str): Font name used for measuring text width.
        fontsize (int): Font size used for measuring text width.
        max_width (int): Maximum allowed width for a line in points.
        
    Returns:
        list of str: Wrapped lines of text.
    """
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        text_width = fitz.get_text_length(test_line, fontname=fontname, fontsize=fontsize)
        if text_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines