import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_file):
    text = ""
    doc = fitz.open(pdf_file)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    doc.close()
    return text

def save_to_markdown(text, markdown_file):
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write(text)

# usage
pdf_file = "Pmg_lds.pdf"  # Replace with your PDF file path
markdown_file = "output.md"  # Replace with desired Markdown file path
extracted_text = extract_text_from_pdf(pdf_file)
save_to_markdown(extracted_text, markdown_file)
