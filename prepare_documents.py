import os
import re
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from configuration import data_directory,parsed_txt_directory


def normalize_text(text):
    text = text.replace("P RO GRAM", "PROGRAM")
    text = re.sub(r'A\s*R\s*T\s*I\s*C\s*L\s*E', 'ARTICLE', text)  # Converting "A RT ICLE" to "ARTICLE"
    return text

# Extract and store text from PDF
def extract_and_store_text(pdf_path, save_path):
    text = ""  # Initialize empty string
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for i, page in enumerate(reader.pages):
            page_num = i + 1
            header = f"DOCUMENT NAME: {os.path.basename(pdf_path)} | PAGE: {page_num}\n---\n"
            page_text = page.extract_text() if page.extract_text() else ""
            text += header + page_text + "\n---\n"  # Add a page break marker
    normalized_text = normalize_text(text)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(normalized_text)

# Process all PDFs in the 'Data' directory
def parse_files():
    pdf_files = [f for f in os.listdir(data_directory) if f.endswith('.pdf')]
    total_files = len(pdf_files)
    print(f"Total PDF files found in the directory: {total_files}")
    for filename in pdf_files:
        file_path = os.path.join(data_directory, filename)
        save_path = os.path.join(parsed_txt_directory, f"{os.path.splitext(filename)[0]}_parsed.txt")
    
    # Check if the parsed text file already exists
        if not os.path.exists(save_path):
            extract_and_store_text(file_path, save_path)
            print(f"Parsed and saved: {save_path}")
        else:
            print(f"Skipped existing file: {save_path}")

    print(f"All files have been processed.")

def prepare_docs():
    documents = []
    txt_files = [f for f in os.listdir(parsed_txt_directory) if f.endswith('.txt')]

    for filename in txt_files:
        file_path = os.path.join(parsed_txt_directory, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            documents.append({'text': text, 'metadata': {'source': filename}})

    texts = [doc['text'] for doc in documents]
    metadatas = [doc['metadata'] for doc in documents]
    return texts,metadatas

def chunking (texts,metadatas):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=500)

    chunks = []
    chunk_metadatas = []

    for i, text in enumerate(texts):
        split_chunks = text_splitter.split_text(text)
        for chunk in split_chunks:
            chunks.append(chunk)
            chunk_metadatas.append({'source': metadatas[i]['source']})
    return chunks,chunk_metadatas

