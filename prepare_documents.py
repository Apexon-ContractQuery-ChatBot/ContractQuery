import os
import re
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from configuration import data_directory,parsed_txt_directory


#Text Normalization
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
            text += header + page_text + "\n---\n"
    normalized_text = normalize_text(text)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(normalized_text)


def check_parsed_files():
    pdf_files = [f for f in os.listdir(data_directory) if f.endswith('.pdf')]
    parsed_files = [f.replace('_parsed.txt', '.pdf') for f in os.listdir(parsed_txt_directory) if f.endswith('_parsed.txt')]

    new_files_found = False  #Flag initialized to False

    # Parse any new files which are not yet processed
    for pdf_file in pdf_files:
        if pdf_file not in parsed_files:
            new_files_found = True  # Flag changes to True when a new file is found
            print(f"New file found: {pdf_file}. Parsing the file now")
            file_path = os.path.join(data_directory, pdf_file)
            save_path = os.path.join(parsed_txt_directory, f"{os.path.splitext(pdf_file)[0]}_parsed.txt")
            extract_and_store_text(file_path, save_path)
            print(f"Parsed and saved: {save_path}")

    if not new_files_found:
        print("Parsed files are up-to-date.")

    return new_files_found  #Returns the flag

    

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


def chunking(texts,metadatas):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=500)
    chunks = []
    chunk_metadatas = []

    for i, text in enumerate(texts):
        split_chunks = text_splitter.split_text(text)
        for chunk in split_chunks:
            chunks.append(chunk)
            chunk_metadatas.append({'source': metadatas[i]['source']})
    return chunks,chunk_metadatas
