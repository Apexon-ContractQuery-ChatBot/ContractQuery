import os
import pdfplumber
import openai
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.llms import OpenAI
#from langchain import PromptTemplate

# Load OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define paths and cache file
pdf_directory = 'C:/Users/Shivani Gangarapollu/GenAI/Trials'  # Replace with your directory path
embedding_cache_path = "embedding_cache.pkl"

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF and return a list of (text, page_number) tuples."""
    extracted_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:  # Ensure the page has text
                extracted_pages.append((text, page_number))
    #print (extracted_pages)
    return extracted_pages

def hash_file(directory, filename):
    """Generate a hash for a file to detect changes."""
    h = hashlib.md5()
    file_path = os.path.join(directory, filename)
    with open(file_path, 'rb') as file:
        while chunk := file.read(8192):
            h.update(chunk)
    return h.hexdigest()

def load_cached_embeddings(cache_path):
    """Load cached embeddings if available."""
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        print("Loaded cached embeddings.")
    else:
        cache = {}
    return cache

def save_embeddings_to_cache(cache_path, cache):
    """Save embeddings to cache."""
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)
    print("Saved embeddings to cache.")

def extract_text_with_page_numbers(filename, pdf_directory):
    """Helper function to extract text and page numbers for multiprocessing."""
    try:
        pdf_path = os.path.join(pdf_directory, filename)
        pages = extract_text_from_pdf(pdf_path)
        return filename, pages
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return filename, []

def load_documents(pdf_directory, num_workers=4):
    documents = []
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith(".pdf")]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(extract_text_with_page_numbers, pdf_files, [pdf_directory] * len(pdf_files)))
    
    for filename, pages in results:
        for text, page_number in pages:
            documents.append(Document(page_content=text, metadata={"source": filename, "page_number": page_number}))
    
    return documents

def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=2000)
    chunked_documents = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            chunked_documents.append(Document(page_content=chunk, metadata=doc.metadata))  # Keep the original metadata
    return chunked_documents

def create_vectorstore(chunked_documents, cache):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory="chromadb_data", embedding_function=embeddings)

    documents_to_add = []
    for doc in chunked_documents:
        file_hash = hash_file(pdf_directory, doc.metadata['source'])  # Use full path for hashing
        if file_hash not in cache:  # Calculate embedding if not cached
            doc_embedding = embeddings.embed_documents([doc.page_content])
            cache[file_hash] = doc_embedding
            documents_to_add.append(Document(page_content=doc.page_content, metadata=doc.metadata))
    
    if documents_to_add:
        vectorstore.add_documents(documents_to_add)
    
    return vectorstore

def create_retrieval_qa_with_semantic_search(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
        
    )
    return retrieval_qa

def query_contract_system(retrieval_qa, query):
    result = retrieval_qa(query)
    answer = result['result']
    source_doc = result['source_documents'][0]
    source = f"{source_doc.metadata.get('source', 'Unknown')} (Page {source_doc.metadata.get('page_number', 'Unknown')})"
    return answer, source

def prepare_contract_query_system():
    documents = load_documents(pdf_directory)
    chunked_documents = chunk_documents(documents)
    
    # Load or create embeddings
    cache = load_cached_embeddings(embedding_cache_path)
    vectorstore = create_vectorstore(chunked_documents, cache)
    save_embeddings_to_cache(embedding_cache_path, cache)

    retrieval_qa = create_retrieval_qa_with_semantic_search(vectorstore)
    return retrieval_qa
