
import os
import openai
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader

# Load OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Step 1: Import the documents from the database
documents_directory = '/Users/vedantjoshi/Desktop/cq_Bot/ContractQuery/Synthetic.Data/'  

# Load all text files
text_loader = DirectoryLoader(documents_directory, glob="*.txt", loader_cls=TextLoader)
text_documents = text_loader.load()

# Load all PDF files
pdf_loader = DirectoryLoader(documents_directory, glob="*.pdf", loader_cls=PyPDFLoader)
pdf_documents = pdf_loader.load()

# Combine all documents
documents = text_documents + pdf_documents

# Step 2: Initialize the OpenAI embedding model
embeddings = OpenAIEmbeddings()

# Step 3: Initialize ChromaDB and add documents
vectorstore = Chroma(persist_directory="chromadb_data", embedding_function=embeddings)
vectorstore.add_texts(texts=[doc.page_content for doc in documents], metadatas=[doc.metadata for doc in documents])

# Persist the ChromaDB instance (this step is necessary if you want to use the database in future sessions)
vectorstore.persist()

# Step 4: Create the RetrievalQA chain using Langchain
retrieval_qa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),  # OpenAI model instance
    chain_type="stuff",  # RAG retrieval strategy
    retriever=vectorstore.as_retriever()
)

def query_rag_system(user_query):
    result = retrieval_qa.run(user_query)
    return result
