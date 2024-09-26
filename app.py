from langchain_core.messages import AIMessage, HumanMessage
import os
import shutil
import streamlit as st
from prepare_documents import prepare_docs, chunking, check_parsed_files
from retrieval import save_to_chroma, retriever_system, query_system
from configuration import parsed_txt_directory, vectorstore_dir, image_path

col1, mid, col2 = st.columns([1, 0.5, 1])
with mid:
    st.image(image_path, width=70)
st.markdown("<h1 style='text-align: center;'>Contract Query Bot</h1>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a Contract Query bot. How can I help you?"),
    ]

# Function to create or load the embeddings
def embedding_func():
    texts, metadatas = prepare_docs()
    chunks, chunk_metadatas = chunking(texts, metadatas)
    vectorstore = save_to_chroma(chunks, chunk_metadatas)
    return vectorstore

# Function to load the vectorstore and prepare the system
def load():
    
    # Check if all files are parsed and up to date
    with st.spinner("Checking for new files in the data directory..."):
        new_files_found = check_parsed_files()  # Get the flag

     # Case 1: New files found and vectorstore exists
    if new_files_found and os.path.exists(vectorstore_dir):
        with st.spinner("New file found .... recreating the vectorstore"):
            shutil.rmtree(vectorstore_dir)  # Delete the existing vectorstore
            vectorstore = embedding_func()  
       
    # Case 2: New files found and vectorstore does not exist
    elif new_files_found and not os.path.exists(vectorstore_dir):
        with st.spinner("Creating embeddings due to new files and missing vectorstore..."):
            vectorstore = embedding_func()  
        
    # Case 3: No new files found and vectorstore exists
    elif not new_files_found and os.path.exists(vectorstore_dir):
        with st.spinner("Loading existing vectorstore..."):
            vectorstore = save_to_chroma([], [])  

    # Case 4: No new files found and vectorstore does not exist
    else:
        with st.spinner("Creating embeddings due to missing Vectorstore"):
            vectorstore = embedding_func()  
        
    qa_chain = retriever_system(vectorstore)
    return qa_chain

    
# Initialize the app and load vectorstore once
if "qa_chain" not in st.session_state:
    print("Initializing vectorstore for the first time...")
    st.session_state.qa_chain = load()  # Load vectorstore only once and store it in session state
else:
    print("Vectorstore already initialized")


# Initialize chat history in session state if not already present
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)


# Input box for user queries
if user_input := st.chat_input("Ask a question"):
    # Append the user query to chat history
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Query the system using the preloaded vectorstore
    response = query_system(user_input, st.session_state.qa_chain)

    # Append the AI's response to chat history
    st.session_state.chat_history.append(AIMessage(content=response))

    # Display the AI response
    with st.chat_message("AI"):
        st.write(response) 
