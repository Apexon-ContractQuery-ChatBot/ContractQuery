from langchain_core.messages import AIMessage, HumanMessage
import os
import streamlit as st
from prepare_documents import prepare_docs, chunking, parse_files, check_parsed_files
from retrieval import save_to_chroma, retriever_system, query_system
from configuration import parsed_txt_directory, vectorstore_dir, image_path

# Function to load the vectorstore and prepare the system
def load():

    # Check if all files are parsed and up to date
    st.spinner("Checking for new files in the data directory...")
    check_parsed_files()

    # Check if the parsed text directory exists and is populated
    if not os.path.exists(parsed_txt_directory) or len(os.listdir(parsed_txt_directory)) == 0:
        print("Parsed text directory is empty or missing. Parsing files...")
        parse_files()  # Trigger parsing of PDFs

    # Check if the vectorstore exists, otherwise process embeddings
    if not os.path.exists(vectorstore_dir) or len(os.listdir(vectorstore_dir)) == 0:
        with st.spinner("Creating embeddings"):
            print("Vectorstore is empty or missing. Processing embeddings...")
            # Extract documents and create the vectorstore
            texts, metadatas = prepare_docs()
            chunks, chunk_metadatas = chunking(texts, metadatas)
            vectorstore = save_to_chroma(chunks, chunk_metadatas)  # Save the new vectorstore
    else:
        print("Vectorstore exists. Loading it...")
        # Load the existing vectorstore without reprocessing
        vectorstore = save_to_chroma([], [])  # Just load it, don't add new data

    # Create the retrieval system based on the loaded vectorstore
    qa_chain = retriever_system(vectorstore)
    return qa_chain

# Initialize the app and load vectorstore once
if "qa_chain" not in st.session_state:
    print("Initializing vectorstore for the first time...")
    st.session_state.qa_chain = load()  # Load vectorstore only once and store it in session state
else:
    print("Vectorstore already initialized")

# Streamlit app layout
col1, mid, col2 = st.columns([1, 0.5, 1])

with mid:
    st.image(image_path, width=70)

st.markdown("<h1 style='text-align: center;'>Contract Query Bot</h1>", unsafe_allow_html=True)

# Initialize chat history if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a Contract Query bot. How can I help you?"),
    ]

# Display chat history
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
