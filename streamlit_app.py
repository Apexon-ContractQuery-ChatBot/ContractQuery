
import streamlit as st
from rag_system import query_rag_system

st.title("RAG-Based Contract Query System")

user_query = st.text_input("Enter your query about the contract:")

if st.button("Get Answer"):
    if user_query:
        response = query_rag_system(user_query)
        st.write("Answer:", response)
    else:
        st.write("Please enter a query.")
