import streamlit as st
from contract_query_system import prepare_contract_query_system, query_contract_system
from langchain_core.messages import AIMessage, HumanMessage

@st.cache_resource
def load_system():
    return prepare_contract_query_system()

# Load the system once and cache it
retrieval_qa = load_system()

# Streamlit app layout
col1, mid, col2 = st.columns([1,1,20])
with col1:
    st.image("C:/Users/Shivani Gangarapollu/Downloads/Contract.png", width=70)
with col2:
    st.title("Contract Query Bot ")
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a Contract Query bot. How can I help you?"),
    ]
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

user_query = st.chat_input("Enter your query about the contract:")

if user_query is not None :
   

    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)
        user_query=user_query.upper()
        answer, source = query_contract_system(retrieval_qa,user_query)
    with st.chat_message("AI"):
        response = answer
        st.write(response)
        st.write("**Source Document:**", source)
    st.session_state.chat_history.append(AIMessage(content=response))
else:
        st.write("")
