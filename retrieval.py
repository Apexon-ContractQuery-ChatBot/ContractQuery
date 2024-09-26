from configuration import openai_api_key,data_directory,parsed_txt_directory,vectorstore_dir
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


# Check if vectorstore exists, otherwise create one
def save_to_chroma(chunks,chunk_metadatas):
    if not os.path.exists(vectorstore_dir) or not os.listdir(vectorstore_dir):
        print("Vector store not found. Creating a new one...")
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings, metadatas=chunk_metadatas, persist_directory=vectorstore_dir)
        vectorstore.persist()
        print("Vector store saved to local")
    else:
        print("Vector store already exists")
        vectorstore = Chroma(persist_directory=vectorstore_dir, embedding_function=OpenAIEmbeddings(openai_api_key=openai_api_key))
    return vectorstore


# Complete retriever logic
def retriever_system(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_k=3)


    # Set up compressor for contextual compressions
    llm = OpenAI(temperature=0, max_tokens=1500,openai_api_key=openai_api_key)
    compressor = LLMChainExtractor.from_llm(llm)


    # Create a contextual compression retriever
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,base_retriever=retriever)


    # Prompt template
    prompt_template = """
You are an advanced **Contract Query Chatbot**, designed to provide precise and professional answers based on legal contract information provided in the context. Follow these guidelines to deliver the most accurate and well-presented response:

1. **Use only the information from the provided context** to answer the query. Do not assume or reference external knowledge.
2. **Clarify vague queries**: If the query lacks specificity, ask for clarification rather than assuming intent.
3. If the query is unclear, **break it down logically** and clarify each part using the context, then provide a comprehensive answer.
4. If the context does not contain enough information to address the query, state: _"The retrieved context does not contain sufficient information to answer the query." - and mention which specific details are missing.
5. For queries involving **multiple relevant sections**, summarize each section concisely and explain its relevance to the query.
6. **Reference specific sections, clauses, or page numbers** within the context to support your response. If necessary, break down complex sections for clarity.
7. Maintain a **neutral, professional tone**, avoiding speculation or interpretation of legal language unless explicitly stated in the context.
8. For **time-sensitive queries** (e.g., contract duration, deadlines), include **exact dates, timelines**, or periods as mentioned in the context.
9. For queries requiring **comparisons or interpretations**, provide a balanced explanation, noting any ambiguities, exceptions, or alternate interpretations present in the contract.
10. For action items or obligations, clearly specify **who is responsible**, when they are expected to fulfill the obligation, and any relevant conditions from the contract.
11. If multiple options or outcomes are possible, clearly outline **the conditions under which each applies**.
12. **If the query spans multiple subtopics**, clearly indicate which aspects have been addressed and which lack sufficient detail in the provided context.
13. **Present the answer in a clear, structured, and visually appealing format**, using appropriate headings, bullet points, or numbered lists for readability. Avoid large blocks of text.

---

**Context (retrieved from documents):**  
{context}

---

**Query:**  
{question}

---

**Answer:**
"""

    # Create the QA chain with prompt, LLM, and retriever
    qa_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm_chain = LLMChain(llm=llm, prompt=qa_prompt)

    document_variable_name = "context"
    combine_documents_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name=document_variable_name)
    qa_chain = RetrievalQA(retriever=compression_retriever, combine_documents_chain=combine_documents_chain)
    return qa_chain


# Querying function
def query_system(query,qa_chain):
    result = qa_chain.run(query)
    elaborative_response = f"**Query:** {query}\n\n"
    elaborative_response += "**Relevant Information Found:**\n\n"
    elaborative_response += f"{result}\n\n"
    return elaborative_response
