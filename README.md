
# Contract Query Chatbot

## Project Description
This project is a B2B Contract Query Chatbot built using a Retrieval-Augmented Generation (RAG) approach. The chatbot assists users in querying contract-related information between B2B vendors. It utilizes Langchain for retrieval and response generation, and ChromaDB for efficient document storage and retrieval. The chatbot is designed with guardrails to ensure that responses are contextually relevant and accurate.

## User Instructions
### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/B2B-Contract-Query-Chatbot.git
    cd B2B-Contract-Query-Chatbot
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv chatbot_env
    source chatbot_env/bin/activate  # On Windows, use `chatbot_env\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application
1. Start the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Open your browser and navigate to `http://localhost:8501` to interact with the chatbot.

### Example Queries
- "What are the payment terms in our contract?"
- "When is the delivery scheduled for the next order?"

For more detailed user instructions, please refer to the [User Manual](https://link-to-user-manual).

## Key Features
- **Query Categorization**: Automatically categorizes user queries to retrieve the most relevant documents.
- **Document Retrieval**: Efficiently retrieves documents using ChromaDB and embeddings.
- **Contextual Response Generation**: Generates contextually accurate responses using Langchain.
- **Relevance Guardrails**: Implements guardrails to filter out irrelevant information.
- **User Feedback Loop**: Allows users to provide feedback to continuously improve the chatbot’s accuracy.

## Project Contributors
- **Contributor 1 Name** - Role
- **Contributor 2 Name** - Role


