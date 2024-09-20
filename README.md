
# Contract Query ChatBot

This project provides a solution for querying contracts using advanced document retrieval and parsing techniques. The app is built using Python and Streamlit. Below are the steps to get started with the project.

---

## ⚙️ Project Setup

### 1. Clone the Repository
To start working with the project, first, clone the repository using the following command:

```bash
git clone https://github.com/im-shivanig/Contract_Query_ChatBot.git
```

Navigate into the project directory:

```bash
cd Contract_Query_ChatBot
```

### 2. Create a Virtual Environment (venv)
Set up a virtual environment to manage dependencies:

```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment
Activate the virtual environment using the appropriate command for your operating system:

For macOS/Linux:

```bash
source venv/bin/activate
```

For Windows:

```bash
.\venv\Scripts\activate
```

### 4. Install Required Dependencies
Once the virtual environment is activated, install the required Python libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## 📁 Data Storage & Management

### 5. Data Storage Plan (TODO)
Make sure to store data in appropriate locations. This will ensure the seamless operation of the app.

### 6. Instructions for Storing Data
Follow the guidelines in the provided PDF on where to store your data. The PDF contains detailed instructions on storing contracts and parsed text data.

---

## 📦 Vector Store & Parsed Text Directory

### 7. Fetching Data
If you already have a vector store and parsed text directory, fetch them from your data storage locations and store them in the appropriate directories within the project folder.

Make sure to place the `vectorstore` and `parsed_txts` in the correct location based on the data storage guidelines.

---

## 🚀 Running the Application

### 8. Running the Streamlit App
To run the Streamlit app, execute the following command from the terminal:

```bash
streamlit run app.py
```

This will start the Contract Query ChatBot web interface, where you can interact with the system to retrieve contract information.

---

## 💡 Additional Notes
- Ensure that all dependencies are properly installed in the virtual environment.
- Data storage guidelines must be followed to ensure proper retrieval and parsing.
- The app can be customized further based on specific use cases and requirements.

---

🌟 We hope this guide helps you set up and run the Contract Query ChatBot with ease! If you have any issues, feel free to create an issue on the [GitHub repository](https://github.com/im-shivanig/Contract_Query_ChatBot/issues).
