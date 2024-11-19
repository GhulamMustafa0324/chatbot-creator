# Flask Chatbot Wizard

This project is a Flask-based web application that provides a step-by-step wizard for creating custom chatbots. 
It allows users to register, log in, create chatbots, upload knowledge bases, and test chatbot responses.

## Features

1. **User Registration and Login**:
   - Unique username and email registration with password validation.
   - Login with email or username, with session timeout after 24 hours.

2. **Dashboard**:
   - View created chatbots with stats like total bots and creation dates.

3. **Chatbot Wizard**:
   - Step 1: Basic Information (name and description).
   - Step 2: Personality Setup (tone and behavior).
   - Step 3: Knowledge Base Upload (PDF, Word, and TXT files).

4. **Document Processing**:
   - Processes uploaded documents using LangChain.
   - Stores embeddings in ChromaDB for efficient retrieval.

5. **Real-Time Chatbot Testing**:
   - Test the chatbot responses during setup.

## Technologies Used

- **Backend**: Flask, Flask-Login, Flask-SQLAlchemy
- **Frontend**: Jinja2 templates (HTML)
- **Machine Learning**: HuggingFace Transformers (`SmolLM2-1.7B-Instruct`)
- **Vector Store**: ChromaDB
- **File Processing**: LangChain for document loading and splitting

## Installation

1. Clone the repository:
   ```bash
   git clone 
   cd flask_chatbot
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Create the DB and Run the Application:
    ```bash
    python app.py
    ```

