M.Sc. IT Program Q&A Assistant – TH OWL

A simple Retrieval-Augmented Generation (RAG) based chatbot that answers
questions about the M.Sc. Information Technology program at Technische
Hochschule Ostwestfalen-Lippe (TH OWL).

The system uses: - ChromaDB for vector storage - Groq LLM API for answer
generation - Python for implementation

------------------------------------------------------------------------

FEATURES

-   Persistent vector database using ChromaDB
-   Context-based answer generation (RAG architecture)
-   Restricts responses to stored knowledge
-   Command-line interactive chatbot
-   Environment variable support for API key

------------------------------------------------------------------------

PROJECT STRUCTURE

. ├── main.py ├── .env ├── chroma_db/ # Persistent ChromaDB storage
(auto-created) └── README.txt

------------------------------------------------------------------------

INSTALLATION

1.  Clone the Repository

git clone cd

2.  Create a Virtual Environment (Recommended)

Mac/Linux: python -m venv venv source venv/bin/activate

Windows: python -m venv venv venv

3.  Install Dependencies

pip install groq chromadb python-dotenv

------------------------------------------------------------------------

ENVIRONMENT VARIABLES SETUP

Create a .env file in the root directory:

GROQ_API_KEY=your_groq_api_key_here

------------------------------------------------------------------------

RUNNING THE APPLICATION

python main.py

Type your question and press Enter. To exit, type: exit or quit

------------------------------------------------------------------------

HOW IT WORKS

Step 1: Build Vector Database The knowledge base is embedded and stored
in ChromaDB. If the collection already exists, it is reused.

Step 2: Retrieve Relevant Context The system searches for the most
relevant document using semantic similarity. Default: top 3 results
(currently only the first result is used).

Step 3: Generate Answer The retrieved context is passed to the Groq LLM.
The system prompt enforces answering only from the provided context.

------------------------------------------------------------------------

LIMITATIONS

-   Uses a small static knowledge base
-   Currently only the first retrieved document is used
-   No web interface (CLI only)
-   No embedding model explicitly defined

------------------------------------------------------------------------

Technologies Used: Python, ChromaDB, Groq API, python-dotenv

License: This project is for educational purposes.
