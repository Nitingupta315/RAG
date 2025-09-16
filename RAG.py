import requests
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Your knowledge base
knowledge_base = [
    "The Master of Science in Information Technology program at TH OWL is a full-time, non-restricted degree taught in English at the Innovation Campus in Lemgo.",
    "The standard duration of the program is four semesters, and there are no tuition fees.",
    "The program can also be pursued part-time.",
    "To be admitted, applicants must have a bachelor's or comparable degree in a related field such as electrical engineering, information technology, computer science, or mechatronics, with a final grade of 2.5 (German) or better.",
    "Proof of English language proficiency at level B2 of the CEFR is required for admission.",
    "Non-EU applicants who have not earned their degree in a country belonging to the Bologna signatory states must also provide a Graduate Record Examination (GRE) or Graduate Aptitude Test in Engineering (GATE) score.",
    "Required application documents include the bachelor's certificate, proof of English skills, and a letter of motivation.",
    "The program starts in both the summer and winter semesters.",
    "The curriculum is structured over four semesters.",
    "The first semester includes advanced topics in algorithms, probability and statistics, and management skills.",
    "The second semester focuses on innovation and development strategies and a compulsory elective module.",
    "The third semester is dedicated to a research project, while the fourth semester is for the master's thesis and colloquium.",
    "The program aims to provide students with expertise in intelligent technical systems, business and academic knowledge, and current topics like artificial intelligence.",
    "Graduates are prepared for management positions in international technology companies."
]

# Step 2: NEW Retrieval Function with Semantic Search
# This now uses TF-IDF, a simple form of semantic search. For a more advanced version,
# you would use an embedding model (like Ollama's nomic-embed-text).

def retrieve_passages_semantic(query, kb, top_n=3):
    """Retrieves passages using TF-IDF vectorization and cosine similarity."""
    vectorizer = TfidfVectorizer().fit(kb + [query])
    kb_vectors = vectorizer.transform(kb)
    query_vector = vectorizer.transform([query])
    
    similarities = cosine_similarity(query_vector, kb_vectors).flatten()
    
    # Get the indices of the top N most similar passages
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    relevant_passages = [kb[i] for i in top_indices if similarities[i] > 0]
    
    if not relevant_passages:
        return "No relevant information found in the knowledge base."
    
    return " ".join(relevant_passages)

# Step 3: Ollama Generation Function (remains the same)
def ollama_generate(user_query, context, model="deepseek-r1:7b"):
    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": f"You are a helpful assistant providing information about the M.Sc. Information Technology program at TH OWL based on the following context. Only use the information provided. If the information isn't available, state that you cannot answer based on the provided data.\n\nContext: {context}"},
                {"role": "user", "content": user_query}
            ],
            "stream": False
        }
        response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()['message']['content']
    except requests.exceptions.RequestException as e:
        return f"Error: Could not connect to Ollama. Please ensure Ollama is running and the '{model}' model is downloaded. Details: {e}"
    except json.JSONDecodeError:
        return "Error: Invalid JSON response from Ollama API."

# Step 4: Main loop (now uses the new retrieval function)
if __name__ == "__main__":
    print("Welcome! Ask about the M.Sc. Information Technology program at TH OWL. (Type 'exit' or 'quit' to end.)")
    while True:
        user_query = input("\nYour question: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Exiting program.")
            break
        # Use the new, more intelligent retrieval function
        context = retrieve_passages_semantic(user_query, knowledge_base)
        answer = ollama_generate(user_query, context)
        print(f"\nAI answer: {answer}")
