# requirements:
# pip install scikit-learn requests numpy

import requests
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Model name (ensure this is the exact model name in your local Ollama)
MODEL = "deepseek-r1:7b"
OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"  # adjust if different

# -------------------------
# Knowledge base (your KB)
# -------------------------
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

# -------------------------
# Precompute TF-IDF vectors
# -------------------------
if not knowledge_base:
    raise ValueError("Knowledge base is empty. Add passages before starting the retriever.")

vectorizer = TfidfVectorizer().fit(knowledge_base)   # FIT ON KB ONLY (once)
kb_vectors = vectorizer.transform(knowledge_base)    # sparse matrix (n_passages x n_features)

# -------------------------
# Retrieval function
# -------------------------
def retrieve_passages_semantic(query: str, top_n: int = 3, min_score: float = 1e-6):
    """
    Returns a list of (passage, score) tuples ordered by descending score.
    - top_n: maximum number of passages to return
    - min_score: minimum cosine similarity to accept (set >0 to require some overlap)
    """
    query = (query or "").strip()
    if not query:
        return []
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, kb_vectors).flatten()  # shape (n_passages,)
    # sort indices by descending similarity
    sorted_idx = np.argsort(sims)[::-1]
    results = []
    for idx in sorted_idx:
        if len(results) >= top_n:
            break
        score = float(sims[idx])
        if score >= min_score:
            results.append((knowledge_base[idx], score))
    return results

# -------------------------
# Build context string for LLM
# -------------------------
def build_context(retrieved):
    if not retrieved:
        return ""  # let the system message tell the model there's no KB info
    parts = []
    for i, (passage, score) in enumerate(retrieved, start=1):
        parts.append(f"[PASSAGE {i} | score={score:.4f}]\n{passage}")
    return "\n\n---\n\n".join(parts)

# -------------------------
# Robust Ollama generator
# -------------------------
def _extract_ollama_content(obj):
    """
    Try a few common patterns to extract assistant text from a response JSON.
    Fallback returns None.
    """
    if isinstance(obj, dict):
        # pattern: { "message": {"content": "..."} }
        m = obj.get("message")
        if isinstance(m, dict):
            if "content" in m:
                return m["content"]
            # sometimes content is a list
            if isinstance(m.get("content"), list):
                # try to find assistant entry
                for c in m["content"]:
                    if isinstance(c, dict) and c.get("role") == "assistant":
                        return c.get("content")
        # pattern: { "output": "..." }
        if "output" in obj:
            return obj["output"]
        # pattern: { "response": [{"content": "..."}] }
        resp = obj.get("response")
        if isinstance(resp, list) and resp and isinstance(resp[0], dict):
            return resp[0].get("content") or resp[0].get("message") or None
        # openai-like choices
        if "choices" in obj:
            ch = obj["choices"]
            if isinstance(ch, list) and ch:
                c0 = ch[0]
                if isinstance(c0, dict):
                    return c0.get("text") or (c0.get("message") or {}).get("content")
    return None

def ollama_generate(user_query: str, context: str, model: str = MODEL, endpoint: str = OLLAMA_ENDPOINT):
    system_prompt = (
        "You are a helpful assistant providing information about the M.Sc. Information Technology program "
        "at TH OWL based ONLY on the context supplied below. If the answer is not in the context, say you cannot answer based on the provided data.\n\n"
        f"Context:\n{context if context else 'NO CONTEXT AVAILABLE FROM KB.'}"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        "stream": False
    }
    try:
        resp = requests.post(endpoint, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = _extract_ollama_content(data)
        if content is None:
            # fallback: return serialized JSON for debugging
            return f"Warning: Unexpected Ollama response shape. Raw response (truncated): {json.dumps(data)[:2000]}"
        return content
    except requests.RequestException as e:
        return f"Error: Could not connect to Ollama at {endpoint}. Ensure Ollama is running and model '{model}' is available. Details: {e}"
    except ValueError:
        return "Error: Invalid JSON response from Ollama API."

# -------------------------
# CLI main loop
# -------------------------
if __name__ == "__main__":
    print("Welcome! Ask about the M.Sc. Information Technology program at TH OWL. (Type 'exit' or 'quit' to end.)")
    while True:
        try:
            user_query = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if user_query.lower() in {"exit", "quit"}:
            print("Exiting program.")
            break
        retrieved = retrieve_passages_semantic(user_query, top_n=3, min_score=1e-6)
        context = build_context(retrieved)
        answer = ollama_generate(user_query, context)
        print(f"\nAI answer: {answer}\n")
