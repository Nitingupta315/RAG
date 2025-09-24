from groq import Groq
from dotenv import load_dotenv
import os
import chromadb

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

def build_from_chroma():
    # Create or load a persistent ChromaDB collection
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    # getting a collection from the database, if it does not exist it will be created
    collection = chroma_client.get_or_create_collection(name="my_collection")

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

    # creates embeddings and adds them to the collection
    # 
    collection.upsert(
        documents=knowledge_base,
        ids=[f"id{i+1}" for i in range(len(knowledge_base))],
    )

    return collection

def retrieve_from_chroma(collection, question, n_results=3):
    results = collection.query(query_texts=[question], n_results=n_results)
    question_context = results["documents"][0][0]
    return question_context

def get_answer_from_groq(question, question_context):
    question_with_context = f"Context: {question_context}\n\nQuestion: {question}"

    client = Groq()

    system_prompt = (
        "You are a helpful assistant providing information about the M.Sc. Information Technology program "
        "at TH OWL based ONLY on the context supplied below. If the answer is not in the context, say you cannot answer based on the provided data.\n\n"
    )

    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": question_with_context},
        ],
    )

    return completion.choices[0].message.content


if __name__ == "__main__":
    print("Welcome! Ask about the M.Sc. Information Technology program at TH OWL. (Type 'exit' or 'quit' to end.)")
    try:
        collection = build_from_chroma()
    except Exception as e:
        print(f"Error building collection: {e}")
        collection = None

    while True:
        try:
            user_query = input("\nYour question: ").strip()
            if not collection:
                print("ChromaDB collection is not available. Cannot process the query.")
                continue
            # Retrieve context from Database
            question_context = retrieve_from_chroma(collection, user_query)
            # Get answer from Groq, passing the retrieved context
            answer = get_answer_from_groq(user_query, question_context)
            # Display the answer
            print(f"\nAI answer: {answer}\n")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if user_query.lower() in {"exit", "quit"}:
            print("Exiting program.")
            break
