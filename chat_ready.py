import os
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

HF_TOKEN = os.environ.get("HF_TOKEN")

# Load LLM
model_name = "microsoft/Phi-3.5-mini-instruct"
hface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

# def llm_load(huggingface_repo_id):
#     llm = HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id
#         , temperature=0.5
#         , token=HF_TOKEN
#         , max_length=512
#         , task="feature-extraction"
#     )
#     return llm

client = InferenceClient(model=model_name, token=HF_TOKEN)

# Load FAISS vector store
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

def retrieve_fromdocs(query,k=2):
    retriever = db.as_retriever(search_kwargs={'k': k})  # Retrieve top 1 document
    docs = retriever.invoke(query)
    return [doc.page_content for doc in docs]  # Return the best match or empty string if no match

# Define the prompt template
custom_template = """
Use ONLY the following information to answer the user's question.
Do NOT use any knowledge from outside the given context.
If you do not find an answer in the provided context, respond with "I don't know."
Do not make up any information or reference anything beyond the context.

Context: {context}
Question: {query}

Start the answer directly. No small talk, no extra information.
"""

# def set_custom_prompt(custom_prompt_template):
#     prompt = PromptTemplate(template=custom_prompt_template,input_variables={"context","question"})
#     return prompt

prompt_template = PromptTemplate(input_variables=["context", "query"], template=custom_template)

def answer_query(query):
    # Retrieve the best matched document from the context
    context_docs = retrieve_fromdocs(query)  # Get the best matched document
    context = "\n".join(context_docs)
    # Fill the prompt template with the context and query
    prompt = prompt_template.format(context=context, query=query)
    
    # Generate the response based only on the context provided
    response = client.text_generation(prompt, max_new_tokens=512, temperature=0.1)

    return response

# def delete_faiss_db():
#     try:
#         if os.path.exists(DB_FAISS_PATH):
#             os.remove(DB_FAISS_PATH)
#             print("FAISS database cleared.")
#         else:
#             print("FAISS database file not found.")
#     except Exception as e:
#         print(f"Error while deleting FAISS database: {e}")

def clear_faiss_db():
    try:
        # Ensure that the FAISS database is unloaded or reset
        global db
        db = None  # Reset the in-memory FAISS database

        # Now attempt to delete the file
        if os.path.exists(DB_FAISS_PATH):
            os.remove(DB_FAISS_PATH)
            print("FAISS database cleared.")
        else:
            print("FAISS database file not found.")
    except Exception as e:
        print(f"Error while deleting FAISS database: {e}")


# user_query = input("Your question: ")
# response = answer_query(user_query)
# print("Answer:", response)
# def start_conversation():
#     # print("Hello! Ask me anything, and I will try to answer. Type 'exit' to end the conversation.")
    
#     while True:
#         try:
#             # Get user query
#             user_query = input("Your question: ")
            
#             # Check if the user wants to exit
#             if user_query.lower() in ["exit", "quit", "end"]:
#                 print("Goodbye! Have a great day.")
#                 # clear_faiss_db()
#                 db.delete_collection()
#                 break
            
#             # Call answer_query to generate the response based on the query
#             response = answer_query(user_query)
            
#             # Print the response
#             print("Answer:", response)

#         except KeyboardInterrupt:
#             # Gracefully handle a manual interruption (Ctrl+C)
#             print("\nConversation terminated. Goodbye!")
#             db.delete_collection() #clear_faiss_db()
#             break

# # Start the conversation
# start_conversation()