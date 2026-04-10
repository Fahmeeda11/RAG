import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# --- Configuration ---
# CHROMA_PATH points to the persisted vector store on disk,
load_dotenv()
CHROMA_PATH = os.getenv('CHROMA_PATH') # shared with rag.py (writes) and streamlit.py (reads/writes via upload).

# --- Embedding Setup ---
#function to create embeddings
def embedding_function():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large" # Must match the model used in rag.py, otherwise similarity search breaks.
                ) # Uses OpenAI's text-embedding-3-large model for both ingestion and search. 
    print("created embeddings using OpenAI.")
    return embeddings


# --- Prompt Template ---
# Constrains agent to answer using only the retrieved context, reducing hallucinations.
# the {context} placeholder gets filled with matched document chunks from similarity search.
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


# #embeddings fuction call
embeddings = embedding_function()

# --- Vector Store ---
# Loaded once at import time so every call to get_response() reuses the same connection instead of reopening the store.
db =Chroma(collection_name="documents", embedding_function=embeddings, persist_directory=CHROMA_PATH)

# --- Query & Response ---
 # Main function called by streamlit.py. Finds the most relevant chunk via similarity search, builds a grounded prompt, and returns
#(response_text, source_file_names).
def get_response(query): 
    results= db.similarity_search_with_relevance_scores(query, k=1) # k=1 returns only the single best-matching chunk — increase for broader context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)
    model = ChatOpenAI()
    response = model.invoke(prompt)
    # Extract source filenames so the user can verify the answer
    sources = ("\n".join([doc.metadata["source"] for doc, _ in results]))
    return [response.content,sources]