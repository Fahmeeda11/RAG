import os
from dotenv import load_dotenv
from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import embeddings

load_dotenv()
CHROMA_PATH = os.getenv('CHROMA_PATH') #loading path

#function to create embeddings
def embedding_function():
    print("creating embeddings using OpenAI...")
    #create embeddings using OpenAI
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
                )
    print("created embeddings using OpenAI.")
    return embeddings


#define prompt template
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


# #embeddings call
embeddings = embedding_function()

# #create vector store using Chroma
db =Chroma(collection_name="documents", embedding_function=embeddings, persist_directory=CHROMA_PATH)

# function to get response
def get_response(query): 
    results= db.similarity_search_with_relevance_scores(query, k=1)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)
    model = ChatOpenAI()
    response = model.invoke(prompt)
    #sources
    sources = ("\n".join([doc.metadata["source"] for doc, _ in results])) #get sources from metadata
    return [response.content,sources]