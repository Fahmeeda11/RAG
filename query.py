import argparse
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import embeddings

load_dotenv()

CHROMA_PATH = os.getenv('CHROMA_PATH')
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

#adding parser
parser = argparse.ArgumentParser()
parser.add_argument("query_text", type=str, help="Query text to search in the documents")
args = parser.parse_args()
query_text = args.query_text

#embeddings
embeddings = embedding_function()

#create vector store using Chroma
db =Chroma(collection_name="documents", embedding_function=embeddings, persist_directory=CHROMA_PATH)

results= db.similarity_search_with_relevance_scores(query_text, k=1)

context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text, question=query_text)
print(prompt)

#generate response using ChatOpenAI
model = ChatOpenAI()
response = model.ainvoke(prompt)
print("Response:", response)
#sources
sources = ("\n".join([doc.metadata["source"] for doc, _ in results])) #get sources from metadata
print("Sources:", sources)
