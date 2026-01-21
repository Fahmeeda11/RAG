#load the data
#import necessary libraries and create env
import argparse
import getpass
import os
import openai
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
#env variables
load_dotenv()
USER_AGENT= os.getenv('USER_AGENT')
ROOT_PATH = os.getenv('SINGLE_FILE_PATH')
OPENAI_API_KEY = getpass.getpass(os.getenv('OPENAI_API_KEY'))
CHROMA_PATH = os.getenv('CHROMA_PATH')
##to load data from web pages we can use WebBaseLoader. lanchain provides various document loaders to load data from different sources. 
from langchain_community.document_loaders import WebBaseLoader

# function to load pdf files from a folder
def load_pdf(DATA_PATH):
    print(f"loading pdf files from {DATA_PATH}...")
    all_docs = [] #list to store all documents
    for file_path in os.listdir(DATA_PATH): #in this folder go over each thing
        if file_path.endswith(".pdf"): # if thing name ends in .pdf
            file_path = os.path.join(DATA_PATH, file_path) #get full path of thing
            
            loaders = PyPDFLoader(file_path) #load the pdf
            all_docs.append( loaders.load()) #append the loaded pdf to the list
            
        else:
            #call the function recursively if there are subfolders
            sub_dir = os.path.join(DATA_PATH, file_path)
            if os.path.isdir(sub_dir):
                all_docs.extend(load_pdf(sub_dir))
    print(f"loaded {len(all_docs)} documents from {DATA_PATH}.")
    return all_docs

#function to split documents into chunks
def split_documents(documents):
    chunks = []
    for doc in documents:
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
        doc_chunks = text_splitter.split_documents(doc)
        chunks.extend(doc_chunks)
    print(f"split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

#function to create embeddings
def embedding_function():
    print("creating embeddings using OpenAI...")
    #create embeddings using OpenAI
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
                )
    print("created embeddings using OpenAI.")
    return embeddings


list_of_docs = load_pdf(ROOT_PATH)
print(f"loaded {len(list_of_docs)} documents from {ROOT_PATH}.")
chunks = split_documents(list_of_docs)

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
db.add_documents(chunks)

results= db.similarity_search_with_relevance_scores(query_text, k=1)

context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text, question=query_text)
print(prompt)

model = ChatOpenAI()
response = model.ainvoke(prompt)
print("Response:", response)
#sources
sources = ("\n".join([doc.metadata["source"] for doc, _ in results]))
print("Sources:", sources)