#import necessary libraries and create env
import getpass
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

#env variables
load_dotenv()
USER_AGENT= os.getenv('USER_AGENT')
ROOT_PATH = os.getenv('SINGLE_FILE_PATH')
OPENAI_API_KEY = getpass.getpass(os.getenv('OPENAI_API_KEY'))
CHROMA_PATH = os.getenv('CHROMA_PATH')

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


#calling load_pdf function with variable.
list_of_docs = load_pdf(ROOT_PATH)
print(f"loaded {len(list_of_docs)} documents from {ROOT_PATH}.")

#calling split_documents function with chunks variable
chunks = split_documents(list_of_docs)

#embeddings 
embeddings = embedding_function()

#create vector store using Chroma
db =Chroma(collection_name="documents", embedding_function=embeddings, persist_directory=CHROMA_PATH)
db.add_documents(chunks)
print("added documents to Chroma vector store:", db)
