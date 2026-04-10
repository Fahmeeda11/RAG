import os

import streamlit
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from query import get_response, embedding_function #importing function from other py file to create connection for query response and embedding function

#-- configuration --
#load env variables and init the openai client
load_dotenv()
CHROMA_PATH = os.getenv('CHROMA_PATH') # CHROMA_PATH is shared with rag.py — both read/write to the same vector store.

streamlit.title("Ask your PDFs") #title for the page

client = OpenAI(api_key=streamlit.secrets[("OPENAI_API_KEY")]) #api key setup for openai client

# -- session state initialization --
# Streamlit reruns this entire script on every user interaction.
# Session state persists data (chat history, model choice, upload status) across reruns.
if "docs_loaded" not in streamlit.session_state:
    streamlit.session_state.docs_loaded = False

if "openai_model" not in streamlit.session_state:
    streamlit.session_state["openai_model"] = "gpt-3.5-turbo"
print("OpenAI model set to:", streamlit.session_state["openai_model"])

if "messages" not in streamlit.session_state:
    streamlit.session_state.messages = []
print("Session state messages initialized.")

#-- file upload and embeddings--
#Shown only before documents are loaded. Saves uploaded PDFs to disk
if not streamlit.session_state.docs_loaded:
    uploaded_files = streamlit.file_uploader(label="Upload PDF files", accept_multiple_files=True, type=["pdf"])
    if uploaded_files:
        with streamlit.spinner("Processing..."):
            path = "./doc_files/"
            all_docs = [] #list to store all documents  
            for file in uploaded_files:
                fullpath = path + file.name
                with open(fullpath, "wb") as f:
                   f.write(file.getvalue())
                   print(f"Saved uploaded file: {file.name}")
                   
            loader = PyPDFLoader(fullpath)
            # 800-char chunks with overlap to preserve context across splits
            pages = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80).split_documents(loader.load())
            all_docs.extend(pages) # splits them into chunks, and adds them to the Chroma vector store
            streamlit.session_state.docs_loaded = True
        embeddings = embedding_function()
        db =Chroma(collection_name="documents", embedding_function=embeddings, persist_directory=CHROMA_PATH)
        db.add_documents(all_docs, metadata={"source": file.name}) # Add chunks to the shared Chroma collection for similarity search


#-- chat history--
# Re-render all previous messages so the conversation persists visually across reruns.
for message in streamlit.session_state.messages:
    with streamlit.chat_message(message["role"]):
        streamlit.markdown(message["content"])
print("Displayed previous messages.")

#-- chat input and response generation --
# Takes user question, runs similarity search against the vector store,
# sends matched context + question to agent, and displays the answer with sources.
if prompt := streamlit.chat_input("How can I help you  ?"):
    streamlit.session_state.messages.append({"role": "user", "content": prompt})
    with streamlit.chat_message("user"):
        streamlit.markdown(prompt)
        print("User prompt added to session state messages.")
    with streamlit.chat_message("assistant"):
        # Returns (answer_text, source_documents) from similarity search + agent response
        content,sources = get_response(prompt)
        streamlit.markdown(content)
        print("Assistant response generated1.")
        with streamlit.expander("sources"):
            streamlit.text(sources)
    streamlit.session_state.messages.append({"role": "assistant", "content": content})



