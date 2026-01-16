#load the data
#import necessary libraries and create env
import os
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#env variables
load_dotenv()
USER_AGENT= os.getenv('USER_AGENT')

##to load data from web pages we can use WebBaseLoader. lanchain provides various document loaders to load data from different sources. 
from langchain_community.document_loaders import WebBaseLoader

#add the url of the web page
DATA_URL = ["https://www.geeksforgeeks.org/nlp/stock-price-prediction-project-using-tensorflow/",
            "https://www.geeksforgeeks.org/deep-learning/training-of-recurrent-neural-networks-rnn-in-tensorflow/"]

#main function 
def main():
    chunks = generate_data_stores()    
    #create embeddings without any api key
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    #used vector store using FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3}
    )

    query = "what is recurrent neural network?"
    docs = retriever.invoke(query)

    print(docs)


#data store generation function
def generate_data_stores():
    documents = load_page()
    chunks = split_text(documents)
    return chunks  

#function to load data from the web page
def load_page():
    loader = WebBaseLoader(DATA_URL)
    return loader.load()

#chunking/text splitting
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=50,
    )
    return text_splitter.split_documents(documents)

#call the main function
if __name__ == "__main__":
    main()


