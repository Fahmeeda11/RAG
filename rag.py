#load the data
#import necessary libraries and create env
import os
from xml.dom.minidom import Document
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()
USER_AGENT= os.getenv('USER_AGENT')

##to load data from web pages we can use WebBaseLoader. lanchain provides various document loaders to load data from different sources. 
from langchain_community.document_loaders import WebBaseLoader

#add the url of the web page
DATA_URL = ["https://www.geeksforgeeks.org/nlp/stock-price-prediction-project-using-tensorflow/",
            "https://www.geeksforgeeks.org/deep-learning/training-of-recurrent-neural-networks-rnn-in-tensorflow/"]

#main function 
def main():
    generate_data_stores()

#data store generation function
def generate_data_stores():
    documents = load_page()
    chunks = split_text(documents)  

#function to load data from the web page
def load_page():
    loader = WebBaseLoader(DATA_URL)
    documents = loader.load()
    return documents

#chunking/text splitting
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=50,
    length_function=len,
    add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks




