#load the data
#import necessary libraries and create env
import os
from dotenv import load_dotenv
load_dotenv()
USER_AGENT= os.getenv('USER_AGENT')

##to load data from web pages we can use WebBaseLoader. lanchain provides various document loaders to load data from different sources. 
from langchain_community.document_loaders import WebBaseLoader

#add the url of the web page
DATA_URL = ["https://www.geeksforgeeks.org/nlp/stock-price-prediction-project-using-tensorflow/",
            "https://www.geeksforgeeks.org/deep-learning/training-of-recurrent-neural-networks-rnn-in-tensorflow/"]

#function to load data from the web page
def load_page():
    loader = WebBaseLoader(DATA_URL)
    page = loader.load()
    print("test")
    return page




