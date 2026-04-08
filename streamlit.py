from openai import OpenAI
import streamlit
from dataclasses import dataclass
from query import get_response #importing function from other py file
from langchain_community.document_loaders import PyPDFLoader


streamlit.title("RAG Chatbot with Streamlit and OpenAI") #title for the page
#api key setup for openai
client = OpenAI(api_key=streamlit.secrets[("OPENAI_API_KEY")])
print("OpenAI API Key loaded successfully.") #testing prints

if "docs_loaded" not in streamlit.session_state:
    streamlit.session_state.docs_loaded = False

if "openai_model" not in streamlit.session_state:
    streamlit.session_state["openai_model"] = "gpt-3.5-turbo"
print("OpenAI model set to:", streamlit.session_state["openai_model"])

if "messages" not in streamlit.session_state:
    streamlit.session_state.messages = []
print("Session state messages initialized.")

streamlit.session_state.docs_loaded = True
print("Documents loaded and processed:", streamlit.session_state.docs_loaded)


#upload ui
if not streamlit.session_state.docs_loaded:
    uploaded_files = streamlit.file_uploader(label="Upload PDF files", accept_multiple_files=True, type=["pdf"])
    if uploaded_files:
        with streamlit.spinner("Processing..."):
            file_path = "./doc_files/"
            for file in uploaded_files:
                with open(file_path + file.name, "wb") as f:
                    f.write(file.getvalue())
                loader = PyPDFLoader(file_path)
                pages = loader.load_and_split()
            

#chat ui
for message in streamlit.session_state.messages:
    with streamlit.chat_message(message["role"]):
        streamlit.markdown(message["content"])
print("Displayed previous messages.")


if prompt := streamlit.chat_input("What is up?"):
    streamlit.session_state.messages.append({"role": "user", "content": prompt})
    with streamlit.chat_message("user"):
        streamlit.markdown(prompt)
        print("User prompt added to session state messages.")
    with streamlit.chat_message("assistant"):
        content,sources = get_response(prompt)
        streamlit.markdown(content)
        print("Assistant response generated1.")
        with streamlit.expander("sources"):
            streamlit.text(sources)
            print("Assistant response generated.")
    streamlit.session_state.messages.append({"role": "assistant", "content": content})
    print("Assistant response added to session state messages.")



# prompt = streamlit.chat_input(
#     "Say something and/or attach an image",
#     accept_file=True,
#     file_type=["pdf"],
# )
# print("User input received:", prompt)
# if prompt:
    # streamlit.session_state.messages.append({"role": "user", "content": prompt})
    # print("User prompt added to session state messages.")
    # with streamlit.chat_message("user"):
    #     streamlit.markdown(prompt)
    #     print("User prompt displayed in chat.")
    # with streamlit.pdf(prompt):
    #     streamlit.markdown(prompt)
    #     print("User PDF displayed in chat.")
    
     # print("User prompt added to session state messages.")
    # with streamlit.chat_message("assistant"):
    #     content,sources = get_response(prompt)
    #     streamlit.markdown(content)
    #     with streamlit.expander("sources"):
    #         streamlit.text(sources)
    # with streamlit.chat_message("assistant"):
    #     uploaded_files = prompt["files"]
    #     streamlit.markdown(uploaded_files)
    #     with streamlit.chat_message("files"):
    #         uploaded_files = prompt.files
    #         if uploaded_files:
    #             for file in uploaded_files:
    #                 streamlit.image(file)
        
        

