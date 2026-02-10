from openai import OpenAI
import streamlit
from dataclasses import dataclass
from query import get_response #importing function from other py file


streamlit.title("RAG Chatbot with Streamlit and OpenAI") #title for the page
#api key setup for openai
client = OpenAI(api_key=streamlit.secrets[("OPENAI_API_KEY")])
print("OpenAI API Key loaded successfully.") #testing prints



if "openai_model" not in streamlit.session_state:
    streamlit.session_state["openai_model"] = "gpt-3.5-turbo"
print("OpenAI model set to:", streamlit.session_state["openai_model"])

if "messages" not in streamlit.session_state:
    streamlit.session_state.messages = []
print("Session state messages initialized.")

for message in streamlit.session_state.messages:
    with streamlit.chat_message(message["role"]):
        streamlit.markdown(message["content"])
print("Displayed previous messages.")
if prompt := streamlit.chat_input("What is up?", accept_file="directory", file_type=["pdf"]):
    streamlit.session_state.messages.append({"role": "user", "content": prompt})
    with streamlit.chat_message("user"):
        streamlit.markdown(prompt)
if prompt and prompt["files"]:
    streamlit.image(prompt["files"][0])
    # if prompt and prompt["files"]:
    #     streamlit.image(prompt["files"][0])
    print("User prompt added to session state messages.")
    with streamlit.chat_message("assistant"):
        content,sources = get_response(prompt)
        streamlit.markdown(content)
        with streamlit.expander("sources"):
            streamlit.text(sources)
    with streamlit.chat_message("assistant"):
        uploaded_files = prompt["files"]
        streamlit.markdown(uploaded_files)
        # with streamlit.chat_message("files"):
        #     uploaded_files = prompt.files
        #     if uploaded_files:
        #         for file in uploaded_files:
        #             streamlit.image(file)
        
        print("Assistant response generated.")
    streamlit.session_state.messages.append({"role": "assistant", "content": content})
    print("Assistant response added to session state messages.")

