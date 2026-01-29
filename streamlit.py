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
if prompt := streamlit.chat_input("What is up?"):
    streamlit.session_state.messages.append({"role": "user", "content": prompt})
    with streamlit.chat_message("user"):
        streamlit.markdown(prompt)
    print("User prompt added to session state messages.")
    with streamlit.chat_message("assistant"):
        content,sources = get_response(prompt)
        streamlit.markdown(content)
        with streamlit.expander("sources"):
            streamlit.text(sources)
        print("Assistant response generated.")
    streamlit.session_state.messages.append({"role": "assistant", "content": content})
    print("Assistant response added to session state messages.")

