import streamlit as st
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
# import logging
import os
import json

# Show title and description.
st.title("💬 Cerebras Chatbot")
st.write(
    "This is a simple chatbot that uses Cerebras' Llama 3.1 model to generate extremely fast responses. "
    "To use this app, you need to provide a Cerebras API key."
)
load_dotenv()

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management


# Create an OpenAI client.
client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))

# Helper to save selection to JSON
def save_selection(prompt, response):
    filename = "selections.json"
    entry = {"prompt": prompt, "response": response}
    
    data = []
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                data = json.load(f)
            except Exception:
                data = []
    
    data.append(entry)
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

# Create a session state variable to store the chat messages. This ensures that the
# messages persist across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_options" not in st.session_state:
    st.session_state.pending_options = None

# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a chat input field to allow the user to enter a message.
# Disable input if we are waiting for a selection.
prompt_input = st.chat_input("What is up?", disabled=st.session_state.pending_options is not None)

if prompt_input:
    # Display the user prompt.
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Generate 2 responses
    options = []
    for i in range(2):
        with st.chat_message("assistant", avatar="🤖"):
            st.caption(f"Option {i+1}")
            stream = client.chat.completions.create(
                model="llama3.1-8b",
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ] + [{"role": "user", "content": prompt_input}],
                stream=True
            )
            def stream_generator():
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content

            response = st.write_stream(stream_generator())
            options.append(response)
    
    # Store in pending state to allow selection
    st.session_state.pending_options = {"prompt": prompt_input, "options": options}
    st.rerun()

# If there's a pending selection, show the options and buttons
if st.session_state.pending_options:
    p = st.session_state.pending_options
    
    # Show the prompt and generated options again (since rerun cleared ephemeral chat messages)
    with st.chat_message("user"):
        st.markdown(p["prompt"])
    
    cols = st.columns(2)
    selection = None
    for i, opt in enumerate(p["options"]):
        with cols[i]:
            with st.chat_message("assistant", avatar="🤖"):
                st.caption(f"Option {i+1}")
                st.markdown(opt)
                if st.button(f"Choose Option {i+1}", key=f"btn_{i}"):
                    selection = opt
    
    if selection:
        # Save to JSON
        save_selection(p["prompt"], selection)
        
        # Add to history
        st.session_state.messages.append({"role": "user", "content": p["prompt"]})
        st.session_state.messages.append({"role": "assistant", "content": selection})
        
        # Clear pending and rerun to show in history
        st.session_state.pending_options = None
        st.rerun()
