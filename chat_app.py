import streamlit as st
from query_data import query_rag

st.title("RAG Chat")

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("What is your question?"):
    # Add user's query to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from RAG system
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = query_rag(prompt)
        content = f"{response['response']}\n\n**Sources:** {', '.join(response['sources']) if response['sources'] else ''}"
        message_placeholder.markdown(content)
    
    # Add assistant's response to chat history
    st.session_state.messages.append({"role": "assistant", "content": content})