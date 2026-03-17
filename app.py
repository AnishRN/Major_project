import streamlit as st
from transformers import pipeline

# ==============================
# LOAD MODEL
# ==============================

@st.cache_resource
def load_model():
    pipe = pipeline(
        "text-generation",
        model="microsoft/phi-2",
        max_new_tokens=300,
        temperature=0.3
    )
    return pipe

chatbot = load_model()

# ==============================
# STREAMLIT UI
# ==============================

st.set_page_config(page_title="Finance Chatbot", page_icon="💰")

st.title("💰 Finance Chatbot")
st.markdown("Ask any finance-related questions.")

# ==============================
# SESSION STATE
# ==============================

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==============================
# USER INPUT
# ==============================

user_input = st.chat_input("Type your question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Internal prompt (hidden from user)
    prompt = f"""
You are a financial expert.

Answer clearly with proper explanations and calculations where needed.

Question: {user_input}

Answer:
"""

    with st.spinner("Thinking..."):
        response = chatbot(prompt)[0]["generated_text"]

    answer = response.split("Answer:")[-1].strip()

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)

# ==============================
# SIDEBAR
# ==============================

st.sidebar.header("Options")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
