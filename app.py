import streamlit as st
import requests

# ==============================
# CONFIGURATION
# ==============================

HF_TOKEN = "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"

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# ==============================
# FUNCTION TO QUERY MODEL
# ==============================

def query_hf(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.3,
            "return_full_text": False
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        return f"Error: {response.json()}"

    result = response.json()

    try:
        return result[0]["generated_text"]
    except:
        return str(result)

# ==============================
# STREAMLIT UI
# ==============================

st.set_page_config(page_title="Finance Chatbot", page_icon="💰")

st.title("💰 Finance Chatbot")
st.markdown("Ask finance-related questions (stocks, risk, valuation, etc.)")

# ==============================
# SESSION STATE
# ==============================

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==============================
# USER INPUT
# ==============================

user_input = st.chat_input("Ask your finance question...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    # Finance-focused system prompt
    system_prompt = (
        "You are a professional financial advisor chatbot. "
        "Answer questions related to finance such as stocks, investments, risk, "
        "valuation, derivatives, and markets. "
        "Explain clearly with examples when needed.\n\n"
    )

    # Maintain conversation context (last 5 messages)
    history = ""
    for msg in st.session_state.messages[-5:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    full_prompt = system_prompt + history + "Assistant:"

    # Generate response
    with st.spinner("Thinking..."):
        answer = query_hf(full_prompt)

    # Store response
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

st.sidebar.markdown("""
### Model Info:
- Model: Mistral 7B Instruct (via API)
- No GPU required
- Works on any system

### Tips:
- Ask multi-step finance questions
- Try calculations + theory together
""")
