import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# ==============================
# CONFIGURATION
# ==============================

# Replace with your Hugging Face token if needed
HF_TOKEN = "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"

# Recommended finance-capable conversational model
# You can switch models if needed
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# ==============================
# LOAD MODEL
# ==============================

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        token=HF_TOKEN
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        temperature=0.3,
        do_sample=True
    )

    return pipe

chatbot = load_model()

# ==============================
# STREAMLIT UI
# ==============================

st.set_page_config(page_title="Finance Chatbot", page_icon="💰")

st.title("💰 Finance Chatbot")
st.markdown("Ask any finance-related questions (stocks, risk, valuation, etc.)")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask your finance question...")

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Construct prompt (finance-focused instruction)
    system_prompt = (
        "You are a highly knowledgeable financial assistant. "
        "Answer questions related to finance including stocks, risk analysis, valuation, derivatives, and markets. "
        "Explain clearly with examples where possible.\n\n"
    )

    full_prompt = system_prompt + f"User: {user_input}\nAssistant:"

    # Generate response
    with st.spinner("Thinking..."):
        response = chatbot(full_prompt)[0]["generated_text"]

        # Extract only assistant reply
        answer = response.split("Assistant:")[-1].strip()

    # Display assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

# ==============================
# SIDEBAR OPTIONS
# ==============================

st.sidebar.header("Settings")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("""
### Notes:
- Model: Mistral 7B Instruct
- Requires good GPU for local run
- You can switch to API-based inference if needed

### Suggested Alternatives:
- meta-llama/Llama-2-7b-chat-hf
- google/gemma-7b-it
- OpenAI-compatible APIs via HuggingFace endpoints
""")
