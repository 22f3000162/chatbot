import streamlit as st
import numpy as np
import json
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load .env and OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load NPZ data
data = np.load("discourse_embeddings.npz", allow_pickle=True)
embeddings = data['embeddings']
ids = data['ids']
urls = data['urls']
dates = data['dates']
texts = data['texts']

# Build post map directly from NPZ data
@st.cache_data(show_spinner=False)
def load_post_map():
    post_map = {}
    for i in range(len(ids)):
        post_map[str(ids[i])] = {
            'text': texts[i],
            'url': urls[i],
            'created_at': dates[i]
        }
    return post_map

post_lookup = load_post_map()

# Cosine similarity function using NumPy
def cosine_similarity_np(a, b):
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)

# Search function
def search_similar(query, k=5):
    query_embedding = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    ).data[0].embedding
    query_vector = np.array(query_embedding).reshape(1, -1)
    similarities = cosine_similarity_np(query_vector, embeddings)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    return top_k_indices, similarities[top_k_indices]

# Streamlit Chat UI
st.set_page_config(page_title="📚 Discourse RAG Chatbot", layout="centered")
st.title("📚 Discourse RAG Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask something about course discussions...")

if user_input:
    with st.spinner("Thinking..."):
        top_indices, _ = search_similar(user_input, k=5)
        responses = []
        for idx in top_indices:
            post_id = str(ids[idx])
            meta = post_lookup.get(post_id, {})
            text = meta.get("text", "[Post not found]")
            url = meta.get("url", "")
            created_at = meta.get("created_at", "")
            responses.append((text, url, created_at))

        # Construct prompt
        context_block = "\n\n".join([f"{i+1}. {text}" for i, (text, *_rest) in enumerate(responses)])
        gpt_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": '''You are a helpful and knowledgeable Teaching Assistant for the Business Analytics (BA) course in the BS Degree program at IIT Madras. Your job is to answer student queries based only on the forum discussion content provided. Use simple, clear, and professional language.
                                                Do not make up answers or speculate. If the context doesn’t contain enough information, respond by saying that the question requires more information or encourage the student to refer to the official course materials or raise a query on the forum.
                                                Be concise, courteous, and focused on academic accuracy. Always cite the provided post discussions as the basis of your answers.
                                                '''},
                {"role": "user", "content": f"Question: {user_input}\n\nRelevant context:\n{context_block}"}
            ]
        ).choices[0].message.content

        st.session_state.chat_history.append((user_input, gpt_response, responses))

# Display chat history
for question, answer, sources in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        st.markdown(answer)
        st.markdown("**Sources:**")
        for i, (_text, url, created_at) in enumerate(sources):
            st.markdown(f"{i+1}. [View Post]({url}) — *{created_at}*")
