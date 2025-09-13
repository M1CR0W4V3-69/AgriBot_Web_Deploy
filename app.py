import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import gradio as gr

# --- LOAD KNOWLEDGE BASE ---
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# --- USE GEMINI (FREE!) ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3
)

# --- PROMPT TEMPLATE ---
template = """
You are AgriBot, a friendly expert agriculture assistant for Indian farmers.
Answer the question using ONLY the context below.
Keep answers:
- Simple, clear, step-by-step
- Practical and actionable
- Encouraging and patient 😊

If unsure, say: “I’m not 100% sure, but here’s general advice: [...]”

Context:
{context}

Question: {question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# --- BUILD CHAIN ---
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- GRADIO INTERFACE ---
def respond(message, history):
    return chain.invoke(message)

demo = gr.ChatInterface(
    respond,
    title="🌾 AgriBot — Ask Me Anything About Farming!",
    description="No install needed. Works on phone, tablet, laptop. Made for Indian farmers.",
    examples=[
        "What crops grow in summer?",
        "How to use neem oil for aphids?",
        "What is PM-KISAN scheme?",
        "How to improve soil fertility?",
        "Best fish to farm in Andhra Pradesh?"
    ],
    theme="soft"
)

# --- LAUNCH ---
if __name__ == "__main__":
    demo.launch()