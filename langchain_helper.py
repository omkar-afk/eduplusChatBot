
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
import requests
import os


load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def create_db() -> FAISS:

    with open("dataset.txt", "r", encoding="utf-8") as file:
        document_content = file.read()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    docs = text_splitter.create_documents([document_content])

    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=4):
    

    docs = db.similarity_search(query, k=k)
    context = " ".join([d.page_content for d in docs])



    prompt = """You are an intelligent chatbot designed to assist users with queries about the EduPlus ERP software. Use the following guidelines to respond:
If relevant context is available, generate a clear, accurate, and concise response directly addressing the user's query. If any links are part of the context, highlight them.
If no relevant context is found, politely inform the user that the system is not yet trained on that specific information and suggest consulting the support team or documentation for assistance.
Always maintain a friendly, professional tone and ensure your response is easy to understand.
The user query: """+query+"""
The retrieved context (if any): """+context+"""
Respond based on the provided input, ensuring to highlight any links if included in the context]."""

    
    api_key = os.getenv("GROQ_API_KEY")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "llama3-70b-8192",
        "messages": [{
            "role": "user",
            "content": prompt
        }]
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
    response_json = response.json()
    response_text = response_json['choices'][0]['message']['content']

    return response_text, docs