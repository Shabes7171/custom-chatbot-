from flask import Flask, request, jsonify
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Flask App
app = Flask(__name__)

# Step 1: Extract Data from Web
loader = WebBaseLoader(["https://brainlox.com/courses/category/technical"])
documents = loader.load()

# Step 2: Create Embeddings and Store in FAISS
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()

# Step 3: Create Retrieval-based QA Chain
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=OPENAI_API_KEY), retriever=retriever)

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chatbot queries via API."""
    user_input = request.json.get("question")
    if not user_input:
        return jsonify({"error": "No question provided."}), 400
    
    response = qa_chain.run(user_input)
    return jsonify({"answer": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
