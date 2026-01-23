from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.helper import get_embedding
from src.prompts import system_prompt
from store import docs

# -------------------------------------------------
# App setup
# -------------------------------------------------
app = Flask(__name__)

# Load environment variables
try:
    load_dotenv()
except:
    # load_dotenv may fail in certain contexts (like exec), try manual loading
    try:
        with open('.env', 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    except:
        pass

# Set Google API key
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
else:
    print("Warning: GOOGLE_API_KEY not found in environment")

# -------------------------------------------------
# Vectorstore & Retriever
# -------------------------------------------------
embedding = get_embedding()

retriever = docs.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# -------------------------------------------------
# LLM & RAG Chain (Lazy Loading)
# -------------------------------------------------
rag_chain = None

def get_rag_chain():
    global rag_chain
    if rag_chain is None:
        print("Initializing RAG chain...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.6
        )
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}")
            ]
        )
        
        rag_chain = (
            {
                "context": retriever,
                "input": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        print("RAG chain initialized!")
    return rag_chain

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")

    if not msg:
        return jsonify({"error": "No message provided"}), 400

    chain = get_rag_chain()
    response = chain.invoke(msg)

    return jsonify({"answer": response})

# -------------------------------------------------
if __name__ == "__main__":
    print("Starting Flask app on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
