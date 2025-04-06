from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from langchain_huggingface import HuggingFaceEmbeddings
from getData import getRelevantDocs
from gemini import generate
from dotenv import load_dotenv
from investmentModel import build_investment_plan

# 🔹 Load environment variables
load_dotenv()

print("🔄 Loading Embedding Model...")

# 🔹 Initialize Hugging Face Embeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print("✅ Embedding Model Loaded")

print("🔄 Initializing Google Gemini AI...")

# 🔹 Securely Load API Key
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("❌ Missing GEMINI_API_KEY in .env file!")

# # 🔹 Configure Google Gemini API
# genai.configure(api_key=API_KEY)
# gemini_model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")

# print("✅ Gemini AI Initialized")

# 🔹 Initialize Flask
app = Flask(__name__)
CORS(app)

PORT = 8080

# ✅ Function to Generate Prompt
def getPrompt(userQuery, relevantDocs):
    return f"""
    You are FinBuddy, a friendly, useful financial investment advisor. You answer questions related to finance, savings, and investments in a friendly tone.
    If the provided data source contains useful and related information, use it for your response. If not, generate your own answer.

    User Question: {userQuery}

    Data Sources Provided:
    {relevantDocs}

    Answer the user query in a concise bullet-point format (Max 150 words). No markdown, no greetings.
    """

# ✅ Route 1: Knowledgebase Query Handler
@app.route("/datastore", methods=["POST"])
def datastore():
    request_data = request.get_json()
    query = request_data.get("userQuery", "").strip()

    if not query:
        return jsonify({"error": "Missing 'userQuery' parameter!"}), 400

    print(f"📩 Received Query: {query}")

    # 🔹 Get Relevant Docs
    relevantData = getRelevantDocs(query, embeddings, top_k=2)

    # 🔹 Generate AI Response
    llm_prompt = getPrompt(query, relevantData)

    try:
        # response = gemini_model.generate_content(llm_prompt)
        response = generate(llm_prompt)
        # llm_response = response.text.strip() if response else "No response generated."
        llm_response = response if response else "No response Generated"

    except Exception as e:
        llm_response = f"⚠️ Error generating response: {str(e)}"

    print(f"🤖 AI Response: {llm_response}")

    return jsonify({
        "answer": llm_response
    })


# ✅ Route 2: Investment Plan Generator
@app.route("/investment-plan", methods=["POST"])
def investment_plan():
    try:
        data = request.get_json()
        required = ["income", "expenses", "current_investment", "risk_tolerance", "investment_horizon"]

        # 🔍 Validate Inputs
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # 💰 Extract Parameters
        income = float(data["income"])
        expenses = float(data["expenses"])
        current_investment = float(data["current_investment"])
        risk_tolerance = data["risk_tolerance"]
        investment_horizon = int(data["investment_horizon"])

        # 🧠 Build Investment Plan
        plan = build_investment_plan(
            income, expenses, current_investment, risk_tolerance, investment_horizon
        )

        return jsonify(plan), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ Boot the Flask Server
if __name__ == "__main__":
    app.run(debug=True, port=PORT)
