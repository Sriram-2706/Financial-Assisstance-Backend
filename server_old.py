from flask import Flask, render_template, request, jsonify
import os
from flask_cors import CORS
from random import randint
from langchain_huggingface import HuggingFaceEmbeddings
from gemini import generate
from getData import getRelevantDocs
from google import genai
from dotenv import load_dotenv
load_dotenv()


print("Loading Embedding Model...")
# Initialize embeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print("Embedding Model Loaded")

app = Flask(__name__) 
CORS(app)

PORT = 8080

def getPrompt(userQuery, relevantDocs):
	llm_prompt = f"""
    You are a FinBuddy, a friendly, useful financial investment advisor. You answer the questions related to finance, savings, and investments in a friendly tone.
	If the provided data source contains useful and related information, use it for your response. iF not give your own response
	User Question: {userQuery}

    Data Sources Provided:
	{relevantDocs}

    Answer to the user query at the maximum 150 words in very concise and bullet point wise manner without mark down format. Include user answer alone in your response
    """
	return llm_prompt
	

@app.route("/datastore", methods = ["POST"])
def datastore():
	request_data = request.get_json()
	query = request_data["userQuery"]
	print(query) 
	
    # get relevant docs
	relevantData = getRelevantDocs(query, embeddings, top_k = 2)
	
    #prompt
	llm_prompt = getPrompt(query, relevantData)
	
    # Generate advice
	llm_response = generate(llm_prompt)
	print(llm_response)
		
	return jsonify({
		"answer" : llm_response
    })

if __name__ == "__main__": 
	app.run(debug=True, port = PORT)