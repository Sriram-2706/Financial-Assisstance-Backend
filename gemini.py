# import google.generativeai as genai



# client = genai.Client(api_key=API_KEY)

from google import genai
import os
from dotenv import load_dotenv
load_dotenv()

def generate(prompt):

    API_KEY = os.getenv("GEMINI_API_KEY")
    # print(API_KEY)

    # Initialize the client with your API key
    client = genai.Client(api_key=API_KEY)

    # Define your prompt
   
    
    # Generate content using the specified model
    response = client.models.generate_content(
        model="gemini-2.5-pro-exp-03-25",
        contents=prompt
    )

    print(response)
    # Output the response text
    return response.text

if __name__ == "__main__":
    prompt = f"""
    You are a FinBuddy, a friendly, useful financial investment advisor. You answer the questions related to finance, savings, and investments in a friendly tone.
	
	User Question: what is the previous question

    Data Sources Provided:
    ""
    
    Answer to the user query at the maximum 150 words in very concise and point wise manner without mark down format.
    """

    response = generate(prompt)
    print(response)
    