import asyncio
from google import genai
from dotenv import load_dotenv
import os 
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

async def main():
    client = genai.Client(api_key=API_KEY) #Replace with your real key

    chat = client.aio.chats.create(
        model='gemini-2.0-flash-thinking-exp-01-21',  # or 
    )
    response = await chat.send_message('What is a car used for?')
    print(response.text)
    response = await chat.send_message('What did you just say before this?')
    print(response.text)

if __name__ == "__main__":
    asyncio.run(main())
