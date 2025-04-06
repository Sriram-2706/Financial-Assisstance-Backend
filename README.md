Backend

To run the backend, do the following

1.  Activate the anvironment by:
cd backend
cd your-python-venv/Souce/activate

2. install the dependencies
pip install -r requirements.txt

3. setup .env file, include GEMINI_API_KEY

4. Make sure  allow_dangerous_deserialization=True  in faiss.py, to open serialized vectorDB files