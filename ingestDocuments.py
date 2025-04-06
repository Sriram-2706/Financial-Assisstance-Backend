import os
import faiss
import pickle
from langchain_community.vectorstores import FAISS
# from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Path to the folder containing text, PDF, and CSV files
DATA_FOLDER = "docSample"
VECTOR_DB_PATH = "vectorDB"

# Function to clean text and handle encoding issues
def clean_text(text):
    return text.encode("utf-8", "ignore").decode("utf-8")

# Function to load and extract text from documents
def load_documents(data_folder):
    documents = []
    for file in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file)
        print(file_path)
        try:
            if file.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
            elif file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.endswith(".csv"):
                loader = CSVLoader(file_path, encoding="utf-8")
            else:
                continue  # Skip unsupported files
            doc_data = loader.load()
            for doc in doc_data:
                doc.page_content = clean_text(doc.page_content)
            documents.extend(doc_data)
            break
        except Exception as e:
            print(f"Error loading {file}: {e}")
    return documents

# Load documents
documents = load_documents(DATA_FOLDER)

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
print("len: ", len(docs))
# Initialize embeddings
print("Loading Embedding Moel..")
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print("Embedding Model Loaded")

print("Creating FAISS Vector DB Instance")
# Create FAISS vector database
vector_db = FAISS.from_documents(docs, embeddings)

# Save the vector database
os.makedirs(VECTOR_DB_PATH, exist_ok=True)
vector_db.save_local(VECTOR_DB_PATH)
print("Saved Faiss Vector DB to local disk")

# Function to query the vector database without using LLM
def query_vector_db(query, vector_db_path, top_k=5):
    # Load FAISS index
    vector_db = FAISS.load_local(vector_db_path, embeddings)
    retriever = vector_db.as_retriever()
    
    # Retrieve relevant documents
    docs = retriever.invoke(query, k=top_k)
    
    # Return retrieved documents
    return [doc.page_content for doc in docs]

# Example query
query = "In how much time with the one time password is usable and active by user?"
response = query_vector_db(query, VECTOR_DB_PATH)
print("Relevant Documents:", response)