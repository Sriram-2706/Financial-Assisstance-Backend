"""
To query documents from the Vector DB Instance
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Function to query the vector database without using LLM
def query_vector_db(query, vector_db_path, embeddings, top_k = 1):
    # Load FAISS index
    vector_db = FAISS.load_local(vector_db_path, embeddings)
    retriever = vector_db.as_retriever()
    
    # Retrieve relevant documents
    docs = retriever.invoke(query, k=top_k)
    
    # Return retrieved documents
    return [doc.page_content for doc in docs]

def getRelevantDocs(prompt, embeddings, top_k = 1):
    VECTOR_DB_PATH = "vectorDB"
    documentList = query_vector_db(prompt, VECTOR_DB_PATH, embeddings, top_k = top_k)
    documentsStr = ""
    for i in range(len(documentList)):
        documentsStr = documentsStr + "Data Source " + str(i + 1) + ": \n"
        documentsStr = documentsStr + documentList[i] + "\n"
    return documentsStr


if __name__ == "__main__":
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    ans = getRelevantDocs("How will I get OTP", embeddings, 3)
    print(ans)

