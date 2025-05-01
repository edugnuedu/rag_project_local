import os
from dotenv import load_dotenv
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.connect import ConnectionParams
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from weaviate.classes.init import Auth


# Load environment variables
load_dotenv()

weaviate_url = os.environ["WEAVIATE_URL"]
weaviate_api_key = os.environ["WEAVIATE_API_KEY"]
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_api_key),
)

print(client.is_ready())

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_documents(query: str, top_k: int = 3) -> List[Dict]:
    """
    Retrieve the top-k most similar documents from Weaviate based on the query.
    
    Args:
        query (str): The user's question.
        top_k (int): Number of documents to retrieve (default: 3).
    
    Returns:
        List[Dict]: List of documents with 'filename' and 'content'.
    """
    try:
        # Generate query embedding
        query_embedding = model.encode(query).tolist()
        
        # Query Weaviate for top-k similar documents
        collection = client.collections.get("Document")
        response = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_properties=["filename", "content"]
        )
        
        # Extract documents from the response
        documents = [
            {"filename": obj.properties["filename"], "content": obj.properties["content"]}
            for obj in response.objects
        ]
        return documents
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return []
    finally:
        client.close()
