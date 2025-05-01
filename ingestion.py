import os
from dotenv import load_dotenv
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.connect import ConnectionParams
from weaviate.classes.config import Configure, Property, DataType
from sentence_transformers import SentenceTransformer
from weaviate.classes.init import Auth


# Load environment variables
load_dotenv()

# Set up Weaviate client with environment variables
weaviate_url = os.environ["WEAVIATE_URL"]
weaviate_api_key = os.environ["WEAVIATE_API_KEY"]
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_api_key),
)

print(client.is_ready())

# Create schema if it doesn't exist
schema_class = "Document"
if not client.collections.exists(schema_class):
    client.collections.create(
        name=schema_class,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="filename", data_type=DataType.TEXT),
            Property(name="content", data_type=DataType.TEXT),
        ]
    )

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Directory containing press releases
data_dir = os.getenv("DATA_DIR")

# Ingest documents into Weaviate
try:
    files = os.listdir(data_dir)
    print(f"Found {len(files)} files to ingest.")
    collection = client.collections.get(schema_class)
    with collection.batch.dynamic() as batch:
        for filename in files:
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Generate embedding
            embedding = model.encode(content).tolist()
            # Add to Weaviate batch
            batch.add_object(
                properties={"filename": filename, "content": content},
                vector=embedding
            )
    print("Ingestion completed successfully.")
except Exception as e:
    print(f"Error during ingestion: {e}")
    exit(1)
finally:
    client.close()
print("Weaviate client closed.")