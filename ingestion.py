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

# Valid file extensions 
VALID_EXTENSIONS = {'.txt'}

# Ingest documents into Weaviate
try:
    # Get all files in the data directory
    files = os.listdir(data_dir)
    valid_files = []
    
    # Filter files by extension and readability
    for filename in files:
        file_path = os.path.join(data_dir, filename)
        # Check if it's a file  and has a valid extension
        if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in VALID_EXTENSIONS:
            try:
                # Test if the file is readable and text-based
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.read(1024)  # Read a small portion to verify
                valid_files.append(filename)
            except (UnicodeDecodeError, IOError) as e:
                print(f"Skipping {filename}: Not a valid text file ({e})")
        else:
            print(f"Skipping {filename}: Invalid file extension or not a file")

    print(f"Found {len(files)} files, {len(valid_files)} valid .txt files to ingest.")

    if not valid_files:
        print("No valid files to ingest.")
        exit(1)

    # Ingest valid files
    collection = client.collections.get(schema_class)
    with collection.batch.dynamic() as batch:
        for filename in valid_files:
            file_path = os.path.join(data_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Generate embedding
                embedding = model.encode(content).tolist()
                # Add to Weaviate batch
                batch.add_object(
                    properties={"filename": filename, "content": content},
                    vector=embedding
                )
                print(f"Ingested {filename}")
            except Exception as e:
                print(f"Error ingesting {filename}: {e}")

    print("Ingestion completed successfully.")
except Exception as e:
    print(f"Error during ingestion: {e}")
    exit(1)
finally:
    client.close()
print("Weaviate client closed.")