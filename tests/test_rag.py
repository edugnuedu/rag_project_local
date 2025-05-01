import os
import pytest
import weaviate
from weaviate.classes.init import Auth
from sentence_transformers import SentenceTransformer
from rag_project_local.retrieval import retrieve_documents
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@pytest.fixture(scope="module")
def weaviate_config():
    """Fixture to provide Weaviate configuration."""
    return {
        "url": os.getenv("WEAVIATE_URL"),
        "api_key": os.getenv("WEAVIATE_API_KEY")
    }

@pytest.fixture(scope="module")
def openai_config():
    """Fixture to provide OpenAI configuration."""
    return {
        "api_key": os.getenv("OPENAI_API_KEY")
    }

@pytest.fixture(scope="module")
def weaviate_client(weaviate_config):
    """Fixture to set up and tear down Weaviate client."""
    url = weaviate_config["url"]
    api_key = weaviate_config["api_key"]
    assert url is not None, "WEAVIATE_URL is not set in .env"
    assert api_key is not None, "WEAVIATE_API_KEY is not set in .env"
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=Auth.api_key(api_key)
    )
    yield client
    client.close()

@pytest.fixture(scope="module")
def sample_data():
    """Fixture to provide sample data for ingestion."""
    return [
        {
            "filename": "test_0.txt",
            "content": "Deutsche Telekom launches 5G campus network for RTL Deutschland to support ultra-low latency broadcasting."
        },
        {
            "filename": "test_1.txt",
            "content": "Deutsche Telekom partners with Siemens to deploy IoT solutions for smart cities using NB-IoT technology."
        }
    ]

def test_weaviate_connection(weaviate_config):
    """Test that the Weaviate API key is valid and the connection is successful."""
    url = weaviate_config["url"]
    api_key = weaviate_config["api_key"]
    
    assert url is not None, "WEAVIATE_URL is not set in .env"
    assert api_key is not None, "WEAVIATE_API_KEY is not set in .env"
    
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=Auth.api_key(api_key)
    )
    try:
        assert client.is_ready(), "Failed to connect to Weaviate cluster"
    finally:
        client.close()

def test_openai_connection(openai_config):
    """Test that the OpenAI API key is valid and a simple API call succeeds."""
    api_key = openai_config["api_key"]
    
    assert api_key is not None, "OPENAI_API_KEY is not set in .env"
    
    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test connection"}],
            max_tokens=10
        )
        assert response.choices[0].message.content, "OpenAI API call failed"
    except openai.error.AuthenticationError:
        pytest.fail("Invalid OpenAI API key")
    except openai.error.RateLimitError:
        pytest.fail("OpenAI quota exceeded")

def test_ingestion(weaviate_client, sample_data):
    """Test that documents are ingested into Weaviate with correct properties and vectors."""
    schema_class = "Document"
    
    # Ensure schema exists
    if weaviate_client.collections.exists(schema_class):
        weaviate_client.collections.delete(schema_class)
    weaviate_client.collections.create(
        name=schema_class,
        vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none(),
        properties=[
            weaviate.classes.config.Property(name="filename", data_type=weaviate.classes.config.DataType.TEXT),
            weaviate.classes.config.Property(name="content", data_type=weaviate.classes.config.DataType.TEXT),
        ]
    )

    # Ingest sample data
    model = SentenceTransformer('all-MiniLM-L6-v2')
    collection = weaviate_client.collections.get(schema_class)
    with collection.batch.dynamic() as batch:
        for doc in sample_data:
            embedding = model.encode(doc["content"]).tolist()
            batch.add_object(
                properties={"filename": doc["filename"], "content": doc["content"]},
                vector=embedding
            )

    # Verify ingestion
    response = collection.query.fetch_objects(limit=10)
    assert len(response.objects) == len(sample_data), "Incorrect number of documents ingested"
    filenames = [obj.properties["filename"] for obj in response.objects]
    for doc in sample_data:
        assert doc["filename"] in filenames, f"Document {doc['filename']} not found"
        assert any(obj.vector is not None for obj in response.objects), "Vectors not generated"

def test_retrieval(weaviate_client, sample_data):
    """Test that retrieval returns relevant documents for a query."""
    query = "What is the purpose of the 5G campus network for RTL Deutschland?"
    documents = retrieve_documents(query, top_k=2)
    
    assert len(documents) > 0, "No documents retrieved"
    assert any("RTL Deutschland" in doc["content"] for doc in documents), "Relevant document not retrieved"

