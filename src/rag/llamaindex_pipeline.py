from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import os

# Path to your data folder
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_path = os.path.join(base_dir, "data")

# Load documents
documents = SimpleDirectoryReader(data_path).load_data()

# Create index
index = VectorStoreIndex.from_documents(documents)

# Create query engine
query_engine = index.as_query_engine()

# Query
response = query_engine.query("Engine temperature is high and coolant level is low")

print("\n--- RESPONSE ---\n")
print(response)