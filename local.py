# llama_rag_pipeline.py

import weaviate
import ollama
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType

# Your list of documents
documents = [
    "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels",
    "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
    "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall",
    "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
    "Llamas are vegetarians and have very efficient digestive systems",
    "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old",
]

# Connect to local Weaviate instance
client = weaviate.connect_to_local()

# Create a new data collection if it doesn't exist
collection_name = "docs"
if collection_name not in [col.name for col in client.collections.list_all()]:
    collection = client.collections.create(
        name=collection_name,
        properties=[
            Property(name="text", data_type=DataType.TEXT),
        ],
    )
else:
    collection = client.collections.get(collection_name)

# Populate collection with document embeddings
with collection.batch.fixed_size(batch_size=200) as batch:
    for d in documents:
        response = ollama.embeddings(model="all-minilm", prompt=d)
        batch.add_object(
            properties={"text": d},
            vector=response["embedding"],
        )

# Step 1: Retrieve context for the query
query = "What animals are llamas related to?"
query_embedding = ollama.embeddings(model="all-minilm", prompt=query)
results = collection.query.near_vector(near_vector=query_embedding["embedding"], limit=1)
context = results.objects[0].properties["text"]

# Step 2: Augment prompt
augmented_prompt = f"Using this data: {context}. Respond to this prompt: {query}"

# Step 3: Generate response
response = ollama.generate(
    model="tinyllama",
    prompt=augmented_prompt,
)

# Print the final response
print("\nFinal Response:\n")
print(response["response"])
