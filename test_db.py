# test_db.py
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Load vector database
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(persist_directory="./brau_db", embedding_function=embeddings)

# Perform a sample query
query = "Signature Brow"
results = vectordb.similarity_search(query, k=2)

# Print results
print(f"Query: {query}")
for i, doc in enumerate(results):
    print(f"Result {i+1}:")
    print(doc.page_content)
    print("-" * 50)

print("✅ Vector database query successful!")
