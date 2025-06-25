# create_db.py
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Load knowledge base
loader = TextLoader('knowledge.txt')
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Create embeddings and store in Chroma
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma.from_documents(
    texts, embeddings, persist_directory="./brau_db")
print("✅ Vector database created at ./brau_db!")
