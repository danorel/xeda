import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDINGS_MODEL = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-ada-002")

VECTOR_STORE_HOST = os.getenv("VECTOR_STORE_HOST")
VECTOR_STORE_PORT = int(os.getenv("VECTOR_STORE_PORT", "8000"))
VECTOR_STORE_COLLECTION = os.getenv("VECTOR_STORE_COLLECTION", "xeda")
