from src.gitcontext.embeddings import EmbeddingModel

embedding_model = EmbeddingModel()

documents = [
    "def verify_password(password, stored_password):",
    "def connect_database():",
    "def calculate_salary(employee):"
]

document_vectors = embedding_model.embed_documents(documents)

print("Number of documents:", len(document_vectors))
print("Vector dimensions:", document_vectors.shape)

query = "Where is password verification implemented?"

query_vector = embedding_model.embed_query(query)

print("Query vector dimensions:", query_vector.shape)