from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Health check endpoint
@app.get("/")
def read_root():
    return {"status": "OK"}

# Define request schema for embedding
class TextRequest(BaseModel):
    text: str

# Load your embedding model once at startup
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@app.post("/embed")
async def get_embedding(request: TextRequest):
    embedding = model.encode(request.text).tolist()
    return {"embedding": embedding}
