# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Initialize FastAPI app and load the model
app = FastAPI()
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Define a request body schema
class TextRequest(BaseModel):
    text: str

@app.post("/embed")
async def get_embedding(request: TextRequest):
    # Compute the embedding for the given text
    embedding = model.encode(request.text).tolist()
    return {"embedding": embedding}