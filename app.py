from fastapi import FastAPI

app = FastAPI()

# Health check endpoint
@app.get("/")
def read_root():
    return {"status": "OK"}

# Your existing /embed endpoint
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class TextRequest(BaseModel):
    text: str

@app.post("/embed")
async def get_embedding(request: TextRequest):
    embedding = model.encode(request.text).tolist()
    return {"embedding": embedding}
