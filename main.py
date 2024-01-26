from fastapi import FastAPI
from bot import training_model, handle_message
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
cv,sc,tags =training_model()

class Message(BaseModel):
    message: str

@app.get("/")
def get_message():
    return {"message": 'Welcome to our bot'}

@app.post("/")
def post_message(req: Message):
    return {"message": handle_message(req.message,cv,sc,tags)}
