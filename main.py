from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from retrieval import retrieve_documents
from llm_assignment.Archive.generation import generate_answer

import uvicorn

# Initialize FastAPI app and templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def index(request: Request):
    """Serve the main page with the question form."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query")
async def query(request: Request, question: str = Form(...)):
    """
    Process the user's question and return an answer.
    
    Args:
        question (str): The user's question from the form.
    
    Returns:
        TemplateResponse: Rendered HTML with question and answer.
    """
    # Retrieve relevant documents
    documents = retrieve_documents(question)
    
    # Generate the answer
    answer = generate_answer(question, documents)
    
    # Render the response
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "question": question, "answer": answer}
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )