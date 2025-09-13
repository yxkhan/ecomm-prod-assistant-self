
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage

from workflow.agentic_rag_workflow import AgenticRAG

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- FastAPI Endpoints ----------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post("/get", response_class=HTMLResponse)
async def chat(msg: str = Form(...)):
    """Call the Agentic RAG workflow."""
    rag_agent = AgenticRAG()
    answer = rag_agent.run(msg)   # run() already returns final answer string
    print(f"Agentic Response: {answer}")
    return answer