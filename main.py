from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from fastapi.staticfiles import StaticFiles


app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Define the prompt template with placeholders for context and question
template = '''
History percakapan: {context}

Pertanyaan: {question}
Jawaban:
'''

# Initialize the Ollama language model
#phi3:latest
#model = OllamaLLM(model="CognitiveComputations/dolphin-gemma2:latest")

model = OllamaLLM(model="gemma2:2b")

# Create a prompt from the template
prompt = ChatPromptTemplate.from_template(template)

# Combine the prompt and the model into a chain
chain = prompt | model

# Initialize conversation context and history
context = ""
history = []  # This will keep the chat history

@app.get("/", response_class=HTMLResponse)
async def get_chat(request: Request):
    return templates.TemplateResponse("diva.html", {"request": request, "history": history})

@app.post("/chat", response_class=HTMLResponse)
async def handle_chat(request: Request, user_input: str = Form(...)):
    global context, history
    if user_input.lower() == 'berak':
        # Clear history and context when user wants to exit
        context = ""
        history = []
        return templates.TemplateResponse("diva.html", {"request": request, "history": history})
    
    # Get model response
    result = chain.invoke({"context": context, "question": user_input})
    context += f"\nUser: {user_input}\nBot: {result}"

    # Append new conversation to the history
    history.append({"user": user_input, "bot": result})
    return templates.TemplateResponse("diva.html", {"request": request, "history": history})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
