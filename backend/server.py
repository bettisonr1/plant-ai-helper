from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Optional, List, Dict
from .agents.chat_agent import agent


app = FastAPI(title="Plant AI Assistant API")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str
    thread_id: Optional[str] = "default"
    user_plants: Optional[List[str]] = []
    user_details: Optional[Dict] = {}

class ChatResponse(BaseModel):
    content: str
    done: bool
    thread_id: str

@app.post("/api/chat/stream")
async def stream_chat(request: ChatMessage):
    """Stream chat responses from the agent."""
    
    async def generate():
        try:
            latest_message = ""
            
            # Stream agent responses
            for chunk in agent.stream(
                input={
                    "messages": [{"role": "user", "content": request.message}],
                    "user_plants": request.user_plants or [],
                    "user_details": request.user_details or {}
                },
                config={"configurable": {"thread_id": request.thread_id}},
                stream_mode="messages"
            ):
                if isinstance(chunk, tuple) and len(chunk) > 0:
                    content = getattr(chunk[0], 'content', '')
                    if content:
                        latest_message += content
                        # Send SSE event
                        yield f"data: {json.dumps({'content': latest_message, 'done': False, 'thread_id': request.thread_id})}\n\n"
            
            # Send completion event
            yield f"data: {json.dumps({'content': latest_message, 'done': True, 'thread_id': request.thread_id})}\n\n"
            
        except Exception as e:
            error_data = json.dumps({
                'error': str(e),
                'done': True
            })
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)