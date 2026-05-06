import os
import sys
import json
from contextlib import asynccontextmanager
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from backend.agent import build_agent, extract_context_from_messages

# Initialize globals
app_state = {
    "mcp_client": None,
    "agent": None,
    "sessions": {}
}

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str
    context: Dict[str, Any]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup MCP client
    docs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))
    
    # Initialize the MCP Client with the Filesystem server configuration
    mcp_client = MultiServerMCPClient({
        "filesystem": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "-y", 
                "@modelcontextprotocol/server-filesystem", 
                docs_path
            ]
        }
    })
    
    # Connect and get tools
    # MultiServerMCPClient initialization might just setup the connection, wait for it
    # No connect() method, we can just call get_tools()
    extra_tools = await mcp_client.get_tools()
    print(f"Loaded {len(extra_tools)} MCP tools.")
    
    app_state["mcp_client"] = mcp_client
    app_state["agent"] = build_agent(extra_tools)
    
    yield
    
    # Cleanup
    # await mcp_client.close()

app = FastAPI(lifespan=lifespan)

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Loadshare RCA Agent</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        #chat { border: 1px solid #ccc; height: 400px; overflow-y: scroll; padding: 10px; margin-bottom: 10px; }
        .msg { margin-bottom: 10px; padding: 10px; border-radius: 5px; }
        .user { background-color: #e3f2fd; text-align: right; }
        .agent { background-color: #f5f5f5; }
        #input-area { display: flex; }
        #msg-input { flex-grow: 1; padding: 10px; }
        button { padding: 10px 20px; }
        #context { font-size: 0.9em; color: #666; margin-top: 10px; }
        pre { white-space: pre-wrap; font-family: monospace; }
    </style>
</head>
<body>
    <h1>Loadshare RCA Agent</h1>
    <div id="chat"></div>
    <div id="input-area">
        <input type="text" id="msg-input" placeholder="Ask about store performance (e.g., 'How did Bangalore do on 2026-04-22?')">
        <button onclick="sendMessage()">Send</button>
    </div>
    <div id="context">Context: None</div>

    <script>
        const sessionId = Math.random().toString(36).substring(7);
        const chatDiv = document.getElementById('chat');
        const inputField = document.getElementById('msg-input');
        const contextDiv = document.getElementById('context');

        inputField.addEventListener('keypress', function (e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        async function sendMessage() {
            const msg = inputField.value.trim();
            if (!msg) return;

            appendMessage('You', msg, 'user');
            inputField.value = '';

            const loadingId = appendMessage('Agent', 'Thinking...', 'agent');

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sessionId, message: msg })
                });
                const data = await response.json();
                
                // Remove loading message
                document.getElementById(loadingId).remove();
                
                appendMessage('Agent', data.reply, 'agent');
                contextDiv.innerText = 'Context: ' + JSON.stringify(data.context);
            } catch (err) {
                document.getElementById(loadingId).remove();
                appendMessage('System', 'Error: ' + err.message, 'agent');
            }
        }

        function appendMessage(sender, text, className) {
            const div = document.createElement('div');
            div.className = 'msg ' + className;
            const id = 'msg-' + Math.random().toString(36).substring(7);
            div.id = id;
            div.innerHTML = `<strong>${sender}:</strong> <pre>${text}</pre>`;
            chatDiv.appendChild(div);
            chatDiv.scrollTop = chatDiv.scrollHeight;
            return id;
        }
    </script>
</body>
</html>
"""

@app.get("/")
def get_index():
    return HTMLResponse(content=HTML_CONTENT)

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id
    user_msg = req.message
    
    if session_id not in app_state["sessions"]:
        app_state["sessions"][session_id] = {
            "messages": [],
            "context": {}
        }
        
    session = app_state["sessions"][session_id]
    
    # 1. Add user message
    session["messages"].append(HumanMessage(content=user_msg))
    
    # 2. Update context
    session["context"] = extract_context_from_messages(session["messages"], session["context"])
    
    # 3. Invoke agent
    agent = app_state["agent"]
    state_input = {
        "messages": session["messages"],
        "context": session["context"]
    }
    
    # Use await if astream/ainvoke is supported, else sync (invoke is sync in agent.py but wait, we are in an async route)
    # langgraph supports .ainvoke
    try:
        result = await agent.ainvoke(state_input, {"recursion_limit": 25})
    except Exception as e:
        # fallback to sync invoke if ainvoke isn't working
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, agent.invoke, state_input, {"recursion_limit": 25})
        
    session["messages"] = result["messages"]
    
    final_message = session["messages"][-1].content
    
    # Simple check if the LLM output is dict (sometimes gemini returns tool calls or dicts)
    if isinstance(final_message, list):
        # Extract text from list of dicts if needed
        texts = [m.get("text", "") for m in final_message if isinstance(m, dict) and "text" in m]
        if texts:
            final_message = "\n".join(texts)
        else:
            final_message = str(final_message)
            
    return ChatResponse(reply=final_message, context=session["context"])

