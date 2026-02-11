
import os
import sys
import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import time

# Setup Paths relative to this script (tools/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")

print(f"Project Root: {PROJECT_ROOT}")
print(f"Static Dir: {STATIC_DIR}")

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

@app.get("/")
async def root(request: Request):
    # Dev Layout Engine
    layout = request.cookies.get("vox_layout", "aurora")
    template_path = f"layouts/{layout}.html"
    if not os.path.exists(os.path.join(TEMPLATES_DIR, "layouts", f"{layout}.html")):
        template_path = "layouts/aurora.html"

    return templates.TemplateResponse(template_path, {
        "request": request, 
        "site_name": "VoxAI Dev",
        "client_name": "Dev User",
        "image_gen_enabled": True,
        "current_layout": layout
    })

@app.post("/api/set_layout")
async def set_layout(request: Request):
    data = await request.json()
    layout = data.get("layout", "standard")
    response = JSONResponse({"status": "success", "layout": layout})
    response.set_cookie(key="vox_layout", value=layout)
    return response

# --- MOCK API ---
@app.post("/chat")
async def mock_chat(request: Request):
    data = await request.json()
    msg = data.get("message", "")
    time.sleep(1) # Simulate think time
    return JSONResponse({"reply": f"**[MOCK AI]** You said: _{msg}_. \n\nI can support Markdown tables:\n| A | B |\n|---|---|\n| 1 | 2 |"})

@app.get("/api/models")
async def mock_models():
    return JSONResponse({
        "models": [
            {"name": "flux1-schnell.safetensors", "family": "flux", "size_gb": 12.4},
            {"name": "sdxl-turbo.safetensors", "family": "sdxl", "size_gb": 6.8},
            {"name": "juggernaut-xl.safetensors", "family": "sdxl", "size_gb": 7.0}
        ],
        "current": "flux1-schnell.safetensors"
    })

@app.post("/api/image/generate")
async def mock_gen(request: Request):
    time.sleep(2)
    return JSONResponse({
        "status": "success",
        "url": "https://dummyimage.com/1024x1024/000/fff&text=Mock+Generation", 
        "filename": "mock_image.png"
    })

if __name__ == "__main__":
    print("Starting UI Dev Server on http://127.0.0.1:8888")
    uvicorn.run(app, host="127.0.0.1", port=8888)
