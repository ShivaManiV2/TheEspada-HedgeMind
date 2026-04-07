from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os

from env.hedge_env import HedgeEnv
from env.state import Observation, EnvState

app = FastAPI(title="HedgeMind Environment API")

# Global environment instance
# In a real deployed Multi-agent setup, you'd manage session IDs.
env_instance = HedgeEnv()

class StepRequest(BaseModel):
    action: Dict[str, float]

class ResetRequest(BaseModel):
    task: Optional[str] = "task_easy"

from fastapi.responses import HTMLResponse
from pathlib import Path

@app.get("/")
def home():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"), status_code=200)
    return {"status": "Welcome to HedgeMind - OpenEnv API is perfectly running!"}

@app.post("/reset")
def reset(req: ResetRequest = None):
    # Depending on the OpenEnv spec, it might send a GET or POST. Allowing POST here.
    task_name = req.task if req else "task_easy"
    obs = env_instance.reset(task_name=task_name)
    return obs.model_dump()

@app.get("/reset")
def reset_get(task: str = "task_easy"):
    # Also support GET /reset for healthchecks/HF constraints
    obs = env_instance.reset(task_name=task)
    return obs.model_dump()

@app.post("/step")
def step(req: StepRequest):
    try:
        obs, reward, done, info = env_instance.step(req.action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    state_obj = env_instance.state()
    return state_obj.model_dump()

def main():
    import uvicorn
    # Default HF Space port is 7860
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
