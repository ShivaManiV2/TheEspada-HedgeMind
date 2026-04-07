import os
import json
import asyncio
from typing import List
from openai import AsyncOpenAI

# The local environment to interact with
from env.hedge_env import HedgeEnv

# Load configuration from environment variables
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Exact logging format wrappers to guarantee strict format
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: dict, reward: float, done: bool, error: str = None):
    err_str = f"{error}" if error else "null"
    action_str = json.dumps(action).replace(" ", "")
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={err_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}", flush=True)

async def run_task(client: AsyncOpenAI, task_name: str):
    log_start(task=task_name, env="HedgeMind", model=MODEL_NAME)
    
    # We use our local environment instance for the baseline script
    env = HedgeEnv()
    obs = env.reset(task_name=task_name)
    
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0
    
    done = False
    
    try:
        # Step through the environment
        for step in range(1, 101):  # Max 100 steps
            if done:
                break
                
            prompt = (
                "You are a hedge fund agent. Based on the observation, output a JSON dictionary assigning allocation weights (0.0 to 1.0) to 5 assets: TECH, FINANCE, ENERGY, HEALTHCARE, BONDS. Example: {'TECH': 0.4, 'FINANCE': 0.4, 'BONDS': 0.2}. Output ONLY valid JSON.\n"
                f"Observation: {json.dumps(obs.model_dump())}"
            )
            
            error = None
            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=50
                )
                raw_action = response.choices[0].message.content.strip()
                
                # Cleanup potential Markdown formatting
                if raw_action.startswith("```json"):
                    raw_action = raw_action[7:]
                if raw_action.endswith("```"):
                    raw_action = raw_action[:-3]
                elif raw_action.startswith("```"):
                    raw_action = raw_action[3:]
                    
                action = json.loads(raw_action.strip())
            except Exception as e:
                action = {"TECH": 0.2, "FINANCE": 0.2, "ENERGY": 0.2, "HEALTHCARE": 0.2, "BONDS": 0.2}
                error = str(e).replace(' ', '_')
                
            obs, reward, done, info = env.step(action)
            
            rewards.append(reward)
            steps_taken = step
            
            log_step(step=step, action=action, reward=reward, done=done, error=error)
            
        # Calculate grade for the task
        from tasks.graders import grade_capital_preservation, grade_profit_maximization, grade_crisis_management
        state = env.state()
        if task_name == "task_easy":
            score = grade_capital_preservation(env.initial_cash, state.total_value, env.peak_value)
        elif task_name == "task_medium":
            score = grade_profit_maximization(env.initial_cash, state.total_value)
        else: # task_hard
            score = grade_crisis_management(env.initial_cash, state.total_value, env.peak_value)
            
        success = score > 0.0
        
    except Exception as e:
        print(f"[DEBUG] Runtime error in task {task_name}: {e}")
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

async def main() -> None:
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "dummy-key-for-local"),
        base_url=API_BASE_URL
    )
    
    for task in ["task_easy", "task_medium", "task_hard"]:
        await run_task(client, task)

if __name__ == "__main__":
    asyncio.run(main())
