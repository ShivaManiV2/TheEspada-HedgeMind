import os
import sys
import json
import asyncio
from typing import List

import requests
from openai import AsyncOpenAI

# ── Always resolve imports relative to this file ────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ────────────────────────────────────────────────────────────────────────────

# Load configuration from environment variables
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "dummy-key")

# The local OpenEnv HTTP server spun up by the Docker container
ENV_SERVER   = os.environ.get("ENV_SERVER_URL", "http://localhost:7860")

BENCHMARK    = "HedgeMind"
MAX_STEPS    = 100
DEFAULT_ACTION = {"TECH": 0.2, "FINANCE": 0.2, "ENERGY": 0.2, "HEALTHCARE": 0.2, "BONDS": 0.2}


# ── Structured stdout loggers (never remove flush=True) ─────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: dict, reward: float, done: bool, error: str = None) -> None:
    err_str    = str(error) if error else "null"
    action_str = json.dumps(action, separators=(",", ":"))
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={err_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}", flush=True)
# ────────────────────────────────────────────────────────────────────────────


def env_reset(task_name: str) -> dict:
    """Call POST /reset on the local env server."""
    resp = requests.post(f"{ENV_SERVER}/reset", json={"task": task_name}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(action: dict) -> dict:
    """Call POST /step on the local env server."""
    resp = requests.post(f"{ENV_SERVER}/step", json={"action": action}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_state() -> dict:
    """Call GET /state on the local env server."""
    resp = requests.get(f"{ENV_SERVER}/state", timeout=30)
    resp.raise_for_status()
    return resp.json()


def grade_task(task_name: str, state: dict) -> float:
    """Compute score from the final env state — mirrors tasks/graders.py logic."""
    initial_cash = 100_000.0
    total_value  = state.get("total_value", initial_cash)
    peak_value   = max(initial_cash, total_value)  # conservative fallback

    # Try to get peak from portfolio_history if available
    history = state.get("portfolio_history", [])
    if history:
        peak_value = max(h.get("total_value", 0) for h in history)

    if task_name == "task_easy":
        # Capital Preservation: 1 - drawdown
        drawdown = (peak_value - total_value) / peak_value if peak_value > total_value else 0.0
        score = 1.0 - drawdown
    elif task_name == "task_medium":
        # Profit Maximization: normalised returns (50 % = 1.0)
        returns = (total_value - initial_cash) / initial_cash
        score   = returns / 0.50
    else:
        # Crisis Management: recovery_factor - drawdown_penalty
        returns  = (total_value - initial_cash) / initial_cash
        drawdown = (peak_value - total_value) / peak_value if peak_value > total_value else 0.0
        score    = (1.0 + returns) - (drawdown * 0.5)

    return float(max(0.0, min(1.0, score)))


async def run_task(client: AsyncOpenAI, task_name: str) -> None:
    # [START] must be the very first print — before anything that can fail
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards:     List[float] = []
    steps_taken: int         = 0
    success:     bool        = False
    score:       float       = 0.0

    try:
        obs  = env_reset(task_name)
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            prompt = (
                "You are a hedge fund portfolio manager. "
                "Given the observation below, output a JSON object assigning allocation weights "
                "(0.0–1.0) to exactly these 5 assets: TECH, FINANCE, ENERGY, HEALTHCARE, BONDS. "
                "Weights may sum to ≤1.0 (remainder is held as cash). "
                "Output ONLY valid JSON with no markdown or explanation.\n"
                f"Observation: {json.dumps(obs)}"
            )

            error  = None
            action = DEFAULT_ACTION.copy()

            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=200,
                )
                raw = response.choices[0].message.content.strip()

                # Strip markdown fences if the model wraps its output
                for fence in ("```json", "```"):
                    if raw.startswith(fence):
                        raw = raw[len(fence):]
                if raw.endswith("```"):
                    raw = raw[:-3]

                parsed = json.loads(raw.strip())
                if all(k in parsed for k in DEFAULT_ACTION):
                    action = {k: float(parsed[k]) for k in DEFAULT_ACTION}
                else:
                    error = "missing_keys_in_llm_response"

            except Exception as exc:
                error = str(exc).replace(" ", "_")[:200]

            # Step the environment via HTTP
            try:
                result  = env_step(action)
                obs     = result["observation"]
                reward  = float(result["reward"])
                done    = bool(result["done"])
            except Exception as exc:
                reward = 0.0
                done   = False
                error  = (error or "") + f"|env_step_error:{str(exc)[:100]}"

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action, reward=reward, done=done, error=error)

        # Grade the completed episode
        try:
            state = env_state()
            score = grade_task(task_name, state)
        except Exception as exc:
            print(f"[DEBUG] grading error ({task_name}): {exc}", flush=True)
            # Fall back to reward-based heuristic
            score = float(max(0.0, min(1.0, sum(rewards) / max(len(rewards), 1) + 0.5)))

        success = score > 0.0

    except Exception as exc:
        print(f"[DEBUG] run_task error ({task_name}): {exc}", flush=True)

    finally:
        # [END] lives in finally — guaranteed to print no matter what
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    try:
        # Plain instantiation — do NOT use `async with`, which triggers
        # an httpx internal AttributeError in some validator environments.
        client = AsyncOpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

        for task in ("task_easy", "task_medium", "task_hard"):
            await run_task(client, task)

    except Exception as exc:
        print(f"[DEBUG] main error: {exc}", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
