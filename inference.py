import os
import sys
import json
import asyncio
from typing import List

# ── CRITICAL: Always resolve imports relative to this file ──────────────────
# This makes `python inference.py` work from ANY working directory,
# which is required by the Phase 2 validator.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ────────────────────────────────────────────────────────────────────────────

from openai import AsyncOpenAI

# Load configuration from environment variables
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "dummy-key")

BENCHMARK    = "HedgeMind"
MAX_STEPS    = 100
DEFAULT_ACTION = {"TECH": 0.2, "FINANCE": 0.2, "ENERGY": 0.2, "HEALTHCARE": 0.2, "BONDS": 0.2}


# ── Structured stdout loggers (never remove flush=True) ──────────────────────
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


async def run_task(client: AsyncOpenAI, task_name: str) -> None:
    # log_start MUST be the very first statement — before ANY import that can fail.
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards:     List[float] = []
    steps_taken: int         = 0
    success:     bool        = False
    score:       float       = 0.0

    try:
        # Deferred imports — if these fail, log_start has already printed
        from env.hedge_env import HedgeEnv
        from tasks.graders import (
            grade_capital_preservation,
            grade_profit_maximization,
            grade_crisis_management,
        )

        env = HedgeEnv()
        obs = env.reset(task_name=task_name)
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
                f"Observation: {json.dumps(obs.model_dump())}"
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

            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action, reward=reward, done=done, error=error)

        # ── Grade the completed episode ──────────────────────────────────
        state = env.state()
        if task_name == "task_easy":
            score = grade_capital_preservation(env.initial_cash, state.total_value, env.peak_value)
        elif task_name == "task_medium":
            score = grade_profit_maximization(env.initial_cash, state.total_value)
        else:  # task_hard
            score = grade_crisis_management(env.initial_cash, state.total_value, env.peak_value)

        score   = float(max(0.0, min(1.0, score)))
        success = score > 0.0

    except Exception as exc:
        # Print to stdout (not stderr) so the validator can see it
        print(f"[DEBUG] run_task error ({task_name}): {exc}", flush=True)

    finally:
        # log_end MUST always execute — it lives in finally to guarantee that
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
        # Exit 0 so any [START]/[STEP]/[END] lines already flushed are kept
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
