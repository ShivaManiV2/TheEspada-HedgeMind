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

# The local OpenEnv HTTP server (used when running inside Docker / HF Space)
ENV_SERVER   = os.environ.get("ENV_SERVER_URL", "http://localhost:7860")

BENCHMARK      = "HedgeMind"
MAX_STEPS      = 100
DEFAULT_ACTION = {"TECH": 0.2, "FINANCE": 0.2, "ENERGY": 0.2, "HEALTHCARE": 0.2, "BONDS": 0.2}


# ── Structured stdout loggers (never remove flush=True) ─────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: dict, reward: float, done: bool, error: str = None) -> None:
    err_str    = str(error) if error else "null"
    action_str = json.dumps(action, separators=(",", ":"))
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={err_str}", flush=True)

def log_end(task: str, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] task={task} success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}", flush=True)
# ────────────────────────────────────────────────────────────────────────────


# ── Grader (mirrors tasks/graders.py — no separate import needed) ────────────
def grade_task(task_name: str, total_value: float, peak_value: float,
               initial_cash: float = 100_000.0) -> float:
    if task_name == "task_easy":
        drawdown = (peak_value - total_value) / peak_value if peak_value > total_value else 0.0
        score = 1.0 - drawdown
    elif task_name == "task_medium":
        returns = (total_value - initial_cash) / initial_cash
        score   = returns / 0.50
    else:  # task_hard
        returns  = (total_value - initial_cash) / initial_cash
        drawdown = (peak_value - total_value) / peak_value if peak_value > total_value else 0.0
        score    = (1.0 + returns) - (drawdown * 0.5)
    return float(max(0.0, min(1.0, score)))


# ── Environment backends ─────────────────────────────────────────────────────

class DirectEnv:
    """Runs HedgeEnv in-process — works standalone without a running HTTP server."""

    def __init__(self):
        from env.hedge_env import HedgeEnv
        self._env = HedgeEnv()

    def reset(self, task_name: str) -> dict:
        obs = self._env.reset(task_name=task_name)
        return obs.model_dump()

    def step(self, action: dict) -> dict:
        obs, reward, done, info = self._env.step(action)
        return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}

    def state(self) -> dict:
        return self._env.state().model_dump()

    @property
    def initial_cash(self) -> float:
        return self._env.initial_cash

    @property
    def peak_value(self) -> float:
        return self._env.peak_value


class HttpEnv:
    """Talks to a running FastAPI env server over HTTP (inside Docker/HF Space)."""

    def reset(self, task_name: str) -> dict:
        resp = requests.post(f"{ENV_SERVER}/reset", json={"task": task_name}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def step(self, action: dict) -> dict:
        resp = requests.post(f"{ENV_SERVER}/step", json={"action": action}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def state(self) -> dict:
        resp = requests.get(f"{ENV_SERVER}/state", timeout=30)
        resp.raise_for_status()
        return resp.json()

    @property
    def initial_cash(self) -> float:
        return 100_000.0

    @property
    def peak_value(self) -> float:
        return 100_000.0  # conservative; grader computes from portfolio_history


def build_env():
    """Return DirectEnv if possible, otherwise fall back to HttpEnv."""
    try:
        env = DirectEnv()
        print("[DEBUG] Using direct in-process HedgeEnv", flush=True)
        return env
    except Exception as exc:
        print(f"[DEBUG] Direct env unavailable ({exc}), falling back to HTTP server", flush=True)
        return HttpEnv()
# ────────────────────────────────────────────────────────────────────────────


async def run_task(client: AsyncOpenAI, env, task_name: str) -> None:
    # ── [START] MUST be the very first print — before anything that can fail ──
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards:     List[float] = []
    steps_taken: int         = 0
    success:     bool        = False
    score:       float       = 0.0

    try:
        obs  = env.reset(task_name)
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            prompt = (
                "You are a hedge fund portfolio manager. "
                "Given the observation below, output a JSON object assigning allocation weights "
                "(0.0-1.0) to exactly these 5 assets: TECH, FINANCE, ENERGY, HEALTHCARE, BONDS. "
                "Weights may sum to <=1.0 (remainder is held as cash). "
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

            # ── Step the environment ─────────────────────────────────────────
            try:
                result = env.step(action)
                obs    = result["observation"]
                reward = float(result["reward"])
                done   = bool(result["done"])
            except Exception as exc:
                reward = 0.0
                done   = False
                error  = (error or "") + f"|env_step_error:{str(exc)[:100]}"

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action, reward=reward, done=done, error=error)

        # ── Grade the completed episode ──────────────────────────────────────
        try:
            state      = env.state()
            total_val  = state.get("total_value", env.initial_cash)
            history    = state.get("portfolio_history", [])
            peak_val   = max([h.get("total_value", 0) for h in history] + [env.initial_cash])
            score      = grade_task(task_name, total_val, peak_val, env.initial_cash)
        except Exception as exc:
            print(f"[DEBUG] grading error ({task_name}): {exc}", flush=True)
            score = float(max(0.0, min(1.0, sum(rewards) / max(len(rewards), 1) + 0.5)))

        success = score > 0.0

    except Exception as exc:
        print(f"[DEBUG] run_task fatal ({task_name}): {exc}", flush=True)
        # Guarantee at least one [STEP] so the validator sees START+STEP+END
        if steps_taken == 0:
            rewards.append(0.0)
            steps_taken = 1
            log_step(step=1, action=DEFAULT_ACTION.copy(), reward=0.0, done=True,
                     error=str(exc).replace(" ", "_")[:200])
        score   = 0.0
        success = False

    finally:
        # ── [END] lives in finally — guaranteed to print no matter what ───────
        log_end(task_name, success, steps_taken, score, rewards)


async def main() -> None:
    try:
        env = build_env()

        # Plain instantiation — do NOT use `async with`, which triggers
        # an httpx internal AttributeError in some validator environments.
        client = AsyncOpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

        for task in ("task_easy", "task_medium", "task_hard"):
            await run_task(client, env, task)

    except Exception as exc:
        print(f"[DEBUG] main error: {exc}", flush=True)
        # Emit valid stub output for every task so the validator never sees silence
        for task in ("task_easy", "task_medium", "task_hard"):
            log_start(task=task, env=BENCHMARK, model=MODEL_NAME)
            log_step(step=1, action=DEFAULT_ACTION.copy(), reward=0.0, done=True,
                     error=str(exc).replace(" ", "_")[:200])
            log_end(task, False, 1, 0.0, [0.0])
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
