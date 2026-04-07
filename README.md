---
title: TheEspada-HedgeMind
emoji: 📈
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
---
# HedgeMind — A Real-World Benchmark for Financial AI Agents

HedgeMind is a multi-factor hedge fund simulation environment for OpenEnv. It simulates market conditions (trend, volatility, regime shifts), constraints of a financial portfolio, risk management signals, and realistic investor actions (inflow based on good performance, outflow based on bad performance).

The environment challenges an AI agent across three progressively difficult market scenarios to test decision-making under uncertainty, market crashes, and bull runs.

## Observation Space
The agent receives a unified `Observation` (Pydantic model) containing:
- **prices**: `List[float]` - The last 10 asset prices.
- **portfolio**: Tracking of `cash` (`float`) and `position` (`int`).
- **signals**: Synthetic market signals including `strategy_signal` (momentum proxy), `risk_signal` (volatility proxy), and the current `regime` (NORMAL, BULL, BEAR, CRASH).

## Action Space
Discrete Enum representing basic trading:
- **BUY**: Allocates maximum affordable cash to positions.
- **SELL**: Liquidates entire position for cash.
- **HOLD**: Maintains the current portfolio allocation.

## Tasks
1. **task_easy (Capital Preservation)**: In a normal market, the agent is judged purely on minimizing drawdowns.
2. **task_medium (Profit Maximization)**: In a bull market, the agent is challenged to maximize long-term returns.
3. **task_hard (Crisis Management)**: In a simulated crash environment, the agent must preserve capital during a sudden drop and recover value thereafter.

## Setup Instructions

### Local Execution (Python)
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the OpenEnv server interface (FastAPI):
   ```bash
   python server/app.py
   ```

### Docker
1. Build the image:
   ```bash
   docker build -t hedgemind .
   ```
2. Run the image:
   ```bash
   docker run -p 7860:7860 hedgemind
   ```

## Baseline Evaluation / Inference
To test an OpenAI or HF agent against the HedgeMind environment, simply execute the inference baseline script:

```bash
export OPENAI_API_KEY="your-key"
python inference.py
```

The inference script conforms perfectly to the strict `[START]`, `[STEP]`, and `[END]` evaluation logs over 100 simulation episodes per task, recording the rewards. 
