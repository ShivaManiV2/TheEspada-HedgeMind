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
# HedgeMind V2 — Advanced Multi-Asset Market & Investor Simulation

HedgeMind V2 is a high-fidelity, highly complex multi-asset hedge fund simulation environment built explicitly for the Meta PyTorch OpenEnv Hackathon. 

Rather than simple "toy" environments that accept basic Buy/Sell signals, HedgeMind forces AI agents to continuously manage a 5-asset portfolio against stochastic market shocks, compounding investor behavior, and strict transaction costs.

## 🌟 What makes HedgeMind V2 unique?

1. **Continuous 5-Asset Engine**: Agents dictate float percentage weights across `TECH`, `FINANCE`, `ENERGY`, `HEALTHCARE`, and `BONDS` instead of a standard `BUY/SELL` discrete array.
2. **Markov-Chain Regime Shifts**: Instead of a "static" market, Geometric Brownian Motion drift and volatility modifiers dynamically shift mid-episode. A Bull market can suddenly collapse into a Flash Crash, shocking the AI agent.
3. **Behavioral Investor Trust System**: Massive drawdowns decay an intrinsic "Trust Score," leading to aggressive compounding AUM (Assets Under Management) outflows from simulated retail investors.
4. **Overtrading Penalties**: High-frequency erratic turnover behavior is aggressively penalized mathematically via basis-point transaction fees, directly bleeding portfolio value if the agent ping-pongs between allocations.
5. **Real-time Glassmorphism Dashboard**: A premium, web-based UI API visualizer to track all 5 asset allocations and trust multipliers live during inference.

## Observation Space (`Dict`)
The agent receives a heavily detailed unified `Observation` dictionary:
- **prices**: `Dict[str, List[float]]` - Historic arrays of the last 10 prices dedicated per asset.
- **portfolio**: Tracking of `cash` (`float`) alongside exact equity `positions` (`Dict[str, int]`).
- **signals**: Synthetic market signals including `strategy_signal` (momentum proxy), `risk_signal` (volatility proxy), and the live Markov `regime`.

## Action Space (`Dict`)
Agents must output a continuous allocation dictionary representing target portfolio weights summing to 1.0. Example output format string:
```json
{"TECH": 0.3, "FINANCE": 0.2, "ENERGY": 0.1, "HEALTHCARE": 0.2, "BONDS": 0.2}
```

## Tasks
1. **task_easy (Capital Preservation)**: Judged heavily on minimizing drawdowns and keeping retail investor trust intact.
2. **task_medium (Profit Maximization)**: Maximizing absolute returns during favorable alpha regimes.
3. **task_hard (Crisis Management)**: Surviving and navigating stochastic flash-crash probability spikes mid-episode without hemorrhaging total AUM.

## Setup Instructions

### Docker (Hackathon Default)
1. Build the image:
   ```bash
   docker build -t hedgemind .
   ```
2. Run the image:
   ```bash
   docker run -p 7860:7860 hedgemind
   ```

## Baseline Evaluation / Inference
To test an OpenAI or HF agent against the HedgeMind environment, simply execute the inference baseline script against the API logic:

```bash
export OPENAI_API_KEY="your-api-key"
export HF_TOKEN="your-hf-token"
python inference.py
```

The inference script conforms strictly to the `[START]`, `[STEP]`, and `[END]` evaluation logs over the validation simulation episodes, accurately translating raw LLM generation into valid continuous dictionaries utilizing standard LLM token ingestion.
