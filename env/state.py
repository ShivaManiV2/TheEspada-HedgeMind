from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class PortfolioConfig(BaseModel):
    cash: float
    positions: Dict[str, float]

class SignalsConfig(BaseModel):
    strategy_signal: float
    risk_signal: float
    regime: str

class Observation(BaseModel):
    prices: Dict[str, List[float]]
    portfolio: PortfolioConfig
    signals: SignalsConfig

class EnvState(BaseModel):
    step: int
    task_name: str
    action_history: List[Dict[str, float]] = []
    portfolio_history: List[Dict[str, Any]] = []
    current_observation: Optional[Observation] = None
    investor_capital: float
    trust_score: float = 1.0
    num_investors: int = 100
    total_value: float
    reward_history: List[float] = []
