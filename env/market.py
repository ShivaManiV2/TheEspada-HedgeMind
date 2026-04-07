import numpy as np
from typing import Dict, List, Any

class MarketSimulator:
    def __init__(self, initial_prices: Dict[str, float] = None, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.assets = ["TECH", "FINANCE", "ENERGY", "HEALTHCARE", "BONDS"]
        if initial_prices is None:
            self.initial_prices = {a: 100.0 for a in self.assets}
        else:
            self.initial_prices = initial_prices
            
        self.current_prices = self.initial_prices.copy()
        self.price_history = {a: [self.current_prices[a]] for a in self.assets}
        
        # Default market conditions
        self.regime = "NORMAL"
        self.base_drift = 0.0001
        self.base_volatility = 0.01

    def set_regime(self, regime: str):
        """Sets the market regime and adjusts drift and volatility."""
        self.regime = regime
        if regime == "BULL":
            self.base_drift = 0.005
            self.base_volatility = 0.015
        elif regime == "BEAR":
            self.base_drift = -0.003
            self.base_volatility = 0.025
        elif regime == "CRASH":
            self.base_drift = -0.05
            self.base_volatility = 0.08
        else: # NORMAL
            self.base_drift = 0.0001
            self.base_volatility = 0.01

    def step(self):
        """Advances the market by one step using Geometric Brownian Motion."""
        # 5% chance to randomly shift regimes mid-episode
        if self.rng.random() < 0.05:
            regimes = ["NORMAL", "BULL", "BEAR", "CRASH"]
            self.set_regime(self.rng.choice(regimes))

        for a in self.assets:
            asset_vol_mod = {"TECH": 1.5, "FINANCE": 1.2, "ENERGY": 1.8, "HEALTHCARE": 0.8, "BONDS": 0.3}[a]
            shock = self.rng.normal(0, 1)
            return_pct = self.base_drift + (self.base_volatility * asset_vol_mod) * shock
            
            # Bonds act oppositely in crash (flight to safety)
            if self.regime == "CRASH" and a == "BONDS":
                return_pct = abs(return_pct)

            self.current_prices[a] = self.current_prices[a] * (1 + return_pct)
            self.current_prices[a] = max(self.current_prices[a], 0.01)
            self.price_history[a].append(self.current_prices[a])
            
        return self.current_prices.copy()

    def get_prices(self, window: int = 10) -> Dict[str, List[float]]:
        """Returns the last 'window' prices for each asset."""
        return {a: self.price_history[a][-window:] for a in self.assets}

    def get_signals(self):
        """Generates synthetic signals based on recent price action (using TECH as proxy for market)."""
        if len(self.price_history["TECH"]) < 5:
            return {"strategy_signal": 0.0, "risk_signal": 0.5, "regime": self.regime}
            
        tech_recent = self.price_history["TECH"][-5:]
        momentum = (tech_recent[-1] - tech_recent[0]) / tech_recent[0]
        strategy_signal = float(np.clip(momentum * 10, -1.0, 1.0))
        
        if len(self.price_history["TECH"]) > 10:
            tech_prices = np.array(self.price_history["TECH"][-10:])
            returns = np.diff(tech_prices) / tech_prices[:-1]
            recent_vol = np.std(returns)
        else:
            recent_vol = 0.0
            
        risk_signal = float(np.clip(recent_vol / 0.05, 0.0, 1.0))

        return {
            "strategy_signal": strategy_signal,
            "risk_signal": risk_signal,
            "regime": self.regime
        }

    def reset(self, initial_prices: dict = None, seed: int = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if initial_prices is not None:
            self.initial_prices = initial_prices
        self.current_prices = self.initial_prices.copy()
        self.price_history = {a: [self.current_prices[a]] for a in self.assets}
        self.set_regime("NORMAL")
