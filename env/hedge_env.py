from typing import Dict, Any, Tuple
from env.state import Observation, EnvState, PortfolioConfig, SignalsConfig
from env.market import MarketSimulator
from env.reward import calculate_reward

class HedgeEnv:
    def __init__(self, initial_cash: float = 100000.0):
        self.initial_cash = initial_cash
        self.market = MarketSimulator()
        self.reset("task_easy") # Default task

    def reset(self, task_name: str = "task_easy", seed: int = None) -> Observation:
        self.market.reset(seed=seed)
        
        # Adjust market conditions based on task
        if task_name == "task_easy":
            self.market.set_regime("NORMAL")
        elif task_name == "task_medium":
            self.market.set_regime("BULL")
        elif task_name == "task_hard":
            self.market.set_regime("CRASH")
            
        self.step_count = 0
        self.cash = self.initial_cash
        self.positions = {a: 0.0 for a in self.market.assets}
        self.investor_capital = self.initial_cash
        self.trust_score = 1.0
        self.num_investors = 100
        
        # State tracking
        self.action_history = []
        self.reward_history = []
        self.portfolio_history = [{"step": 0, "total_value": self.initial_cash}]
        self.peak_value = self.initial_cash
        self.task_name = task_name
        
        return self._get_observation()

    def _get_portfolio_value(self) -> float:
        return self.cash + sum(self.positions[a] * self.market.current_prices[a] for a in self.market.assets)

    def _get_observation(self) -> Observation:
        prices = self.market.get_prices()
        signals = self.market.get_signals()
        
        return Observation(
            prices=prices,
            portfolio=PortfolioConfig(cash=self.cash, positions=self.positions),
            signals=SignalsConfig(**signals)
        )

    def step(self, action: Any) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        # Validate action dict. If missing keys, default to current weights.
        if not isinstance(action, dict):
            # Fallback for LLMs outputting HOLD strings or invalid structs
            action = {a: 0.0 for a in self.market.assets}
            
        validated_action = {}
        for a in self.market.assets:
            if a in action and isinstance(action[a], (int, float)):
                validated_action[a] = max(0.0, float(action[a]))
            else:
                validated_action[a] = 0.0
                
        # Normalize weights to sum to 1.0 max (cash is remainder)
        total_weight = sum(validated_action.values())
        if total_weight > 1.0:
            validated_action = {a: w / total_weight for a, w in validated_action.items()}
            total_weight = 1.0

        self.action_history.append(validated_action)
        self.step_count += 1
        
        prev_value = self._get_portfolio_value()
        
        # Market shift
        current_prices = self.market.step()
        
        # Calculate new portfolio value before transaction costs
        current_value_pre_realloc = self.cash + sum(self.positions[a] * current_prices[a] for a in self.market.assets)
        target_cash = current_value_pre_realloc * (1.0 - total_weight)
        
        # Reallocate and calculate transaction costs (overtrading penalty)
        transaction_fees = 0.0
        for a in self.market.assets:
            target_alloc_dollars = current_value_pre_realloc * validated_action[a]
            current_alloc_dollars = self.positions[a] * current_prices[a]
            turnover_dollars = abs(target_alloc_dollars - current_alloc_dollars)
            transaction_fees += turnover_dollars * 0.001 # 0.1% fee
            
            # Update position
            self.positions[a] = target_alloc_dollars / current_prices[a]
            
        # Deduct fees from cash
        self.cash = target_cash - transaction_fees
        
        current_value = self._get_portfolio_value()
        
        if current_value > self.peak_value:
            self.peak_value = current_value
            
        # Advanced Investor Dynamics & Compounding Trust Score
        returns_since_start = (current_value - self.initial_cash) / self.initial_cash
        drawdown_from_peak = (self.peak_value - current_value) / self.peak_value if self.peak_value > 0 else 0
        
        # Update Trust Score
        if drawdown_from_peak > 0.05:
            self.trust_score *= 0.95 # lose trust rapidly on drawdowns
        elif returns_since_start > 0:
            self.trust_score = min(1.0, self.trust_score * 1.01) # regain slowly
            
        if self.step_count % 10 == 0:
            if self.trust_score > 0.8 and returns_since_start > 0.05:
                # Investors flock in
                new_investors = int(self.num_investors * 0.1)
                self.num_investors += new_investors
                inflow = self.investor_capital * 0.1
                self.cash += inflow
                self.investor_capital += inflow
            elif self.trust_score < 0.5 or returns_since_start < -0.05:
                # Investors panic sell
                loss = int(self.num_investors * 0.2)
                self.num_investors = max(1, self.num_investors - loss)
                outflow = self.investor_capital * 0.2
                if self.cash >= outflow:
                    self.cash -= outflow
                    self.investor_capital -= outflow
                else:
                    # Forced liquidation constraint (simplified for env boundary)
                    self.investor_capital -= self.cash
                    outflow = self.cash
                    self.cash = 0.0
                    
        current_value_after_investors = self._get_portfolio_value()
        
        # Calculate Reward
        reward = calculate_reward(
            current_value_after_investors,
            prev_value,
            self.peak_value,
            validated_action,
            self.action_history
        )
        self.reward_history.append(reward)
        self.portfolio_history.append({
            "step": self.step_count,
            "total_value": current_value_after_investors
        })
        
        done = self.step_count >= 100 or current_value_after_investors < (self.initial_cash * 0.2)
        
        info = {
            "portfolio_value": current_value_after_investors,
            "returns": returns_since_start,
            "trust_score": self.trust_score,
            "num_investors": self.num_investors
        }
        
        return self._get_observation(), reward, done, info

    def state(self) -> EnvState:
        return EnvState(
            step=self.step_count,
            task_name=self.task_name,
            action_history=self.action_history,
            current_observation=self._get_observation(),
            investor_capital=self.investor_capital,
            trust_score=self.trust_score,
            num_investors=self.num_investors,
            total_value=self._get_portfolio_value(),
            reward_history=self.reward_history,
            portfolio_history=self.portfolio_history
        )
