import numpy as np

def calculate_reward(
    current_value: float,
    previous_value: float,
    peak_value: float,
    action: str,
    action_history: list
) -> float:
    """
    Computes a continuous reward signal.
    Formula: (returns * 100) - (drawdown * 20) - risk_penalty
    """
    # 1. Calculate returns
    if previous_value == 0:
        returns = 0.0
    else:
        returns = (current_value - previous_value) / previous_value
        
    return_component = returns * 100.0
    
    # 2. Calculate drawdown
    if peak_value > current_value:
        drawdown = (peak_value - current_value) / peak_value
    else:
        drawdown = 0.0
        
    drawdown_component = drawdown * 20.0
    
    # 3. Calculate risk/behavior penalty
    risk_penalty = 0.0
    
    # Overtrading penalty (penalize massive reallocation swings between steps)
    if len(action_history) >= 2:
        current_alloc = action_history[-1]
        prev_alloc = action_history[-2]
        
        # Calculate sum of absolute differences in weights
        turnover = sum(abs(current_alloc.get(k, 0) - prev_alloc.get(k, 0)) for k in current_alloc.keys())
        
        if turnover > 1.0: 
            risk_penalty += 0.5 # Penalize erratic turnover swings
            
    # Calculate final reward
    reward = return_component - drawdown_component - risk_penalty
    
    return float(reward)
