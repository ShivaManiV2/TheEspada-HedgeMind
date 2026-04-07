def grade_capital_preservation(initial_value: float, current_value: float, peak_value: float) -> float:
    """
    Task 1: Capital Preservation (Easy)
    Score = 1 - normalized_drawdown
    Returns score guaranteed to be in [0.0, 1.0].
    """
    if peak_value > current_value:
        drawdown = (peak_value - current_value) / peak_value
    else:
        drawdown = 0.0
        
    score = 1.0 - drawdown
    return max(0.0, min(1.0, score))


def grade_profit_maximization(initial_value: float, current_value: float) -> float:
    """
    Task 2: Profit Maximization (Medium)
    Score = normalized_returns
    Maps return of 0% -> 0.0, and 50%+ -> 1.0. 
    """
    returns = (current_value - initial_value) / initial_value
    
    # Normalize returns: assuming a max expected return of 50% for scaling
    MAX_EXPECTED = 0.50
    score = returns / MAX_EXPECTED
    
    return max(0.0, min(1.0, score))


def grade_crisis_management(initial_value: float, current_value: float, peak_value: float) -> float:
    """
    Task 3: Crisis Management (Hard)
    Score = recovery_factor - drawdown_penalty
    """
    returns = (current_value - initial_value) / initial_value
    
    # The recovery_factor is just ending up ok compared to initial value
    # E.g. anything > 0% return is a 1.0 recovery factor
    # But if we're down 20%, recovery factor is much lower
    recovery_factor = 1.0 + returns # If returns = -0.2, factor = 0.8. If returns > 0, factor > 1.
    
    # Drawdown penalty
    if peak_value > current_value:
        drawdown = (peak_value - current_value) / peak_value
    else:
        drawdown = 0.0
        
    drawdown_penalty = drawdown * 0.5 # penalized but leaves room for recovery

    score = recovery_factor - drawdown_penalty
    return max(0.0, min(1.0, score))
