from env.hedge_env import HedgeEnv
from tasks.graders import grade_profit_maximization

class ProfitMaximizationTask:
    def __init__(self):
        self.env = HedgeEnv()
        
    def get_env(self):
        return self.env
        
    def reset(self):
        return self.env.reset(task_name="task_medium")
        
    def step(self, action: str):
        return self.env.step(action)
        
    def grade(self) -> float:
        state = self.env.state()
        return grade_profit_maximization(
            initial_value=self.env.initial_cash,
            current_value=state.total_value
        )
