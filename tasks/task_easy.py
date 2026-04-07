from env.hedge_env import HedgeEnv
from tasks.graders import grade_capital_preservation

class CapitalPreservationTask:
    def __init__(self):
        self.env = HedgeEnv()
        
    def get_env(self):
        return self.env
        
    def reset(self):
        return self.env.reset(task_name="task_easy")
        
    def step(self, action: str):
        return self.env.step(action)
        
    def grade(self) -> float:
        state = self.env.state()
        return grade_capital_preservation(
            initial_value=self.env.initial_cash,
            current_value=state.total_value,
            peak_value=self.env.peak_value
        )
