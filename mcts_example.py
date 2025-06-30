import os
from dotenv import load_dotenv
from agent import Agent
from MCTS import MCTS, MCTSNode
import random

# Load environment variables
load_dotenv()

class GridWorldNode(MCTSNode):
    """
    MCTS node for a simple 3x3 grid navigation problem.
    State: {'pos': (x, y), 'path': [(x0, y0), ...]}
    """
    GRID_SIZE = 3
    GOAL = (2, 2)

    def apply_action(self, action):
        x, y = self.state['pos']
        path = self.state['path'][:]
        if action == 'up':
            y = max(0, y - 1)
        elif action == 'down':
            y = min(self.GRID_SIZE - 1, y + 1)
        elif action == 'left':
            x = max(0, x - 1)
        elif action == 'right':
            x = min(self.GRID_SIZE - 1, x + 1)
        new_pos = (x, y)
        if new_pos not in path:
            path.append(new_pos)
        return {'pos': new_pos, 'path': path}

    def get_possible_actions(self):
        x, y = self.state['pos']
        actions = []
        if y > 0:
            actions.append('up')
        if y < self.GRID_SIZE - 1:
            actions.append('down')
        if x > 0:
            actions.append('left')
        if x < self.GRID_SIZE - 1:
            actions.append('right')
        return actions

    def is_terminal_state(self):
        return self.state['pos'] == self.GOAL or len(self.state['path']) > self.GRID_SIZE * self.GRID_SIZE

    def evaluate_state(self):
        if not self.agent:
            # Heuristic: closer to goal and shorter path is better
            x, y = self.state['pos']
            gx, gy = self.GOAL
            dist = abs(x - gx) + abs(y - gy)
            score = 1.0 - (dist / (2 * (self.GRID_SIZE - 1))) - 0.05 * (len(self.state['path']) - 1)
            return max(0.0, min(1.0, score))
        # Use agent for evaluation
        state_desc = f"Robot at {self.state['pos']} with path {self.state['path']}. Goal: {self.GOAL}. Score closer to 1.0 if at goal in fewer steps."
        response = self.reason_with_role(state_desc)
        try:
            import re
            scores = re.findall(r'\b0?\.\d+\b|\b1\.0\b', response)
            if scores:
                return float(scores[0])
        except:
            pass
        return 0.5

class GridWorldMCTS(MCTS):
    def __init__(self, agent=None):
        initial_state = {'pos': (0, 0), 'path': [(0, 0)]}
        role_prompt = "You are a robot navigating a 3x3 grid from (0,0) to (2,2). Choose moves to reach the goal in as few steps as possible."
        super().__init__(root_state=initial_state, agent=agent, role_prompt=role_prompt, max_iterations=100)
        self.root.__class__ = GridWorldNode
        self.root.possible_actions = self.root.get_possible_actions()
        self.root.untried_actions = self.root.possible_actions.copy()


def run_gridworld_example():
    print("=== GridWorld MCTS Example ===\n")
    try:
        agent = Agent(api_key=os.environ["AZURE_OPENAI_API_KEY"])
        print("Agent created successfully")
    except Exception:
        agent = None
        print("No agent found, using heuristic evaluation.")
    mcts = GridWorldMCTS(agent)
    print(f"Initial state: {mcts.root.state}")
    print(f"Possible actions: {mcts.root.possible_actions}")
    print("\nRunning MCTS search...")
    best_child = mcts.search(iterations=50)
    if best_child:
        print(f"\nBest first move: {best_child.value}")
        print(f"Action statistics:")
        stats = mcts.get_action_statistics()
        for action, stat in stats.items():
            print(f"  {action}: visits={stat['visits']}, win_rate={stat['win_rate']:.3f}")
        print(f"\nMCTS Tree (top 2 levels):")
        mcts.print_tree(max_depth=2)
    else:
        print("No best action found.")

if __name__ == "__main__":
    run_gridworld_example() 