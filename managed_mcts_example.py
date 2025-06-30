import os
import random
from dotenv import load_dotenv
from agent import Agent
from MCTS import MCTSNode
from mcts_manager import MCTSManager, ManagedMCTS

# Load environment variables
load_dotenv()

class ResourceGatherNode(MCTSNode):
    GRID_SIZE = 5
    RESOURCES = {'wood': (1, 3), 'stone': (3, 1)}
    GOAL = (4, 4)
    def apply_action(self, action):
        state = self.state.copy()
        pos = state['pos']
        carried = state['carried']
        delivered = state['delivered'][:]
        path = state['path'][:]
        resources = state['resources'].copy()
        if action in ['up', 'down', 'left', 'right']:
            x, y = pos
            if action == 'up':
                y = max(0, y - 1)
            elif action == 'down':
                y = min(self.GRID_SIZE - 1, y + 1)
            elif action == 'left':
                x = max(0, x - 1)
            elif action == 'right':
                x = min(self.GRID_SIZE - 1, x + 1)
            pos = (x, y)
            path.append(pos)
        elif action.startswith('pickup_'):
            res = action.split('_')[1]
            if carried is None and pos == resources[res] and res not in delivered:
                carried = res
        elif action == 'deliver':
            if carried and pos == self.GOAL:
                delivered.append(carried)
                carried = None
        return {
            'pos': pos,
            'carried': carried,
            'delivered': delivered,
            'resources': resources,
            'path': path
        }
        
    def get_possible_actions(self):
        actions = []
        x, y = self.state['pos']
        # Movement
        if y > 0:
            actions.append('up')
        if y < self.GRID_SIZE - 1:
            actions.append('down')
        if x > 0:
            actions.append('left')
        if x < self.GRID_SIZE - 1:
            actions.append('right')
        # Pickup
        for res, loc in self.state['resources'].items():
            if self.state['carried'] is None and self.state['pos'] == loc and res not in self.state['delivered']:
                actions.append(f'pickup_{res}')
        # Deliver
        if self.state['carried'] and self.state['pos'] == self.GOAL:
            actions.append('deliver')
        return actions
    
    def is_terminal_state(self):
        return len(self.state['delivered']) == len(self.state['resources'])
    
    def evaluate_state(self):
        if not self.agent:
            # Heuristic: more delivered, closer to goal, fewer steps
            delivered = len(self.state['delivered'])
            x, y = self.state['pos']
            gx, gy = self.GOAL
            dist = abs(x - gx) + abs(y - gy)
            score = delivered + (1.0 - dist / (2 * (self.GRID_SIZE - 1))) - 0.01 * (len(self.state['path']) - 1)
            return max(0.0, min(2.0, score)) / 2.0
        state_desc = f"Agent at {self.state['pos']} carrying {self.state['carried']}, delivered {self.state['delivered']}, path {self.state['path']}. Goal: deliver all resources to {self.GOAL}."
        response = self.agent.chat(state_desc)
        try:
            import re
            scores = re.findall(r'\b0?\.\d+\b|\b1\.0\b', response)
            if scores:
                return float(scores[0])
        except:
            pass
        return 0.5

def extract_most_visited_path(root):
    path = []
    node = root
    while not node.is_terminal_state() and node.children:
        # Pick child with most visits
        best = max(node.children.values(), key=lambda c: getattr(c, 'visits', 0))
        path.append((best.value, best.state['pos'], best.state['carried'], list(best.state['delivered'])))
        node = best
    return path

def extract_best_terminal_path(root):
    best_path = []
    best_score = -float('inf')
    def dfs(node, path):
        nonlocal best_path, best_score
        if node.is_terminal_state():
            score = getattr(node, 'wins', 0) / getattr(node, 'visits', 1)
            if score > best_score:
                best_score = score
                best_path = path[:]
            return
        for child in node.children.values():
            dfs(child, path + [(child.value, child.state['pos'], child.state['carried'], list(child.state['delivered']))])
    dfs(root, [])
    return best_path

def run_resource_gather_example():
    print("=== Resource Gathering & Delivery MCTS Example ===\n")
    try:
        master_agent = Agent()
        print("Master agent created successfully")
    except Exception:
        master_agent = None
        print("No agent found, using heuristic evaluation.")
    problem_description = """
    Resource Gathering: An agent in a 5x5 grid must collect wood at (1,3) and stone at (3,1), delivering both to (4,4). The agent can only carry one resource at a time. Find the optimal sequence of moves to deliver both resources in the fewest steps.
    """
    manager = MCTSManager(
        master_agent=master_agent,
        problem_description=problem_description,
        max_agents=2
    )
    print("Analyzing problem and designing MCTS structure...")
    config = manager.analyze_problem_and_create_structure()
    print("\nCreating specialized agents...")
    agents = manager.create_specialized_agents(config)
    print("\nMCTS Configuration:")
    manager.print_configuration()
    initial_state = {
        'pos': (0, 0),
        'carried': None,
        'delivered': [],
        'resources': {'wood': (1, 3), 'stone': (3, 1)},
        'path': [(0, 0)]
    }
    print(f"\nInitial State: {initial_state}")
    print(f"\nCreating Managed MCTS...")
    managed_mcts = manager.create_managed_mcts(initial_state, ResourceGatherNode)
    print(f"Available actions: {managed_mcts.root.get_possible_actions()}")
    print(f"\nRunning MCTS search with specialized agents...")
    best_child = managed_mcts.search(iterations=200)
    if best_child:
        print(f"\nRecommended first move: {best_child.value}")
        print(f"\nAction Statistics:")
        stats = managed_mcts.get_action_statistics()
        for action, stat in stats.items():
            print(f"  {action}: visits={stat['visits']}, win_rate={stat['win_rate']:.3f}")
        print(f"\nMCTS Tree Structure (top 2 levels):")
        managed_mcts.print_tree(max_depth=2)
        print(f"\nMost Visited Path (Best Plan):")
        plan = extract_most_visited_path(managed_mcts.root)
        for i, (action, pos, carried, delivered) in enumerate(plan):
            print(f"  Step {i+1}: {action} -> pos={pos}, carried={carried}, delivered={delivered}")
    else:
        print("No best action found.")


def run_managed_mcts_example():
    """Run the managed MCTS example for resource gathering and delivery."""
    print("=== Managed MCTS Resource Gathering & Delivery Example ===\n")
    try:
        # Create master agent
        master_agent = Agent()
        print("Master agent created successfully")
    except KeyError:
        print("AZURE_OPENAI_API_KEY not found in environment variables")
        return
    except Exception as e:
        print(f"Error creating master agent: {e}")
        return
    # Define the resource gathering problem
    problem_description = """
    Resource Gathering: An agent in a 5x5 grid must collect wood at (1,3) and stone at (3,1), delivering both to (4,4). The agent can only carry one resource at a time. Find the optimal sequence of moves to deliver both resources in the fewest steps.
    """
    # Create MCTS Manager
    manager = MCTSManager(
        master_agent=master_agent,
        problem_description=problem_description,
        max_agents=2,
        max_iterations=200
    )
    # Analyze problem and create structure
    print("Analyzing problem and designing MCTS structure...")
    config = manager.analyze_problem_and_create_structure()
    # Create specialized agents
    print("\nCreating specialized agents...")
    agents = manager.create_specialized_agents(config)
    # Print configuration
    print("\nMCTS Configuration:")
    manager.print_configuration()
    # Create initial resource gathering state
    initial_state = {
        'pos': (0, 0),
        'carried': None,
        'delivered': [],
        'resources': {'wood': (1, 3), 'stone': (3, 1)},
        'path': [(0, 0)]
    }
    print(f"\nInitial State: {initial_state}")
    print(f"\nCreating Managed MCTS...")
    managed_mcts = manager.create_managed_mcts(initial_state, ResourceGatherNode)
    print(f"Available actions: {managed_mcts.root.get_possible_actions()}")
    # Run MCTS search
    print(f"\nRunning MCTS search with specialized agents...")
    try:
        best_child = managed_mcts.search(iterations=200)
        if best_child:
            print(f"\nRecommended first move: {best_child.value}")
            print(f"\nAction Statistics:")
            stats = managed_mcts.get_action_statistics()
            for action, stat in stats.items():
                print(f"  {action}: visits={stat['visits']}, win_rate={stat['win_rate']:.3f}")
            print(f"\nMCTS Tree Structure (top 2 levels):")
            managed_mcts.print_tree(max_depth=2)
            print(f"\nMost Visited Path (Best Plan):")
            plan = extract_most_visited_path(managed_mcts.root)
            for i, (action, pos, carried, delivered) in enumerate(plan):
                print(f"  Step {i+1}: {action} -> pos={pos}, carried={carried}, delivered={delivered}")
            print(f"\nBest Full Solution Found:")
            best_full = extract_best_terminal_path(managed_mcts.root)
            if best_full:
                for i, (action, pos, carried, delivered) in enumerate(best_full):
                    print(f"  Step {i+1}: {action} -> pos={pos}, carried={carried}, delivered={delivered}")
            else:
                print("  No complete solution found.")
        else:
            print("No best action found")
    except Exception as e:
        print(f"Error during MCTS search: {e}")
        import traceback
        traceback.print_exc()


def run_simple_managed_example():
    """Run a simple managed example without API calls."""
    print("=== Simple Managed MCTS Example (No API) ===\n")
    class DummyAgent:
        def chat(self, prompt, statement=None, parameters=None):
            # Heuristic: more delivered, closer to goal, fewer steps
            import re
            delivered = 0
            if "delivered" in prompt:
                delivered_match = re.search(r"delivered (\[.*?\])", prompt)
                if delivered_match:
                    delivered = len([x for x in delivered_match.group(1).strip('[]').split(',') if x.strip()])
            pos = (0, 0)
            pos_match = re.search(r"pos=\((\d+), (\d+)\)", prompt)
            if pos_match:
                pos = (int(pos_match.group(1)), int(pos_match.group(2)))
            gx, gy = 4, 4
            dist = abs(pos[0] - gx) + abs(pos[1] - gy)
            steps = 0
            path_match = re.search(r"path (\[.*?\])", prompt)
            if path_match:
                steps = len([x for x in path_match.group(1).strip('[]').split('),') if x.strip()])
            score = delivered + (1.0 - dist / 8.0) - 0.01 * (steps - 1)
            return str(max(0.0, min(1.0, score)))
    master_agent = DummyAgent()
    manager = MCTSManager(
        master_agent=master_agent,
        problem_description="Simple resource gathering problem",
        max_agents=2,
        max_iterations=200
    )
    config = manager._get_default_config()
    manager.mcts_config = config
    manager.agents_pool = {
        "general_strategist": DummyAgent(),
        "explorer": DummyAgent()
    }
    print("Using default configuration")
    manager.print_configuration()
    initial_state = {
        'pos': (0, 0),
        'carried': None,
        'delivered': [],
        'resources': {'wood': (1, 3), 'stone': (3, 1)},
        'path': [(0, 0)]
    }
    managed_mcts = manager.create_managed_mcts(initial_state, ResourceGatherNode)
    print(f"\nRunning simple MCTS search...")
    best_child = managed_mcts.search(iterations=200)
    if best_child:
        print(f"\nRecommended action: {best_child.value}")
        stats = managed_mcts.get_action_statistics()
        for action, stat in stats.items():
            print(f"  {action}: visits={stat['visits']}, win_rate={stat['win_rate']:.3f}")
        print(f"\nMost Visited Path (Best Plan):")
        plan = extract_most_visited_path(managed_mcts.root)
        for i, (action, pos, carried, delivered) in enumerate(plan):
            print(f"  Step {i+1}: {action} -> pos={pos}, carried={carried}, delivered={delivered}")
        print(f"\nBest Full Solution Found:")
        best_full = extract_best_terminal_path(managed_mcts.root)
        if best_full:
            for i, (action, pos, carried, delivered) in enumerate(best_full):
                print(f"  Step {i+1}: {action} -> pos={pos}, carried={carried}, delivered={delivered}")
        else:
            print("  No complete solution found.")
    else:
        print("No best action found.")


if __name__ == "__main__":
    print("Managed MCTS Example\n")
    print("Choose an option:")
    print("1. Run with AzureOpenAI Agents (requires API key)")
    print("2. Run simple example (no API required)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        run_managed_mcts_example()
    elif choice == "2":
        run_simple_managed_example()
    else:
        print("Invalid choice. Running simple example...")
        run_simple_managed_example() 