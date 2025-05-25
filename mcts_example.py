import os
from dotenv import load_dotenv
from agent import Agent
from MCTS import MCTS, MCTSNode
import random

# Load environment variables
load_dotenv()


class DecisionProblemNode(MCTSNode):
    """
    Example MCTS node for a simple decision-making problem.
    """
    
    def apply_action(self, action):
        """Apply an action to get the next state."""
        new_state = self.state.copy()
        new_state['decisions'].append(action)
        new_state['step'] += 1
        
        # Simulate some consequences of the action
        if action == 'invest':
            new_state['resources'] += random.randint(-10, 20)
        elif action == 'save':
            new_state['resources'] += random.randint(0, 5)
        elif action == 'research':
            new_state['knowledge'] += random.randint(5, 15)
        elif action == 'market':
            new_state['reputation'] += random.randint(-5, 10)
        
        return new_state
    
    def get_possible_actions(self):
        """Get possible actions from current state."""
        if self.state['step'] >= 5:  # Max 5 decisions
            return []
        
        actions = ['invest', 'save', 'research', 'market']
        
        # Filter actions based on current state
        if self.state['resources'] < 10:
            actions.remove('invest')  # Can't invest without resources
        
        return actions
    
    def is_terminal_state(self):
        """Check if this is a terminal state."""
        return self.state['step'] >= 5 or len(self.get_possible_actions()) == 0
    
    def evaluate_state(self):
        """Evaluate the current state."""
        if not self.agent:
            # Simple heuristic evaluation
            score = (self.state['resources'] * 0.3 + 
                    self.state['knowledge'] * 0.4 + 
                    self.state['reputation'] * 0.3) / 100
            return max(0.0, min(1.0, score))
        
        # Use agent for evaluation
        state_description = f"""
        Current business state:
        - Resources: {self.state['resources']}
        - Knowledge: {self.state['knowledge']}
        - Reputation: {self.state['reputation']}
        - Decisions made: {self.state['decisions']}
        - Step: {self.state['step']}/5
        
        Evaluate how good this business position is on a scale of 0.0 to 1.0.
        Consider long-term sustainability and growth potential.
        """
        
        response = self.reason_with_role(state_description)
        
        try:
            import re
            scores = re.findall(r'\b0?\.\d+\b|\b1\.0\b', response)
            if scores:
                return float(scores[0])
        except:
            pass
        
        return 0.5


class BusinessDecisionMCTS(MCTS):
    """
    MCTS for business decision making.
    """
    
    def __init__(self, agent):
        initial_state = {
            'resources': 50,
            'knowledge': 20,
            'reputation': 30,
            'decisions': [],
            'step': 0
        }
        
        role_prompt = """You are an expert business strategist and decision maker. 
        You understand market dynamics, resource management, and long-term business planning.
        Analyze each situation carefully and provide strategic insights."""
        
        super().__init__(
            root_state=initial_state,
            agent=agent,
            role_prompt=role_prompt,
            max_iterations=500
        )
    
    def create_node(self, value=None, parent=None, state=None):
        """Create a new decision problem node."""
        return DecisionProblemNode(
            value=value,
            parent=parent,
            agent=self.root.agent,
            role_prompt=self.root.role_prompt,
            state=state
        )


def run_business_decision_example():
    """Run the business decision MCTS example."""
    print("=== Business Decision Making with MCTS ===\n")
    
    # Create agent
    try:
        agent = Agent(api_key=os.environ["OPENAI_API_KEY"])
        print("Agent created successfully")
    except KeyError:
        print("OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in a .env file")
        return
    except Exception as e:
        print(f"Error creating agent: {e}")
        return
    
    # Create MCTS
    mcts = BusinessDecisionMCTS(agent)
    print("MCTS initialized")
    
    print(f"\nInitial state: {mcts.root.state}")
    
    # Override the root node's methods
    mcts.root.__class__ = DecisionProblemNode
    mcts.root.possible_actions = mcts.root.get_possible_actions()
    mcts.root.untried_actions = mcts.root.possible_actions.copy()
    
    print(f"Possible actions: {mcts.root.possible_actions}")
    
    # Run MCTS search
    print("\nRunning MCTS search...")
    try:
        best_child = mcts.search(iterations=100)  # Reduced for faster demo
        
        if best_child:
            print(f"\nBest action found: {best_child.value}")
            print(f"Action statistics:")
            
            stats = mcts.get_action_statistics()
            for action, stat in stats.items():
                print(f"  {action}: visits={stat['visits']}, win_rate={stat['win_rate']:.3f}")
            
            print(f"\nMCTS Tree (top 2 levels):")
            mcts.print_tree(max_depth=2)
            
        else:
            print("No best action found")
            
    except Exception as e:
        print(f"Error during MCTS search: {e}")


def run_simple_example():
    """Run a simple example without OpenAI API."""
    print("=== Simple MCTS Example (No API Required) ===\n")
    
    # Create MCTS without agent (will use random evaluation)
    initial_state = {
        'resources': 50,
        'knowledge': 20,
        'reputation': 30,
        'decisions': [],
        'step': 0
    }
    
    mcts = MCTS(
        root_state=initial_state,
        agent=None,  # No agent - will use random evaluation
        role_prompt="Simple decision maker",
        max_iterations=200
    )
    
    # Override root node class
    mcts.root.__class__ = DecisionProblemNode
    mcts.root.possible_actions = mcts.root.get_possible_actions()
    mcts.root.untried_actions = mcts.root.possible_actions.copy()
    
    print(f"Initial state: {mcts.root.state}")
    print(f"Possible actions: {mcts.root.possible_actions}")
    
    print("\nRunning MCTS search (random evaluation)...")
    best_child = mcts.search(iterations=200)
    
    if best_child:
        print(f"\nBest action found: {best_child.value}")
        
        stats = mcts.get_action_statistics()
        for action, stat in stats.items():
            print(f"  {action}: visits={stat['visits']}, win_rate={stat['win_rate']:.3f}")
        
        print(f"\nMCTS Tree (top 2 levels):")
        mcts.print_tree(max_depth=2)


if __name__ == "__main__":
    print("MCTS Agent Example\n")
    print("Choose an option:")
    print("1. Run with OpenAI Agent (requires API key)")
    print("2. Run simple example (no API required)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        run_business_decision_example()
    elif choice == "2":
        run_simple_example()
    else:
        print("Invalid choice. Running simple example...")
        run_simple_example() 