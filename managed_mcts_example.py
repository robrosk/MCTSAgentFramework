import os
import random
from dotenv import load_dotenv
from agent import Agent
from MCTS import MCTSNode
from mcts_manager import MCTSManager, ManagedMCTS

# Load environment variables
load_dotenv()


class BusinessStrategyNode(MCTSNode):
    """
    MCTS node for business strategy decisions.
    """
    
    def apply_action(self, action):
        """Apply a business action to get the next state."""
        new_state = self.state.copy()
        new_state['quarter'] += 1
        new_state['decisions'].append(action)
        
        # Simulate consequences of different actions
        if action == 'product_development':
            new_state['product_quality'] += random.randint(10, 25)
            new_state['resources'] -= random.randint(15, 25)
            new_state['technical_debt'] += random.randint(0, 10)
            
        elif action == 'marketing_investment':
            new_state['market_awareness'] += random.randint(15, 30)
            new_state['resources'] -= random.randint(10, 20)
            new_state['customer_acquisition'] += random.randint(5, 15)
            
        elif action == 'hire_technical':
            new_state['technical_capacity'] += random.randint(20, 35)
            new_state['resources'] -= random.randint(20, 30)
            new_state['operational_costs'] += random.randint(5, 10)
            
        elif action == 'hire_sales':
            new_state['sales_capacity'] += random.randint(15, 25)
            new_state['resources'] -= random.randint(15, 25)
            new_state['operational_costs'] += random.randint(3, 8)
            
        elif action == 'expand_markets':
            new_state['market_reach'] += random.randint(20, 40)
            new_state['resources'] -= random.randint(25, 40)
            new_state['complexity'] += random.randint(10, 20)
            
        elif action == 'deepen_current_market':
            new_state['market_penetration'] += random.randint(15, 30)
            new_state['resources'] -= random.randint(10, 20)
            new_state['customer_loyalty'] += random.randint(10, 20)
            
        elif action == 'seek_funding':
            funding_amount = random.randint(100, 200)
            new_state['resources'] += funding_amount
            new_state['investor_pressure'] += random.randint(15, 25)
            new_state['equity_dilution'] += random.randint(10, 20)
            
        elif action == 'bootstrap':
            new_state['autonomy'] += random.randint(10, 20)
            new_state['resource_efficiency'] += random.randint(5, 15)
            # No resource cost, but slower growth
        
        # Apply some general effects
        new_state['resources'] = max(0, new_state['resources'])  # Can't go negative
        
        # Random market events
        if random.random() < 0.1:  # 10% chance of market event
            market_event = random.choice(['positive', 'negative'])
            if market_event == 'positive':
                new_state['resources'] += random.randint(10, 30)
                new_state['market_awareness'] += random.randint(5, 15)
            else:
                new_state['resources'] -= random.randint(5, 20)
                new_state['competitive_pressure'] += random.randint(5, 15)
        
        return new_state
    
    def get_possible_actions(self):
        """Get possible business actions from current state."""
        if self.state['quarter'] >= 5:  # Max 5 quarters
            return []
        
        actions = [
            'product_development',
            'marketing_investment', 
            'hire_technical',
            'hire_sales',
            'expand_markets',
            'deepen_current_market',
            'seek_funding',
            'bootstrap'
        ]
        
        # Filter actions based on current state
        filtered_actions = []
        for action in actions:
            if action in ['product_development', 'marketing_investment', 'hire_technical', 
                         'hire_sales', 'expand_markets', 'deepen_current_market']:
                if self.state['resources'] >= 10:  # Need minimum resources
                    filtered_actions.append(action)
            elif action == 'seek_funding':
                if self.state['equity_dilution'] < 50:  # Don't dilute too much
                    filtered_actions.append(action)
            else:  # bootstrap
                filtered_actions.append(action)
        
        return filtered_actions if filtered_actions else ['bootstrap']  # Always have at least one option
    
    def is_terminal_state(self):
        """Check if this is a terminal state."""
        return (self.state['quarter'] >= 5 or 
                self.state['resources'] <= 0 or 
                len(self.get_possible_actions()) == 0)
    
    def evaluate_state(self):
        """Evaluate the current business state."""
        if not self.agent:
            # Simple heuristic evaluation
            score = (
                self.state['resources'] * 0.15 +
                self.state['product_quality'] * 0.20 +
                self.state['market_awareness'] * 0.15 +
                self.state['technical_capacity'] * 0.15 +
                self.state['sales_capacity'] * 0.10 +
                self.state['market_reach'] * 0.10 +
                self.state['customer_loyalty'] * 0.15 -
                self.state['technical_debt'] * 0.05 -
                self.state['operational_costs'] * 0.05 -
                self.state['investor_pressure'] * 0.03 -
                self.state['competitive_pressure'] * 0.05
            ) / 200  # Normalize
            
            return max(0.0, min(1.0, score))
        
        # Use agent for evaluation with detailed state description
        state_description = f"""
        Business State Analysis (Quarter {self.state['quarter']}/5):
        
        Financial Metrics:
        - Resources: {self.state['resources']}
        - Operational Costs: {self.state['operational_costs']}
        
        Product & Technology:
        - Product Quality: {self.state['product_quality']}
        - Technical Capacity: {self.state['technical_capacity']}
        - Technical Debt: {self.state['technical_debt']}
        
        Market Position:
        - Market Awareness: {self.state['market_awareness']}
        - Market Reach: {self.state['market_reach']}
        - Market Penetration: {self.state['market_penetration']}
        - Customer Loyalty: {self.state['customer_loyalty']}
        - Customer Acquisition: {self.state['customer_acquisition']}
        
        Team & Operations:
        - Sales Capacity: {self.state['sales_capacity']}
        
        Strategic Factors:
        - Autonomy: {self.state['autonomy']}
        - Resource Efficiency: {self.state['resource_efficiency']}
        - Equity Dilution: {self.state['equity_dilution']}%
        - Investor Pressure: {self.state['investor_pressure']}
        - Competitive Pressure: {self.state['competitive_pressure']}
        - Complexity: {self.state['complexity']}
        
        Decision History: {self.state['decisions']}
        
        Evaluate this business position on a scale of 0.0 to 1.0, considering:
        - Financial sustainability
        - Market position strength
        - Growth potential
        - Risk factors
        - Strategic positioning for long-term success
        
        Provide only a numerical score between 0.0 and 1.0.
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


def run_managed_mcts_example():
    """Run the managed MCTS example."""
    print("=== Managed MCTS Business Strategy Example ===\n")
    
    try:
        # Create master agent
        master_agent = Agent(api_key=os.environ["OPENAI_API_KEY"])
        print("Master agent created successfully")
        
    except KeyError:
        print("OPENAI_API_KEY not found in environment variables")
        return
    except Exception as e:
        print(f"Error creating master agent: {e}")
        return
    
    # Define the business strategy problem
    problem_description = """
    Startup Business Strategy Optimization:
    
    A technology startup needs to make strategic decisions over 5 quarters to achieve sustainable growth.
    The company starts with limited resources and must balance multiple competing priorities:
    
    DECISION AREAS:
    1. Product Development vs Marketing Investment
    2. Technical Hiring vs Sales Hiring  
    3. Market Expansion vs Market Deepening
    4. Funding Strategy (Seek Investment vs Bootstrap)
    
    KEY METRICS TO OPTIMIZE:
    - Financial sustainability (resources, costs)
    - Product quality and technical capability
    - Market position (awareness, reach, penetration)
    - Team capabilities (technical, sales)
    - Strategic positioning (autonomy, efficiency)
    
    CONSTRAINTS:
    - Limited resources each quarter
    - Market competition and external pressures
    - Technical debt accumulation
    - Investor pressure if seeking funding
    - Operational complexity with growth
    
    The goal is to find the optimal sequence of decisions that maximizes long-term business success
    while managing risks and resource constraints.
    """
    
    # Create MCTS Manager
    manager = MCTSManager(
        master_agent=master_agent,
        problem_description=problem_description,
        max_agents=4
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
    
    # Create initial business state
    initial_state = {
        'quarter': 0,
        'resources': 100,
        'product_quality': 30,
        'market_awareness': 20,
        'technical_capacity': 25,
        'sales_capacity': 15,
        'market_reach': 10,
        'market_penetration': 15,
        'customer_loyalty': 20,
        'customer_acquisition': 10,
        'technical_debt': 5,
        'operational_costs': 10,
        'autonomy': 50,
        'resource_efficiency': 30,
        'equity_dilution': 0,
        'investor_pressure': 0,
        'competitive_pressure': 20,
        'complexity': 15,
        'decisions': []
    }
    
    print(f"\nInitial Business State:")
    print(f"  Resources: {initial_state['resources']}")
    print(f"  Product Quality: {initial_state['product_quality']}")
    print(f"  Market Awareness: {initial_state['market_awareness']}")
    print(f"  Technical Capacity: {initial_state['technical_capacity']}")
    
    # Create managed MCTS
    print(f"\nCreating Managed MCTS...")
    managed_mcts = manager.create_managed_mcts(initial_state, BusinessStrategyNode)
    
    print(f"Available actions: {managed_mcts.root.get_possible_actions()}")
    
    # Run MCTS search
    print(f"\nRunning MCTS search with specialized agents...")
    try:
        best_child = managed_mcts.search(iterations=50)  # Reduced for demo
        
        if best_child:
            print(f"\nRecommended action: {best_child.value}")
            
            # Show action statistics
            print(f"\nAction Statistics:")
            stats = managed_mcts.get_action_statistics()
            for action, stat in stats.items():
                print(f"  {action}: visits={stat['visits']}, win_rate={stat['win_rate']:.3f}")
            
            # Show tree structure
            print(f"\nMCTS Tree Structure (top 2 levels):")
            managed_mcts.print_tree(max_depth=2)
            
        else:
            print("No best action found")
            
    except Exception as e:
        print(f"Error during MCTS search: {e}")
        import traceback
        traceback.print_exc()


def run_simple_managed_example():
    """Run a simple managed example without API calls."""
    print("=== Simple Managed MCTS Example (No API) ===\n")
    
    # Create a dummy agent
    class DummyAgent:
        def reason(self, prompt, context=None):
            return "0.5"  # Always return neutral score
    
    master_agent = DummyAgent()
    
    # Create manager with default config
    manager = MCTSManager(
        master_agent=master_agent,
        problem_description="Simple test problem",
        max_agents=2
    )
    
    # Use default configuration
    config = manager._get_default_config()
    manager.mcts_config = config
    
    # Create dummy agents
    manager.agents_pool = {
        "general_strategist": DummyAgent(),
        "explorer": DummyAgent()
    }
    
    print("Using default configuration")
    manager.print_configuration()
    
    # Create simple initial state
    initial_state = {
        'quarter': 0,
        'resources': 100,
        'product_quality': 30,
        'market_awareness': 20,
        'technical_capacity': 25,
        'sales_capacity': 15,
        'market_reach': 10,
        'market_penetration': 15,
        'customer_loyalty': 20,
        'customer_acquisition': 10,
        'technical_debt': 5,
        'operational_costs': 10,
        'autonomy': 50,
        'resource_efficiency': 30,
        'equity_dilution': 0,
        'investor_pressure': 0,
        'competitive_pressure': 20,
        'complexity': 15,
        'decisions': []
    }
    
    # Create managed MCTS
    managed_mcts = manager.create_managed_mcts(initial_state, BusinessStrategyNode)
    
    print(f"\nRunning simple MCTS search...")
    best_child = managed_mcts.search(iterations=100)
    
    if best_child:
        print(f"\nRecommended action: {best_child.value}")
        
        stats = managed_mcts.get_action_statistics()
        for action, stat in stats.items():
            print(f"  {action}: visits={stat['visits']}, win_rate={stat['win_rate']:.3f}")


if __name__ == "__main__":
    print("Managed MCTS Example\n")
    print("Choose an option:")
    print("1. Run with OpenAI Agents (requires API key)")
    print("2. Run simple example (no API required)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        run_managed_mcts_example()
    elif choice == "2":
        run_simple_managed_example()
    else:
        print("Invalid choice. Running simple example...")
        run_simple_managed_example() 