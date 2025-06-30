import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv
from agent import Agent
from MCTS import MCTS, MCTSNode
import re
import random

# Load environment variables
load_dotenv()


class MCTSManager:
    """
    Manager class that uses an agent to decide MCTS structure, agent assignments, and configurations.
    """
    
    def __init__(self, 
                 master_agent: Agent,
                 problem_description: str,
                 max_agents: int = 5,
                 max_iterations: int = 1000):
        """
        Initialize the MCTS Manager.
        
        Args:
            master_agent: The master agent that makes decisions about MCTS structure
            problem_description: Description of the problem to solve
            max_agents: Maximum number of agents to create
            max_iterations: Maximum MCTS iterations
        """
        self.master_agent = master_agent
        self.problem_description = problem_description
        self.max_agents = max_agents
        self.max_iterations = max_iterations
        self.agents_pool: Dict[str, Agent] = {}
        self.mcts_config: Optional[Dict] = None
        
    def analyze_problem_and_create_structure(self) -> Dict[str, Any]:
        """
        Analyze the problem and create MCTS structure with specialized agents.
        
        Returns:
            Dictionary containing the MCTS configuration
        """
        analysis_prompt = f"""
        Analyze this problem and design an optimal Monte Carlo Tree Search structure:
        
        PROBLEM: {self.problem_description}
        
        You need to decide:
        1. How many specialized agents to create (max {self.max_agents})
        2. What role/expertise each agent should have
        3. How to assign agents to different parts of the search tree
        4. MCTS parameters (exploration constant, iterations)
        
        RESPOND IN THIS EXACT JSON FORMAT:
        {{
            "analysis": "Brief analysis of the problem",
            "recommended_agents": [
                {{
                    "agent_id": "unique_id",
                    "role_name": "descriptive_name",
                    "role_prompt": "detailed role description for the agent",
                    "specialization": "what this agent is expert in",
                    "tree_assignment": "root|exploration|exploitation|evaluation|specific_depth_X"
                }}
            ],
            "mcts_parameters": {{
                "exploration_constant": 1.414,
                "max_iterations": 500,
                "max_depth": 10
            }},
            "node_strategy": {{
                "root_agent": "agent_id_for_root",
                "default_agent": "agent_id_for_general_nodes",
                "depth_assignments": {{
                    "0": "agent_id",
                    "1-3": "agent_id",
                    "4+": "agent_id"
                }}
            }},
            "reasoning": "Explanation of why this structure was chosen"
        }}
        
        Make sure the JSON is valid and complete.
        """
        # Use chat for both reasoning and execution
        response = self.master_agent.chat(analysis_prompt, statement=analysis_prompt)
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                config = json.loads(json_match.group())
                self.mcts_config = config
                return config
            else:
                raise ValueError("No JSON found in response")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing agent response: {e}")
            print(f"Raw response: {response}")
            # Return default configuration
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get a default MCTS configuration if parsing fails.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "analysis": "Using default configuration due to parsing error",
            "recommended_agents": [
                {
                    "agent_id": "general_strategist",
                    "role_name": "General Strategist",
                    "role_prompt": "You are a general problem-solving strategist. Analyze situations comprehensively and make balanced decisions.",
                    "specialization": "general problem solving",
                    "tree_assignment": "root"
                },
                {
                    "agent_id": "explorer",
                    "role_name": "Explorer",
                    "role_prompt": "You are an exploration specialist. Focus on discovering new possibilities and creative solutions.",
                    "specialization": "exploration and discovery",
                    "tree_assignment": "exploration"
                }
            ],
            "mcts_parameters": {
                "exploration_constant": 1.414,
                "max_iterations": 500,
                "max_depth": 10
            },
            "node_strategy": {
                "root_agent": "general_strategist",
                "default_agent": "explorer",
                "depth_assignments": {
                    "0": "general_strategist",
                    "1+": "explorer"
                }
            },
            "reasoning": "Default configuration with general strategist and explorer agents"
        }
    
    def create_specialized_agents(self, config: Dict[str, Any]) -> Dict[str, Agent]:
        """
        Create specialized agents based on the configuration.
        Args:
            config: Configuration dictionary from analysis
        Returns:
            Dictionary mapping agent IDs to Agent instances
        Note: Agent configuration is now handled via environment variables only.
        """
        agents = {}
        for agent_config in config["recommended_agents"]:
            agent_id = agent_config["agent_id"]
            role_prompt = agent_config["role_prompt"]
            try:
                agent = Agent(system_prompt=role_prompt)
                agents[agent_id] = agent
                print(f"Created agent: {agent_config['role_name']} ({agent_id})")
            except Exception as e:
                print(f"Failed to create agent {agent_id}: {e}")
        self.agents_pool = agents
        return agents
    
    def get_agent_for_node(self, node: MCTSNode, depth: int = 0) -> Tuple[Agent, str]:
        """
        Determine which agent and role prompt to use for a specific node.
        
        Args:
            node: The MCTS node
            depth: Depth of the node in the tree
            
        Returns:
            Tuple of (Agent instance, role prompt)
        """
        if not self.mcts_config or not self.agents_pool:
            # Return master agent as fallback
            return self.master_agent, "You are a general problem solver."
        
        node_strategy = self.mcts_config["node_strategy"]
        
        # Check depth-specific assignments
        depth_assignments = node_strategy.get("depth_assignments", {})
        
        # Find matching depth assignment
        agent_id = None
        for depth_range, assigned_agent_id in depth_assignments.items():
            if self._depth_matches_range(depth, depth_range):
                agent_id = assigned_agent_id
                break
        
        # Fallback to default agent
        if not agent_id:
            agent_id = node_strategy.get("default_agent")
        
        # Get agent and role prompt
        if agent_id and agent_id in self.agents_pool:
            agent = self.agents_pool[agent_id]
            # Find role prompt for this agent
            role_prompt = self._get_role_prompt_for_agent(agent_id)
            return agent, role_prompt
        
        # Final fallback
        return self.master_agent, "You are a general problem solver."
    
    def _depth_matches_range(self, depth: int, depth_range: str) -> bool:
        """
        Check if depth matches a range specification.
        
        Args:
            depth: Current depth
            depth_range: Range specification (e.g., "0", "1-3", "4+")
            
        Returns:
            True if depth matches the range
        """
        if depth_range == str(depth):
            return True
        
        if "-" in depth_range:
            start, end = depth_range.split("-")
            return int(start) <= depth <= int(end)
        
        if depth_range.endswith("+"):
            min_depth = int(depth_range[:-1])
            return depth >= min_depth
        
        return False
    
    def _get_role_prompt_for_agent(self, agent_id: str) -> str:
        """
        Get the role prompt for a specific agent ID.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Role prompt string
        """
        if not self.mcts_config:
            return "You are a general problem solver."
        
        for agent_config in self.mcts_config["recommended_agents"]:
            if agent_config["agent_id"] == agent_id:
                return agent_config["role_prompt"]
        
        return "You are a general problem solver."
    
    def create_managed_mcts(self, root_state: Any, problem_node_class: type) -> 'ManagedMCTS':
        """
        Create an MCTS instance with the managed configuration.
        
        Args:
            root_state: Initial state for the MCTS
            problem_node_class: Class to use for MCTS nodes
            
        Returns:
            ManagedMCTS instance
        """
        if not self.mcts_config:
            raise ValueError("Must call analyze_problem_and_create_structure() first")
        
        # Get root agent
        root_agent_id = self.mcts_config["node_strategy"]["root_agent"]
        root_agent = self.agents_pool.get(root_agent_id, self.master_agent)
        root_role_prompt = self._get_role_prompt_for_agent(root_agent_id)
        
        # Get MCTS parameters
        params = self.mcts_config["mcts_parameters"]
        
        return ManagedMCTS(
            root_state=root_state,
            agent=root_agent,
            role_prompt=root_role_prompt,
            exploration_constant=params.get("exploration_constant", 1.414),
            max_iterations=params.get("max_iterations", 500),
            manager=self,
            problem_node_class=problem_node_class
        )
    
    def print_configuration(self) -> None:
        """Print the current MCTS configuration."""
        if not self.mcts_config:
            print("No configuration available. Run analyze_problem_and_create_structure() first.")
            return
        
        config = self.mcts_config
        
        print("=== MCTS Manager Configuration ===")
        print(f"\nProblem Analysis: {config['analysis']}")
        
        print(f"\nRecommended Agents ({len(config['recommended_agents'])}):")
        for agent in config["recommended_agents"]:
            print(f"  - {agent['role_name']} ({agent['agent_id']})")
            print(f"    Specialization: {agent['specialization']}")
            print(f"    Assignment: {agent['tree_assignment']}")
        
        print(f"\nMCTS Parameters:")
        params = config["mcts_parameters"]
        for key, value in params.items():
            print(f"  - {key}: {value}")
        
        print(f"\nNode Strategy:")
        strategy = config["node_strategy"]
        print(f"  - Root agent: {strategy['root_agent']}")
        print(f"  - Default agent: {strategy['default_agent']}")
        print(f"  - Depth assignments:")
        for depth, agent_id in strategy["depth_assignments"].items():
            print(f"    - Depth {depth}: {agent_id}")
        
        print(f"\nReasoning: {config['reasoning']}")

    def create_child_node(self, parent, action, depth, tree, node_class=None):
        """
        Create and add a child node to the tree, using manager's agent/role assignment.
        Args:
            parent: Parent node
            action: Action leading to the new node
            depth: Depth of the new node
            tree: The Tree instance
            node_class: Node class to instantiate (default: parent's class)
        Returns:
            The newly created child node
        """
        agent, role_prompt = self.get_agent_for_node(parent, depth)
        child_state = parent.apply_action(action)
        node_class = node_class or parent.__class__
        child = node_class(
            value=action,
            parent=parent,
            agent=agent,
            role_prompt=role_prompt,
            state=child_state
        )
        tree.add_node(parent, child)
        return child


class ManagedMCTS(MCTS):
    """
    MCTS implementation that uses the manager for agent assignment.
    """
    
    def __init__(self, 
                 root_state: Any,
                 agent: Any,
                 role_prompt: str,
                 exploration_constant: float,
                 max_iterations: int,
                 manager: MCTSManager,
                 problem_node_class: type):
        """
        Initialize managed MCTS.
        
        Args:
            root_state: Initial state
            agent: Root agent
            role_prompt: Root role prompt
            exploration_constant: UCB1 parameter
            max_iterations: Max iterations
            manager: MCTS manager instance
            problem_node_class: Node class to use
        """
        super().__init__(root_state, agent, role_prompt, exploration_constant, max_iterations)
        self.manager = manager
        self.problem_node_class = problem_node_class
        
        # Override root node with problem-specific class
        self.root.__class__ = problem_node_class
        self.root.possible_actions = self.root.get_possible_actions()
        self.root.untried_actions = self.root.possible_actions.copy()
    
    def select_and_expand(self) -> MCTSNode:
        """
        Enhanced selection and expansion with manager-assigned agents.
        
        Returns:
            Selected/expanded leaf node
        """
        node = self.root
        depth = 0
        
        # Selection: traverse down the tree using UCB1
        while not node.is_terminal_state() and node.is_fully_expanded():
            best_child = node.best_child(self.exploration_constant)
            if best_child is None:
                break
            node = best_child
            depth += 1
        
        # Expansion: add a new child if possible
        if not node.is_terminal_state() and not node.is_fully_expanded():
            action = random.choice(node.untried_actions)
            
            # Get agent assignment from manager
            child_agent, child_role_prompt = self.manager.get_agent_for_node(node, depth + 1)
            
            # Create child with manager-assigned agent
            child_state = node.apply_action(action)
            child = self.problem_node_class(
                value=action,
                parent=node,
                agent=child_agent,
                role_prompt=child_role_prompt,
                state=child_state
            )
            
            node.add_child(child)
            
            # Remove action from untried actions
            if action in node.untried_actions:
                node.untried_actions.remove(action)
            
            # Initialize new node's actions
            child.possible_actions = child.get_possible_actions()
            child.untried_actions = child.possible_actions.copy()
            
            node = child
        
        return node


# Example usage
if __name__ == "__main__":
    print("MCTS Manager Example")
    
    try:
        # Create master agent
        master_agent = Agent(api_key=os.environ["OPENAI_API_KEY"])
        
        # Define problem
        problem_description = """
        Business Strategy Problem: A startup company needs to make a series of strategic decisions
        over 5 quarters to maximize growth while managing limited resources. Decisions include:
        - Product development vs marketing investment
        - Hiring technical staff vs sales staff  
        - Expanding to new markets vs deepening current market
        - Seeking funding vs bootstrapping
        
        Each decision affects resources, market position, team capabilities, and future options.
        The goal is to achieve sustainable growth and market leadership.
        """
        
        # Create manager
        manager = MCTSManager(
            master_agent=master_agent,
            problem_description=problem_description,
            max_agents=4
        )
        
        # Analyze and create structure
        print("Analyzing problem and creating MCTS structure...")
        config = manager.analyze_problem_and_create_structure()
        
        # Create specialized agents
        print("\Creating specialized agents...")
        agents = manager.create_specialized_agents(config)
        
        # Print configuration
        print("\ Configuration:")
        manager.print_configuration()
        
        print(f"MCTS Manager setup complete!")
        print(f"Created {len(agents)} specialized agents")
        
    except KeyError:
        print("OPENAI_API_KEY not found in environment variables")
    except Exception as e:
        print(f"Error: {e}") 