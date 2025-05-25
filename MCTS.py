import math
import random
from typing import List, Optional, Any, Dict, Callable
from node import Node
from tree import Tree
import json


class MCTSNode(Node):
    """
    Extended Node class for MCTS with additional statistics and methods.
    """
    
    def __init__(self, 
                 value: Any = None,
                 node_id: Optional[str] = None,
                 parent: Optional['MCTSNode'] = None,
                 agent: Optional[Any] = None,
                 role_prompt: Optional[str] = None,
                 state: Optional[Any] = None):
        """
        Initialize an MCTS node.
        
        Args:
            value: The value/data stored in this node
            node_id: Unique identifier for the node
            parent: Parent node reference
            agent: Agent instance for reasoning and execution
            role_prompt: Role prompt for the agent
            state: Game/problem state represented by this node
        """
        super().__init__(value, node_id, parent, agent, role_prompt)
        
        # MCTS-specific attributes
        self.visits = 0
        self.wins = 0.0
        self.state = state
        self.is_terminal = False
        self.possible_actions = []
        self.untried_actions = []
        
    def ucb1_score(self, exploration_constant: float = math.sqrt(2)) -> float:
        """
        Calculate UCB1 score for this node.
        
        Args:
            exploration_constant: Exploration vs exploitation balance
            
        Returns:
            UCB1 score
        """
        if self.visits == 0:
            return float('inf')
        
        if self.parent is None or self.parent.visits == 0:
            return self.wins / self.visits
        
        exploitation = self.wins / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        
        return exploitation + exploration
    
    def is_fully_expanded(self) -> bool:
        """
        Check if all possible actions have been tried.
        
        Returns:
            True if fully expanded, False otherwise
        """
        return len(self.untried_actions) == 0
    
    def best_child(self, exploration_constant: float = math.sqrt(2)) -> Optional['MCTSNode']:
        """
        Select the best child using UCB1.
        
        Args:
            exploration_constant: Exploration vs exploitation balance
            
        Returns:
            Best child node or None if no children
        """
        if not self.children:
            return None
        
        # Filter children to only include MCTSNode instances
        mcts_children = [child for child in self.children.values() if isinstance(child, MCTSNode)]
        if not mcts_children:
            return None
        
        return max(mcts_children, key=lambda child: child.ucb1_score(exploration_constant))
    
    def add_child_with_action(self, action: Any, agent: Optional[Any] = None, role_prompt: Optional[str] = None) -> 'MCTSNode':
        """
        Add a child node representing an action.
        
        Args:
            action: The action that leads to this child
            agent: Agent for the child node
            role_prompt: Role prompt for the child's agent
            
        Returns:
            The newly created child node
        """
        child_state = self.apply_action(action)
        child = MCTSNode(
            value=action,
            parent=self,
            agent=agent or self.agent,
            role_prompt=role_prompt or self.role_prompt,
            state=child_state
        )
        
        self.add_child(child)
        
        # Remove action from untried actions
        if action in self.untried_actions:
            self.untried_actions.remove(action)
        
        return child
    
    def apply_action(self, action: Any) -> Any:
        """
        Apply an action to the current state to get the next state.
        This should be overridden by specific implementations.
        
        Args:
            action: Action to apply
            
        Returns:
            New state after applying the action
        """
        # Default implementation - should be overridden
        return self.state
    
    def get_possible_actions(self) -> List[Any]:
        """
        Get all possible actions from this state.
        This should be overridden by specific implementations.
        
        Returns:
            List of possible actions
        """
        # Default implementation - should be overridden
        return []
    
    def is_terminal_state(self) -> bool:
        """
        Check if this is a terminal state.
        This should be overridden by specific implementations.
        
        Returns:
            True if terminal, False otherwise
        """
        return self.is_terminal
    
    def evaluate_state(self) -> float:
        """
        Evaluate the current state using the agent.
        
        Returns:
            Evaluation score (0.0 to 1.0)
        """
        if not self.agent:
            return random.random()  # Random evaluation if no agent
        
        # Use agent to evaluate the state
        evaluation_prompt = f"Evaluate this state and provide a score between 0.0 and 1.0: {self.state}"
        response = self.reason_with_role(evaluation_prompt)
        
        try:
            # Try to extract a numerical score from the response
            import re
            scores = re.findall(r'\b0?\.\d+\b|\b1\.0\b|\b[01]\b', response)
            if scores:
                return float(scores[0])
        except:
            pass
        
        return 0.5  # Default neutral score


class MCTS:
    """
    Monte Carlo Tree Search implementation using agent-enabled nodes.
    """
    
    def __init__(self, 
                 root_state: Any,
                 agent: Any,
                 role_prompt: str = "You are an expert decision maker and evaluator.",
                 exploration_constant: float = math.sqrt(2),
                 max_iterations: int = 1000):
        """
        Initialize MCTS.
        
        Args:
            root_state: Initial state of the problem
            agent: Agent instance for reasoning
            role_prompt: Role prompt for the agent
            exploration_constant: UCB1 exploration parameter
            max_iterations: Maximum number of MCTS iterations
        """
        self.root = MCTSNode(
            value="root",
            state=root_state,
            agent=agent,
            role_prompt=role_prompt
        )
        self.exploration_constant = exploration_constant
        self.max_iterations = max_iterations
        
        # Initialize root node
        self.root.possible_actions = self.root.get_possible_actions()
        self.root.untried_actions = self.root.possible_actions.copy()
    
    def search(self, iterations: Optional[int] = None) -> MCTSNode:
        """
        Perform MCTS search.
        
        Args:
            iterations: Number of iterations (uses max_iterations if None)
            
        Returns:
            Best child node found
        """
        iterations = iterations or self.max_iterations
        
        for i in range(iterations):
            # Selection and Expansion
            leaf = self.select_and_expand()
            
            # Simulation
            reward = self.simulate(leaf)
            
            # Backpropagation
            self.backpropagate(leaf, reward)
            
            if i % 100 == 0:
                print(f"MCTS iteration {i}/{iterations}")
        
        # Return best child
        return self.root.best_child(exploration_constant=0)  # Pure exploitation
    
    def select_and_expand(self) -> MCTSNode:
        """
        Selection and expansion phase of MCTS.
        
        Returns:
            Selected/expanded leaf node
        """
        node = self.root
        
        # Selection: traverse down the tree using UCB1
        while not node.is_terminal_state() and node.is_fully_expanded():
            best_child = node.best_child(self.exploration_constant)
            if best_child is None:
                break
            node = best_child
        
        # Expansion: add a new child if possible
        if not node.is_terminal_state() and not node.is_fully_expanded():
            action = random.choice(node.untried_actions)
            node = node.add_child_with_action(action, agent=node.agent, role_prompt=node.role_prompt)
            
            # Initialize new node's actions
            node.possible_actions = node.get_possible_actions()
            node.untried_actions = node.possible_actions.copy()
        
        return node
    
    def simulate(self, node: MCTSNode) -> float:
        """
        Simulation phase - evaluate the node using the agent.
        
        Args:
            node: Node to simulate from
            
        Returns:
            Simulation reward
        """
        # Use agent-based evaluation instead of random simulation
        if node.is_terminal_state():
            return node.evaluate_state()
        
        # For non-terminal states, use agent to evaluate
        simulation_prompt = f"""
        Analyze this game/problem state and predict the likely outcome.
        State: {node.state}
        Provide a score between 0.0 (worst) and 1.0 (best) representing the quality of this position.
        """
        
        response = node.reason_with_role(simulation_prompt)
        
        try:
            # Extract numerical score from agent response
            import re
            scores = re.findall(r'\b0?\.\d+\b|\b1\.0\b', response)
            if scores:
                return float(scores[0])
        except:
            pass
        
        return node.evaluate_state()
    
    def backpropagate(self, node: MCTSNode, reward: float) -> None:
        """
        Backpropagation phase - update statistics up the tree.
        
        Args:
            node: Starting node for backpropagation
            reward: Reward to propagate
        """
        current = node
        while current is not None:
            if isinstance(current, MCTSNode):
                current.visits += 1
                current.wins += reward
            current = current.parent
    
    def get_best_action(self) -> Any:
        """
        Get the best action from the root state.
        
        Returns:
            Best action to take
        """
        best_child = self.search()
        return best_child.value if best_child else None
    
    def get_action_statistics(self) -> Dict[Any, Dict[str, float]]:
        """
        Get statistics for all actions from the root.
        
        Returns:
            Dictionary mapping actions to their statistics
        """
        stats = {}
        for child in self.root.children.values():
            stats[child.value] = {
                'visits': child.visits,
                'wins': child.wins,
                'win_rate': child.wins / child.visits if child.visits > 0 else 0,
                'ucb1_score': child.ucb1_score(self.exploration_constant)
            }
        return stats
    
    def print_tree(self, node: Optional[MCTSNode] = None, depth: int = 0, max_depth: int = 3) -> None:
        """
        Print the MCTS tree structure.
        
        Args:
            node: Node to start printing from (root if None)
            depth: Current depth
            max_depth: Maximum depth to print
        """
        if node is None:
            node = self.root
        
        if depth > max_depth:
            return
        
        indent = "  " * depth
        win_rate = node.wins / node.visits if node.visits > 0 else 0
        print(f"{indent}{node.value} (visits: {node.visits}, win_rate: {win_rate:.3f}, ucb1: {node.ucb1_score():.3f})")
        
        for child in node.children.values():
            self.print_tree(child, depth + 1, max_depth)


# Example problem-specific MCTS implementation
class TicTacToeMCTS(MCTS):
    """
    Example MCTS implementation for Tic-Tac-Toe.
    """
    
    def __init__(self, agent: Any, player: str = 'X'):
        """
        Initialize Tic-Tac-Toe MCTS.
        
        Args:
            agent: Agent for reasoning
            player: Player symbol ('X' or 'O')
        """
        initial_board = [' '] * 9  # Empty 3x3 board
        role_prompt = f"You are an expert Tic-Tac-Toe player playing as '{player}'. Analyze positions strategically."
        
        super().__init__(
            root_state={'board': initial_board, 'current_player': player},
            agent=agent,
            role_prompt=role_prompt
        )
        
        self.player = player
        self.opponent = 'O' if player == 'X' else 'X'


# Example usage
if __name__ == "__main__":
    print("MCTS implementation created successfully!")
    print("To use MCTS:")
    print("1. Create an agent instance")
    print("2. Initialize MCTS with your problem state")
    print("3. Call search() to find the best action")
    print("4. Use get_best_action() to get the recommended move")
