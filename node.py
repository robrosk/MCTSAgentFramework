from typing import Dict, Optional, Any
import uuid

class Node:
    """
    A basic node class that uses a dictionary for children connections.
    Each node can have multiple children and maintains parent-child relationships.
    """
    
    def __init__(self, 
                 value: Any = None, 
                 node_id: Optional[str] = None,
                 parent: Optional['Node'] = None,
                 agent: Optional[Any] = None,
                 role_prompt: Optional[str] = None):
        """
        Initialize a node.
        
        Args:
            value: The value/data stored in this node
            node_id: Unique identifier for the node (auto-generated if None)
            parent: Parent node reference
            agent: Agent instance for reasoning and execution (dependency injection)
            role_prompt: Role prompt for the agent
        """
        self.value = value
        self.node_id = node_id or str(uuid.uuid4())
        self.parent = parent
        self.children: Dict[str, 'Node'] = {}  # Dictionary mapping child_id -> child_node
        self.metadata: Dict[str, Any] = {}  # Additional metadata storage
        self.agent = agent  # Injected agent dependency
        self.role_prompt = role_prompt or "You are a helpful AI assistant."
    
    def add_child(self, child: 'Node') -> bool:
        """
        Add a child node to this node.
        
        Args:
            child: The child node to add
            
        Returns:
            True if child was added successfully, False if already exists
        """
        if child.node_id in self.children:
            return False
        
        self.children[child.node_id] = child
        child.parent = self
        return True
    
    def remove_child(self, child_id: str) -> Optional['Node']:
        """
        Remove a child node by ID.
        
        Args:
            child_id: ID of the child to remove
            
        Returns:
            The removed child node, or None if not found
        """
        if child_id in self.children:
            child = self.children.pop(child_id)
            child.parent = None
            return child
        return None
    
    def get_child(self, child_id: str) -> Optional['Node']:
        """
        Get a child node by ID.
        
        Args:
            child_id: ID of the child to retrieve
            
        Returns:
            The child node, or None if not found
        """
        return self.children.get(child_id)
    
    def has_children(self) -> bool:
        """
        Check if this node has any children.
        
        Returns:
            True if node has children, False otherwise
        """
        return len(self.children) > 0
    
    def is_leaf(self) -> bool:
        """
        Check if this node is a leaf (has no children).
        
        Returns:
            True if node is a leaf, False otherwise
        """
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        """
        Check if this node is a root (has no parent).
        
        Returns:
            True if node is root, False otherwise
        """
        return self.parent is None
    
    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set metadata for this node.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get metadata for this node.
        
        Args:
            key: Metadata key
            default: Default value if key not found
            
        Returns:
            Metadata value or default
        """
        return self.metadata.get(key, default)
    
    def reason_with_role(self, query: str, context: Optional[str] = None) -> str:
        """
        Use the agent to reason about a query with the node's role prompt.
        
        Args:
            query: The question or problem to reason about
            context: Optional additional context
            
        Returns:
            The agent's reasoning response
        """
        if not self.agent:
            return "No agent available for reasoning"
        
        # Prepare the full prompt with role
        full_prompt = f"Role: {self.role_prompt}\n\nQuery: {query}"
        if context:
            full_prompt += f"\n\nContext: {context}"
        
        return self.agent.reason(full_prompt, context)
    
    def execute_with_role(self, task: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a task using the agent with the node's role prompt.
        
        Args:
            task: Description of the task to execute
            parameters: Optional parameters for the task
            
        Returns:
            Dictionary containing execution results
        """
        if not self.agent:
            return {
                "success": False,
                "error": "No agent available for execution",
                "task": task,
                "parameters": parameters
            }
        
        # Prepare the full task with role
        full_task = f"Role: {self.role_prompt}\n\nTask: {task}"
        
        return self.agent.execute(full_task, parameters)
    
    def chat_with_role(self, message: str) -> str:
        """
        Chat with the agent using the node's role prompt.
        
        Args:
            message: Message to send to the agent
            
        Returns:
            Agent's response
        """
        if not self.agent:
            return "No agent available for chat"
        
        # Prepare the full message with role
        full_message = f"Role: {self.role_prompt}\n\nMessage: {message}"
        
        return self.agent.chat(full_message)
    
    def __str__(self) -> str:
        """String representation of the node."""
        return f"Node(id={self.node_id[:8]}..., value={self.value}, children={len(self.children)})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the node."""
        return f"Node(id='{self.node_id}', value={repr(self.value)}, children={len(self.children)}, parent={'Yes' if self.parent else 'No'})"


# Example usage
if __name__ == "__main__":
    # Create some nodes
    root = Node("Root")
    child1 = Node("Child 1")
    child2 = Node("Child 2")
    
    # Add children
    root.add_child(child1)
    root.add_child(child2)
    
    print(f"Root: {root}")
    print(f"Child1: {child1}")
    print(f"Root has children: {root.has_children()}")
    print(f"Child1 is leaf: {child1.is_leaf()}")
    print(f"Root is root: {root.is_root()}")
    
    print("Node class created successfully!") 