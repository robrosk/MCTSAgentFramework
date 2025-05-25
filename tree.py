from typing import List, Optional, Any, Callable
from collections import deque
from node import Node


class Tree:
    """
    A tree class that provides tree-specific functionality and operations.
    Works with the Node class to provide comprehensive tree management.
    """
    
    def __init__(self, root: Optional[Node] = None):
        """
        Initialize a tree.
        
        Args:
            root: Root node of the tree (creates empty node if None)
        """
        self.root = root or Node("Root")
    
    def get_root(self) -> Node:
        """
        Get the root node of the tree.
        
        Returns:
            The root node
        """
        return self.root
    
    def set_root(self, node: Node) -> None:
        """
        Set a new root node for the tree.
        
        Args:
            node: New root node
        """
        node.parent = None
        self.root = node
    
    def get_size(self) -> int:
        """
        Get the total number of nodes in the tree.
        
        Returns:
            Total number of nodes
        """
        return self._get_subtree_size(self.root)
    
    def _get_subtree_size(self, node: Node) -> int:
        """
        Get the size of a subtree rooted at the given node.
        
        Args:
            node: Root of the subtree
            
        Returns:
            Size of the subtree
        """
        size = 1  # Count this node
        for child in node.children.values():
            size += self._get_subtree_size(child)
        return size
    
    def get_height(self) -> int:
        """
        Get the height of the tree.
        
        Returns:
            Height of the tree (single node = 0)
        """
        return self._get_subtree_height(self.root)
    
    def _get_subtree_height(self, node: Node) -> int:
        """
        Get the height of a subtree rooted at the given node.
        
        Args:
            node: Root of the subtree
            
        Returns:
            Height of the subtree
        """
        if node.is_leaf():
            return 0
        return 1 + max(self._get_subtree_height(child) for child in node.children.values())
    
    def get_depth(self, node: Node) -> int:
        """
        Get the depth of a node (distance from root).
        
        Args:
            node: Node to get depth for
            
        Returns:
            Depth of the node (root = 0)
        """
        depth = 0
        current = node.parent
        while current is not None:
            depth += 1
            current = current.parent
        return depth
    
    def get_path_to_root(self, node: Node) -> List[Node]:
        """
        Get the path from a node to the root.
        
        Args:
            node: Starting node
            
        Returns:
            List of nodes from the given node to root (inclusive)
        """
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        return path
    
    def get_ancestors(self, node: Node) -> List[Node]:
        """
        Get all ancestor nodes of a given node.
        
        Args:
            node: Node to get ancestors for
            
        Returns:
            List of ancestor nodes (excluding the node itself)
        """
        ancestors = []
        current = node.parent
        while current is not None:
            ancestors.append(current)
            current = current.parent
        return ancestors
    
    def get_descendants(self, node: Node) -> List[Node]:
        """
        Get all descendant nodes of a given node.
        
        Args:
            node: Node to get descendants for
            
        Returns:
            List of all descendant nodes (excluding the node itself)
        """
        descendants = []
        for child in node.children.values():
            descendants.append(child)
            descendants.extend(self.get_descendants(child))
        return descendants
    
    def get_leaves(self) -> List[Node]:
        """
        Get all leaf nodes in the tree.
        
        Returns:
            List of all leaf nodes
        """
        leaves = []
        self._collect_leaves(self.root, leaves)
        return leaves
    
    def _collect_leaves(self, node: Node, leaves: List[Node]) -> None:
        """
        Helper method to collect leaf nodes.
        
        Args:
            node: Current node
            leaves: List to collect leaves in
        """
        if node.is_leaf():
            leaves.append(node)
        else:
            for child in node.children.values():
                self._collect_leaves(child, leaves)
    
    def find_node(self, node_id: str) -> Optional[Node]:
        """
        Find a node by ID in the tree.
        
        Args:
            node_id: ID of the node to find
            
        Returns:
            The found node, or None if not found
        """
        return self._find_node_recursive(self.root, node_id)
    
    def _find_node_recursive(self, node: Node, node_id: str) -> Optional[Node]:
        """
        Recursive helper for finding a node by ID.
        
        Args:
            node: Current node
            node_id: ID to search for
            
        Returns:
            Found node or None
        """
        if node.node_id == node_id:
            return node
        
        for child in node.children.values():
            found = self._find_node_recursive(child, node_id)
            if found:
                return found
        
        return None
    
    def find_nodes_by_value(self, value: Any) -> List[Node]:
        """
        Find all nodes with a specific value in the tree.
        
        Args:
            value: Value to search for
            
        Returns:
            List of nodes with the specified value
        """
        nodes = []
        self._find_nodes_by_value_recursive(self.root, value, nodes)
        return nodes
    
    def _find_nodes_by_value_recursive(self, node: Node, value: Any, nodes: List[Node]) -> None:
        """
        Recursive helper for finding nodes by value.
        
        Args:
            node: Current node
            value: Value to search for
            nodes: List to collect matching nodes
        """
        if node.value == value:
            nodes.append(node)
        
        for child in node.children.values():
            self._find_nodes_by_value_recursive(child, value, nodes)
    
    def traverse_dfs(self, visit_func: Optional[Callable[[Node], None]] = None) -> List[Node]:
        """
        Depth-first traversal of the tree.
        
        Args:
            visit_func: Optional function to call on each node
            
        Returns:
            List of nodes in DFS order
        """
        result = []
        self._traverse_dfs_recursive(self.root, visit_func, result)
        return result
    
    def _traverse_dfs_recursive(self, node: Node, visit_func: Optional[Callable[[Node], None]], result: List[Node]) -> None:
        """
        Recursive helper for DFS traversal.
        
        Args:
            node: Current node
            visit_func: Function to call on each node
            result: List to collect nodes
        """
        # Visit this node
        if visit_func:
            visit_func(node)
        result.append(node)
        
        # Visit children
        for child in node.children.values():
            self._traverse_dfs_recursive(child, visit_func, result)
    
    def traverse_bfs(self, visit_func: Optional[Callable[[Node], None]] = None) -> List[Node]:
        """
        Breadth-first traversal of the tree.
        
        Args:
            visit_func: Optional function to call on each node
            
        Returns:
            List of nodes in BFS order
        """
        result = []
        queue = deque([self.root])
        
        while queue:
            node = queue.popleft()
            
            # Visit this node
            if visit_func:
                visit_func(node)
            result.append(node)
            
            # Add children to queue
            for child in node.children.values():
                queue.append(child)
        
        return result
    
    def get_level_nodes(self, level: int) -> List[Node]:
        """
        Get all nodes at a specific level.
        
        Args:
            level: Level to get nodes from (root = 0)
            
        Returns:
            List of nodes at the specified level
        """
        if level < 0:
            return []
        
        nodes = []
        self._collect_level_nodes(self.root, 0, level, nodes)
        return nodes
    
    def _collect_level_nodes(self, node: Node, current_level: int, target_level: int, nodes: List[Node]) -> None:
        """
        Helper method to collect nodes at a specific level.
        
        Args:
            node: Current node
            current_level: Current depth level
            target_level: Target depth level
            nodes: List to collect nodes
        """
        if current_level == target_level:
            nodes.append(node)
        elif current_level < target_level:
            for child in node.children.values():
                self._collect_level_nodes(child, current_level + 1, target_level, nodes)
    
    def copy(self, deep: bool = True) -> 'Tree':
        """
        Create a copy of the tree.
        
        Args:
            deep: If True, recursively copy all nodes
            
        Returns:
            Copy of the tree
        """
        if deep:
            root_copy = self._copy_subtree(self.root)
        else:
            root_copy = Node(value=self.root.value, node_id=self.root.node_id)
        
        return Tree(root_copy)
    
    def _copy_subtree(self, node: Node) -> Node:
        """
        Recursively copy a subtree.
        
        Args:
            node: Root of subtree to copy
            
        Returns:
            Copy of the subtree
        """
        new_node = Node(value=node.value, node_id=node.node_id)
        new_node.metadata = node.metadata.copy()
        
        for child in node.children.values():
            child_copy = self._copy_subtree(child)
            new_node.add_child(child_copy)
        
        return new_node
    
    def print_tree(self, indent: int = 0, prefix: str = "") -> None:
        """
        Print a visual representation of the tree.
        
        Args:
            indent: Current indentation level
            prefix: Prefix for the current line
        """
        self._print_node(self.root, indent, prefix)
    
    def _print_node(self, node: Node, indent: int, prefix: str) -> None:
        """
        Helper method to print a node and its children.
        
        Args:
            node: Node to print
            indent: Current indentation level
            prefix: Prefix for the current line
        """
        print(f"{' ' * indent}{prefix}{node}")
        
        child_list = list(node.children.values())
        for i, child in enumerate(child_list):
            is_last = i == len(child_list) - 1
            child_prefix = "└── " if is_last else "├── "
            child_indent = indent + (4 if is_last else 4)
            self._print_node(child, child_indent, child_prefix)
    
    def get_stats(self) -> dict:
        """
        Get statistics about the tree.
        
        Returns:
            Dictionary with tree statistics
        """
        return {
            "size": self.get_size(),
            "height": self.get_height(),
            "leaves": len(self.get_leaves()),
            "root_id": self.root.node_id,
            "root_value": self.root.value
        }


# Example usage
if __name__ == "__main__":
    # Create a tree
    tree = Tree()
    root = tree.get_root()
    
    # Add some nodes
    child1 = Node("Child 1")
    child2 = Node("Child 2")
    grandchild1 = Node("Grandchild 1")
    grandchild2 = Node("Grandchild 2")
    
    root.add_child(child1)
    root.add_child(child2)
    child1.add_child(grandchild1)
    child1.add_child(grandchild2)
    
    print("Tree structure:")
    tree.print_tree()
    
    print(f"\nTree stats: {tree.get_stats()}")
    
    print("\nDFS traversal:")
    for node in tree.traverse_dfs():
        print(f"  {node.value}")
    
    print("\nBFS traversal:")
    for node in tree.traverse_bfs():
        print(f"  {node.value}")
    
    print("\nLeaf nodes:")
    for leaf in tree.get_leaves():
        print(f"  {leaf.value}")
    
    print("\nTree class created successfully!")
