# MCTS Agents

A sophisticated Monte Carlo Tree Search (MCTS) implementation with intelligent agent integration for complex decision-making problems.

## Overview

This project combines Monte Carlo Tree Search algorithms with OpenAI-powered agents to create an intelligent decision-making system. The system can automatically analyze problems, design optimal MCTS structures, and assign specialized agents to different parts of the search tree for enhanced performance.

## Key Features

### Intelligent MCTS Management
- **Automated Structure Design**: AI agents analyze problems and design optimal MCTS configurations
- **Specialized Agent Assignment**: Different agents with specific expertise handle different parts of the search tree
- **Dynamic Role Assignment**: Agents are assigned based on tree depth and problem requirements

### Agent Integration
- **OpenAI Integration**: Seamless integration with OpenAI's GPT models
- **Role-Based Reasoning**: Each agent operates with specific role prompts for specialized expertise
- **Dependency Injection**: Flexible agent assignment system

### Advanced Tree Operations
- **Comprehensive Node Management**: Full-featured node class with parent-child relationships
- **Tree Utilities**: Complete set of tree operations (traversal, search, statistics)
- **Flexible Architecture**: Easily extensible for different problem domains

### Structured Decision Making
- **JSON-Based Configuration**: Structured output parsing for reliable agent responses
- **Statistical Analysis**: Comprehensive statistics and performance metrics
- **Visualization**: Tree structure printing and analysis tools

## Project Structure

```
MCTSAgents/
├── agent.py                    # Core agent class with OpenAI integration
├── node.py                     # Basic node class with dictionary-based children
├── tree.py                     # Tree management and operations
├── mcts.py                     # Monte Carlo Tree Search implementation
├── mcts_manager.py             # Intelligent MCTS structure management
├── mcts_example.py             # Basic MCTS usage examples
├── managed_mcts_example.py     # Advanced managed MCTS examples
├── requirements.txt            # Python dependencies
└── README.md                   
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd MCTSAgents
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```
   AZURE_OPENAI_API_KEY="your AzureOpenAI key"
   AZURE_OPENAI_ENDPOINT="your endpoint"
   AZURE_OPENAI_DEPLOYMENT_NAME="model name"
   ```

## Quick Start

### Basic MCTS Usage

```python
from agent import Agent
from mcts import MCTS, MCTSNode
import os

# Create an agent
agent = Agent(api_key=os.environ["OPENAI_API_KEY"])

# Define your problem state
initial_state = {"resources": 100, "step": 0}

# Create MCTS
mcts = MCTS(
    root_state=initial_state,
    agent=agent,
    role_prompt="You are a strategic decision maker.",
    max_iterations=1000
)

# Find the best action
best_action = mcts.get_best_action()
```

### Managed MCTS with Intelligent Structure Design

```python
from agent import Agent
from mcts_manager import MCTSManager
import os

# Create master agent
master_agent = Agent(api_key=os.environ["OPENAI_API_KEY"])

# Define your problem
problem_description = """
Your complex decision-making problem description here.
Include constraints, objectives, and decision areas.
"""

# Create MCTS Manager
manager = MCTSManager(
    master_agent=master_agent,
    problem_description=problem_description,
    max_agents=4
)

# Analyze problem and create structure
config = manager.analyze_problem_and_create_structure()
agents = manager.create_specialized_agents(config)

# Create managed MCTS
managed_mcts = manager.create_managed_mcts(initial_state, YourNodeClass)

# Run intelligent search
best_action = managed_mcts.get_best_action()
```

## Core Components

### Agent Class (`agent.py`)
- OpenAI API integration
- Reasoning and execution capabilities
- Message history management
- Role-based prompting

### Node Class (`node.py`)
- Dictionary-based children storage
- Parent-child relationship management
- Metadata support
- Agent integration with role prompts

### Tree Class (`tree.py`)
- Tree-wide operations and utilities
- Search and traversal methods
- Statistical analysis
- Visualization tools

### MCTS Class (`mcts.py`)
- Complete Monte Carlo Tree Search implementation
- UCB1 selection strategy
- Agent-based state evaluation
- Comprehensive statistics tracking

### MCTS Manager (`mcts_manager.py`)
- Intelligent problem analysis
- Automated MCTS structure design
- Specialized agent creation and assignment
- JSON-structured configuration parsing

## Examples

### Business Strategy Decision Making

Run the business strategy example:

```bash
python managed_mcts_example.py
```

This example demonstrates:
- Startup business strategy optimization
- Multiple decision areas (product, hiring, funding, marketing)
- Complex state space with 18+ business metrics
- Intelligent agent assignment based on expertise

### Simple Decision Problems

Run the basic MCTS example:

```bash
python mcts_example.py
```

Features:
- Basic decision-making scenarios
- Resource management
- Simple state evaluation
- Both agent-based and heuristic evaluation modes

## Configuration

### MCTS Parameters
- `exploration_constant`: UCB1 exploration parameter (default: √2)
- `max_iterations`: Maximum search iterations
- `max_depth`: Maximum tree depth

### Agent Configuration
- `model`: OpenAI model to use (default: "gpt-4o")
- `system_prompt`: Base system prompt for agents
- `max_history`: Maximum message history length

### Manager Settings
- `max_agents`: Maximum number of specialized agents
- `problem_description`: Detailed problem description for analysis

## Advanced Usage

### Custom Node Classes

Create problem-specific node classes:

```python
class YourProblemNode(MCTSNode):
    def apply_action(self, action):
        # Implement state transition logic
        pass
    
    def get_possible_actions(self):
        # Return available actions
        pass
    
    def is_terminal_state(self):
        # Check if state is terminal
        pass
    
    def evaluate_state(self):
        # Evaluate state quality
        pass
```

### Custom Agent Roles

Define specialized agent roles:

```python
role_prompt = """
You are a [specific expertise] specialist.
Your role is to [specific responsibilities].
Focus on [key areas of expertise].
"""
```

### Depth-Based Agent Assignment

Configure agents for different tree depths:

```json
{
  "depth_assignments": {
    "0": "strategic_planner",
    "1-3": "tactical_analyst", 
    "4+": "execution_specialist"
  }
}
```

## API Reference

### Agent Class
- `reason(query, context=None)`: Perform reasoning
- `execute(task, parameters=None)`: Execute tasks
- `chat(message)`: Simple chat interface

### MCTS Class
- `search(iterations=None)`: Run MCTS search
- `get_best_action()`: Get recommended action
- `get_action_statistics()`: Get detailed statistics
- `print_tree(max_depth=3)`: Visualize tree structure

### MCTSManager Class
- `analyze_problem_and_create_structure()`: Analyze and design structure
- `create_specialized_agents(config)`: Create agent pool
- `create_managed_mcts(state, node_class)`: Create managed MCTS

## Performance Considerations

- **API Costs**: Monitor OpenAI API usage, especially with many agents
- **Iteration Limits**: Balance search quality with computation time
- **Memory Usage**: Large trees can consume significant memory
- **Agent Specialization**: More specialized agents generally improve performance

## Troubleshooting

### Common Issues

1. **API Key Errors**:
   - Ensure `OPENAI_API_KEY` is set in your `.env` file
   - Verify the API key is valid and has sufficient credits

2. **JSON Parsing Errors**:
   - The system includes fallback configurations for parsing failures
   - Check agent responses for malformed JSON

3. **Memory Issues**:
   - Reduce `max_iterations` for large problems
   - Implement state compression for complex state spaces

4. **Performance Issues**:
   - Use fewer agents for faster execution
   - Reduce tree depth limits
   - Implement more efficient state evaluation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Acknowledgments

- OpenAI for providing the GPT models
- Monte Carlo Tree Search research community
- Contributors and testers

## Contact

[Add your contact information here] 