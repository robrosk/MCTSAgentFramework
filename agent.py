import openai
from typing import List, Dict, Any, Optional
import json
import logging
import os
from dotenv import load_dotenv

load_dotenv()

class Agent:
    """
    Agent class that works with OpenAI's API.
    Includes reasoning, execution capabilities, and message history management.
    """
    
    def __init__(self, 
                 api_key: str,
                 model: str = "gpt-4o",
                 system_prompt: str = "You are a helpful AI assistant.",
                 max_history: int = 50):
        """
        Initialize the agent.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use (default: gpt-4)
            system_prompt: System prompt for the agent
            max_history: Maximum number of messages to keep in history
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.message_history: List[Dict[str, str]] = []
        
        # Initialize with system prompt
        self._add_message("system", system_prompt)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _add_message(self, role: str, content: str) -> None:
        """Add a message to the history and manage history size."""
        self.message_history.append({"role": role, "content": content})
        
        # Keep system message and manage history size
        if len(self.message_history) > self.max_history:
            # Keep system message (first) and remove oldest user/assistant messages
            system_msg = self.message_history[0]
            self.message_history = [system_msg] + self.message_history[-(self.max_history-1):]
    
    def reason(self, query: str, context: Optional[str] = None) -> str:
        """
        Use the agent to reason about a query.
        
        Args:
            query: The question or problem to reason about
            context: Optional additional context
            
        Returns:
            The agent's reasoning response
        """
        # Prepare the reasoning prompt
        reasoning_prompt = f"Please reason through this query step by step: {query}"
        if context:
            reasoning_prompt += f"\n\nAdditional context: {context}"
        
        self._add_message("user", reasoning_prompt)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.message_history,
                temperature=0.7
            )
            
            reasoning = response.choices[0].message.content
            self._add_message("assistant", reasoning)
            
            self.logger.info(f"Reasoning completed for query: {query[:50]}...")
            return reasoning
            
        except Exception as e:
            self.logger.error(f"Error in reasoning: {str(e)}")
            return f"Error occurred during reasoning: {str(e)}"
    
    def execute(self, task: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a task using the agent.
        
        Args:
            task: Description of the task to execute
            parameters: Optional parameters for the task
            
        Returns:
            Dictionary containing execution results
        """
        # Prepare execution prompt
        execution_prompt = f"Execute this task: {task}"
        if parameters:
            execution_prompt += f"\n\nParameters: {json.dumps(parameters, indent=2)}"
        
        execution_prompt += "\n\nPlease provide a structured response with your execution plan and results."
        
        self._add_message("user", execution_prompt)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.message_history,
                temperature=0.3
            )
            
            execution_result = response.choices[0].message.content
            self._add_message("assistant", execution_result)
            
            self.logger.info(f"Task executed: {task[:50]}...")
            
            return {
                "success": True,
                "task": task,
                "parameters": parameters,
                "result": execution_result,
                "timestamp": self._get_timestamp()
            }
            
        except Exception as e:
            self.logger.error(f"Error in execution: {str(e)}")
            return {
                "success": False,
                "task": task,
                "parameters": parameters,
                "error": str(e),
                "timestamp": self._get_timestamp()
            }
    
    def get_message_history(self) -> List[Dict[str, str]]:
        """Get the current message history."""
        return self.message_history.copy()
    
    def clear_history(self, keep_system: bool = True) -> None:
        """
        Clear the message history.
        
        Args:
            keep_system: Whether to keep the system prompt (default: True)
        """
        if keep_system and self.message_history:
            system_msg = self.message_history[0]
            self.message_history = [system_msg]
        else:
            self.message_history = []
            if not keep_system:
                self._add_message("system", self.system_prompt)
        
        self.logger.info("Message history cleared")
    
    def update_system_prompt(self, new_prompt: str) -> None:
        """Update the system prompt and reset history."""
        self.system_prompt = new_prompt
        self.message_history = []
        self._add_message("system", new_prompt)
        self.logger.info("System prompt updated")
    
    def chat(self, message: str) -> str:
        """
        Simple chat interface with the agent.
        
        Args:
            message: User message
            
        Returns:
            Agent's response
        """
        self._add_message("user", message)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.message_history,
                temperature=0.7
            )
            
            agent_response = response.choices[0].message.content
            self._add_message("assistant", agent_response)
            
            return agent_response
            
        except Exception as e:
            self.logger.error(f"Error in chat: {str(e)}")
            return f"Error occurred: {str(e)}"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "model": self.model,
            "message_count": len(self.message_history),
            "max_history": self.max_history,
            "system_prompt": self.system_prompt
        }


# Example usage
if __name__ == "__main__":
    # Example of how to use the agent
    # Note: You'll need to provide your actual OpenAI API key
    
    agent = Agent(api_key=os.environ["OPENAI_API_KEY"])
    
    # Reasoning example
    reasoning_result = agent.reason("What are the pros and cons of renewable energy?")
    print("Reasoning Result:", reasoning_result)
    
    # Execution example
    execution_result = agent.execute("Create a simple plan for learning Python", 
                                    {"timeframe": "30 days", "skill_level": "beginner"})
    print("Execution Result:", execution_result)
    
    # Chat example
    response = agent.chat("Hello! How can you help me today?")
    print("Chat Response:", response)
    
    # View message history
    print("Message History:", agent.get_message_history())
    
    print("Agent class created successfully! Uncomment the example code above to test it.")
