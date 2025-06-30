import os
import json
import logging
import time
import random
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential

load_dotenv()

class Agent:
    """
    Agent class that works with Azure OpenAI's API (2024-12-01-preview).
    Includes reasoning, execution capabilities, and message history management.
    """
    
    def __init__(self, system_prompt: str = "You are a helpful AI assistant.", max_history: int = 50):
        """
        Initialize the agent for Azure OpenAI.
        
        Args:
            system_prompt: System prompt for the agent
            max_history: Maximum number of messages to keep in history
        """
        # Load from environment if not provided
        self.api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        self.endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        self.deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
        # Parameter validation for required env vars
        missing = []
        if not self.api_key:
            missing.append("AZURE_OPENAI_API_KEY")
        if not self.endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not self.deployment_name:
            missing.append("AZURE_OPENAI_DEPLOYMENT_NAME")
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        self.model = self.deployment_name
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.message_history: List[Dict[str, str]] = []

        # Use AzureOpenAI client with correct parameters
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
            api_key=self.api_key
        )
        
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
    
    def _call_with_backoff(self, func, *args, max_retries: int = 5, base_delay: float = 1.0, **kwargs):
        """
        Call a function with exponential backoff on exceptions.
        Args:
            func: The function to call
            *args, **kwargs: Arguments to pass to the function
            max_retries: Maximum number of retries
            base_delay: Initial delay in seconds
        Returns:
            The result of the function call
        Raises:
            The last exception if all retries fail
        """
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                sleep_time = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                self.logger.warning(f"Retrying after error: {e}. Attempt {attempt+1}/{max_retries}. Sleeping {sleep_time:.2f}s...")
                time.sleep(sleep_time)
    
    def reason(self, query: str, context: Optional[str] = None) -> str:
        """
        Use the agent to reason about a query.
        
        Args:
            query: The question or problem to reason about
            context: Optional additional context
            add_to_history: Whether to add the reasoning to the message history
            
        Returns:
            The agent's reasoning response
        """
        reasoning_prompt = f"Please reason through this query step by step: {query}"
        if context:
            reasoning_prompt += f"\n\nAdditional context: {context}"
        
        self._add_message("user", reasoning_prompt)
        
        try:
            response = self._call_with_backoff(
                self.client.chat.completions.create,
                model=self.model,
                messages=self.message_history,
                temperature=0.7
            )
            
            reasoning = response.choices[0].message.content
            
            self.logger.info(f"Reasoning completed for query: {query[:50]}...")
            return reasoning
            
        except Exception as e:
            self.logger.error(f"Error in reasoning: {str(e)}")
            return f"Error occurred during reasoning: {str(e)}"
    
    def execute(self, task: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute a task using the agent.
        
        Args:
            task: Description of the task to execute
            parameters: Optional parameters for the task
            add_to_history: Whether to add the execution result to the message history
            
        Returns:
            Dictionary containing execution results
        """
        execution_prompt = f"Execute the following task: {task}"
        if parameters:
            self._validate_json_serializable(parameters, context="parameters (execute)")
            execution_prompt += f"\n\nParameters: {json.dumps(parameters, indent=2)}"
        execution_prompt += "\n\nPlease provide only the results."
        self._add_message("user", execution_prompt)
        
        try:
            response = self._call_with_backoff(
                self.client.chat.completions.create,
                model=self.model,
                messages=self.message_history,
                temperature=0.3
            )
            
            execution_result = response.choices[0].message.content
            
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
    
    def chat(self, message: str, statement: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Chat with the agent, optionally reasoning about a separate statement before executing the message.

        Args:
            message: The message to execute (and default to reason about if statement is not provided)
            statement: Optional statement to reason about before executing the message
            parameters: Optional parameters for the execution step

        Returns:
            Agent's response (execution result as a string)
        """
        self._add_message("user", message)
        to_reason = statement if statement is not None else message
        if parameters is not None:
            self._validate_json_serializable(parameters, context="parameters (chat)")
        result = self.reason_and_execute(to_reason, message, parameters)
        # Always return a string for the assistant's reply
        if isinstance(result, dict):
            if "result" in result and isinstance(result["result"], str):
                agent_response = result["result"]
            elif "error" in result and isinstance(result["error"], str):
                agent_response = result["error"]
            else:
                agent_response = json.dumps(result, indent=2)
        else:
            agent_response = str(result)
        self._add_message("assistant", agent_response)
        
        return agent_response
    
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

    def reason_and_execute(self, query: str, task: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """
        Perform reasoning on a query, then use the reasoning result to inform the execution step.

        Args:
            query: The question or problem to reason about
            task: Description of the task to execute
            parameters: Optional parameters for the task

        Returns:
            The execution result only (not both reasoning and execution)
        """
        reasoning_result = self.reason(query)
        
        # Remove the last user message (reasoning prompt) from history before execution
        if self.message_history and self.message_history[-1]["role"] == "user":
            self.message_history.pop()
            
        execution_context = parameters.copy() if parameters else {}
        execution_context["reasoning"] = reasoning_result
        
        # Remove the last user message (execution prompt) from history before execution
        if self.message_history and self.message_history[-2]["role"] == "user":
            self.message_history.pop(-2)
        
        execution_result = self.execute(task, execution_context)
        return execution_result

    def _validate_json_serializable(self, obj, context: str = "parameters"):
        try:
            json.dumps(obj)
        except Exception as e:
            raise ValueError(f"{context} must be JSON serializable: {e}")

# Example usage for Azure OpenAI
if __name__ == "__main__":
    # Example of how to use the agent with Azure OpenAI
    # Set these environment variables in your .env or system:
    #   AZURE_OPENAI_API_KEY
    #   AZURE_OPENAI_ENDPOINT
    #   AZURE_OPENAI_API_VERSION (optional, default: 2024-12-01-preview)
    #   AZURE_OPENAI_DEPLOYMENT_NAME (your deployment name)

    agent = Agent()
    
    # Combined reasoning and execution example
    result = agent.chat(
        message="Create a simple plan for learning Python",
        statement="What are best practices to learn programming?",
        parameters={"timeframe": "30 days", "skill_level": "beginner"}
    )
    print(f"COT Result:\n\n{result}\n\n")
    
    input("Press Enter to continue...")
    
    print("\n\nMessage History:\n", agent.get_message_history())