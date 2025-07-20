dev
"""
State management for lite-agent framework.
Provides AgentState for short-term memory and context storage.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class AgentState:
    """
    AgentState object for managing short-term memory and internal context.
    No persistent memory - all state is ephemeral for the current session.
    """
    
    # Core state
    memory: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    session_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Agent execution context
    current_agent: Optional[str] = None
    execution_stack: List[str] = field(default_factory=list)
    tool_results: Dict[str, Any] = field(default_factory=dict)
    
    def set_memory(self, key: str, value: Any) -> None:
        """Set a memory value by key."""
        self.memory[key] = value
        self.last_updated = datetime.now()
    
    def get_memory(self, key: str, default: Any = None) -> Any:
        """Get a memory value by key."""
        return self.memory.get(key, default)
    
    def update_memory(self, updates: Dict[str, Any]) -> None:
        """Update multiple memory values at once."""
        self.memory.update(updates)
        self.last_updated = datetime.now()
    
    def clear_memory(self) -> None:
        """Clear all memory."""
        self.memory.clear()
        self.last_updated = datetime.now()
    
    def set_context(self, key: str, value: Any) -> None:
        """Set a context value by key."""
        self.context[key] = value
        self.last_updated = datetime.now()
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context value by key."""
        return self.context.get(key, default)
    
    def add_to_history(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a message to conversation history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.conversation_history.append(message)
        self.last_updated = datetime.now()
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation history with optional limit."""
        if limit is None:
            return self.conversation_history.copy()
        return self.conversation_history[-limit:] if limit > 0 else []
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
        self.last_updated = datetime.now()
    
    def push_agent(self, agent_name: str) -> None:
        """Push an agent onto the execution stack."""
        self.execution_stack.append(agent_name)
        self.current_agent = agent_name
        self.last_updated = datetime.now()
    
    def pop_agent(self) -> Optional[str]:
        """Pop an agent from the execution stack."""
        if self.execution_stack:
            popped = self.execution_stack.pop()
            self.current_agent = self.execution_stack[-1] if self.execution_stack else None
            self.last_updated = datetime.now()
            return popped
        return None
    
    def store_tool_result(self, tool_name: str, result: Any) -> None:
        """Store the result of a tool execution."""
        self.tool_results[tool_name] = {
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        self.last_updated = datetime.now()
    
    def get_tool_result(self, tool_name: str) -> Any:
        """Get the result of a previous tool execution."""
        tool_data = self.tool_results.get(tool_name)
        return tool_data["result"] if tool_data else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "memory": self.memory,
            "context": self.context,
            "conversation_history": self.conversation_history,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "current_agent": self.current_agent,
            "execution_stack": self.execution_stack,
            "tool_results": self.tool_results
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        """Create AgentState from dictionary."""
        state = cls()
        state.memory = data.get("memory", {})
        state.context = data.get("context", {})
        state.conversation_history = data.get("conversation_history", [])
        state.session_id = data.get("session_id")
        
        if data.get("created_at"):
            state.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("last_updated"):
            state.last_updated = datetime.fromisoformat(data["last_updated"])
        
        state.current_agent = data.get("current_agent")
        state.execution_stack = data.get("execution_stack", [])
        state.tool_results = data.get("tool_results", {})
        
        return state
    
    def to_json(self) -> str:
        """Serialize state to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "AgentState":
        """Create AgentState from JSON string."""
        return cls.from_dict(json.loads(json_str))
=======


class AgentState:
    def __init__(self,input_data=None):
        self.input_data = input_data or {}

    def get(self,key,default=None):
        return self.data_get(key,default)

    def set(self,key,value):
        self.input_data[key] = value

    def all(self):
        return self._data
    

    #Shares state between agents
main
