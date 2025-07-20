"""
Tool execution framework for lite-agent framework.
Provides dynamic tool selection and execution capabilities.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, Type
from dataclasses import dataclass
import inspect
import json
import re
from datetime import datetime
import math
import random


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class Tool(ABC):
    """Abstract base class for tools."""
    
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    def validate_parameters(self, **kwargs) -> None:
        """Validate tool parameters."""
        required = self.parameters.get("required", [])
        missing = [param for param in required if param not in kwargs]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")


class FunctionTool(Tool):
    """Tool wrapper for Python functions."""
    
    def __init__(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None):
        self.func = func
        self.name = name or func.__name__
        self.description = description or func.__doc__ or f"Function tool: {self.name}"
        
        # Extract parameters from function signature
        sig = inspect.signature(func)
        self.parameters = {
            "properties": {},
            "required": []
        }
        
        for param_name, param in sig.parameters.items():
            param_info = {"type": "string"}  # Default type
            
            # Try to infer type from annotation
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_info["type"] = "integer"
                elif param.annotation == float:
                    param_info["type"] = "number"
                elif param.annotation == bool:
                    param_info["type"] = "boolean"
                elif param.annotation == list:
                    param_info["type"] = "array"
                elif param.annotation == dict:
                    param_info["type"] = "object"
            
            self.parameters["properties"][param_name] = param_info
            
            # Check if parameter is required
            if param.default == inspect.Parameter.empty:
                self.parameters["required"].append(param_name)
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute the wrapped function."""
        try:
            start_time = datetime.now()
            result = self.func(**kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ToolResult(
                success=True,
                result=result,
                execution_time=execution_time
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=str(e)
            )


# Built-in tools
class CalculatorTool(Tool):
    """Calculator tool for basic math operations."""
    
    name = "calculator"
    description = "Performs basic mathematical calculations"
    parameters = {
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate (e.g., '2 + 3 * 4')"
            }
        },
        "required": ["expression"]
    }
    
    def execute(self, expression: str) -> ToolResult:
        """Execute mathematical calculation."""
        try:
            # Basic security: only allow safe mathematical operations
            allowed_chars = set("0123456789+-*/().^ %")
            if not all(c in allowed_chars or c.isspace() for c in expression):
                raise ValueError("Invalid characters in expression")
            
            # Replace ^ with ** for Python exponentiation
            expression = expression.replace("^", "**")
            
            # Evaluate safely (limited scope)
            safe_dict = {
                "__builtins__": {},
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "pow": pow, "sqrt": math.sqrt,
                "sin": math.sin, "cos": math.cos, "tan": math.tan,
                "log": math.log, "exp": math.exp, "pi": math.pi,
                "e": math.e
            }
            
            result = eval(expression, safe_dict)
            
            return ToolResult(
                success=True,
                result=result,
                metadata={"expression": expression}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Calculation error: {str(e)}"
            )


class DateTimeTool(Tool):
    """Date and time utility tool."""
    
    name = "datetime"
    description = "Get current date/time or perform date calculations"
    parameters = {
        "properties": {
            "action": {
                "type": "string",
                "description": "Action to perform: 'now', 'format', 'add_days', 'add_hours'",
                "enum": ["now", "format", "add_days", "add_hours"]
            },
            "format": {
                "type": "string",
                "description": "Date format string (for 'format' action)"
            },
            "days": {
                "type": "integer",
                "description": "Number of days to add (for 'add_days' action)"
            },
            "hours": {
                "type": "integer",
                "description": "Number of hours to add (for 'add_hours' action)"
            }
        },
        "required": ["action"]
    }
    
    def execute(self, action: str, **kwargs) -> ToolResult:
        """Execute date/time operation."""
        try:
            from datetime import datetime, timedelta
            
            now = datetime.now()
            
            if action == "now":
                result = now.isoformat()
            elif action == "format":
                format_str = kwargs.get("format", "%Y-%m-%d %H:%M:%S")
                result = now.strftime(format_str)
            elif action == "add_days":
                days = kwargs.get("days", 0)
                result = (now + timedelta(days=days)).isoformat()
            elif action == "add_hours":
                hours = kwargs.get("hours", 0)
                result = (now + timedelta(hours=hours)).isoformat()
            else:
                raise ValueError(f"Unknown action: {action}")
            
            return ToolResult(
                success=True,
                result=result,
                metadata={"action": action, "timestamp": now.isoformat()}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=str(e)
            )


class TextProcessingTool(Tool):
    """Text processing utility tool."""
    
    name = "text_processing"
    description = "Perform various text processing operations"
    parameters = {
        "properties": {
            "action": {
                "type": "string",
                "description": "Action to perform: 'count_words', 'count_chars', 'extract_emails', 'extract_urls'",
                "enum": ["count_words", "count_chars", "extract_emails", "extract_urls", "summarize"]
            },
            "text": {
                "type": "string",
                "description": "Text to process"
            }
        },
        "required": ["action", "text"]
    }
    
    def execute(self, action: str, text: str) -> ToolResult:
        """Execute text processing operation."""
        try:
            if action == "count_words":
                result = len(text.split())
            elif action == "count_chars":
                result = len(text)
            elif action == "extract_emails":
                email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                result = re.findall(email_pattern, text)
            elif action == "extract_urls":
                url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                result = re.findall(url_pattern, text)
            elif action == "summarize":
                # Simple summarization - take first and last sentences
                sentences = text.split('.')
                if len(sentences) <= 2:
                    result = text
                else:
                    result = f"{sentences[0].strip()}. ... {sentences[-2].strip()}."
            else:
                raise ValueError(f"Unknown action: {action}")
            
            return ToolResult(
                success=True,
                result=result,
                metadata={"action": action, "text_length": len(text)}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=str(e)
            )


class RandomTool(Tool):
    """Random number and choice generation tool."""
    
    name = "random"
    description = "Generate random numbers, choices, and data"
    parameters = {
        "properties": {
            "action": {
                "type": "string",
                "description": "Action: 'number', 'choice', 'shuffle', 'uuid'",
                "enum": ["number", "choice", "shuffle", "uuid"]
            },
            "min": {"type": "number", "description": "Minimum value for random number"},
            "max": {"type": "number", "description": "Maximum value for random number"},
            "choices": {"type": "array", "description": "List of choices to pick from"},
            "items": {"type": "array", "description": "List of items to shuffle"}
        },
        "required": ["action"]
    }
    
    def execute(self, action: str, **kwargs) -> ToolResult:
        """Execute random operation."""
        try:
            import uuid
            
            if action == "number":
                min_val = kwargs.get("min", 0)
                max_val = kwargs.get("max", 100)
                result = random.uniform(min_val, max_val)
            elif action == "choice":
                choices = kwargs.get("choices", [])
                if not choices:
                    raise ValueError("No choices provided")
                result = random.choice(choices)
            elif action == "shuffle":
                items = kwargs.get("items", [])
                shuffled = items.copy()
                random.shuffle(shuffled)
                result = shuffled
            elif action == "uuid":
                result = str(uuid.uuid4())
            else:
                raise ValueError(f"Unknown action: {action}")
            
            return ToolResult(
                success=True,
                result=result,
                metadata={"action": action}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=str(e)
            )


class ToolRegistry:
    """Registry for managing tools."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        
        # Register built-in tools
        self.register(CalculatorTool())
        self.register(DateTimeTool())
        self.register(TextProcessingTool())
        self.register(RandomTool())
    
    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
    
    def register_function(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None) -> None:
        """Register a function as a tool."""
        tool = FunctionTool(func, name, description)
        self.register(tool)
    
    def get_tool(self, name: str) -> Tool:
        """Get a tool by name."""
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found in registry")
        return self._tools[name]
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def get_tool_info(self, name: str) -> Dict[str, Any]:
        """Get tool information including parameters."""
        tool = self.get_tool(name)
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters
        }
    
    def execute_tool(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        tool = self.get_tool(name)
        tool.validate_parameters(**kwargs)
        return tool.execute(**kwargs)


class ToolExecutor:
    """Handles dynamic tool execution and selection."""
    
    def __init__(self, registry: Optional[ToolRegistry] = None):
        self.registry = registry or ToolRegistry()
    
    def execute(self, tool_name: str, **parameters) -> ToolResult:
        """Execute a specific tool."""
        return self.registry.execute_tool(tool_name, **parameters)
    
    def execute_from_text(self, text: str) -> List[ToolResult]:
        """Parse tool calls from text and execute them."""
        # Simple pattern matching for tool calls
        # Format: TOOL[tool_name](param1=value1, param2=value2)
        pattern = r'TOOL\[(\w+)\]\(([^)]*)\)'
        matches = re.findall(pattern, text)
        
        results = []
        for tool_name, params_str in matches:
            try:
                # Parse parameters (simple key=value format)
                params = {}
                if params_str.strip():
                    for param in params_str.split(','):
                        if '=' in param:
                            key, value = param.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"\'')
                            
                            # Try to parse as JSON for complex types
                            try:
                                value = json.loads(value)
                            except:
                                pass  # Keep as string
                            
                            params[key] = value
                
                result = self.execute(tool_name, **params)
                results.append(result)
                
            except Exception as e:
                results.append(ToolResult(
                    success=False,
                    result=None,
                    error=f"Failed to execute {tool_name}: {str(e)}"
                ))
        
        return results
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get information about all available tools."""
        return [
            self.registry.get_tool_info(name)
            for name in self.registry.list_tools()
        ]


# Global tool registry and executor
tool_registry = ToolRegistry()
tool_executor = ToolExecutor(tool_registry)


def register_tool(tool: Tool) -> None:
    """Register a tool in the global registry."""
    tool_registry.register(tool)


def register_function_as_tool(func: Callable, name: Optional[str] = None, description: Optional[str] = None) -> None:
    """Register a function as a tool in the global registry."""
    tool_registry.register_function(func, name, description)


def execute_tool(name: str, **kwargs) -> ToolResult:
    """Execute a tool from the global registry."""
    return tool_executor.execute(name, **kwargs)


def get_available_tools() -> List[Dict[str, Any]]:
    """Get all available tools from the global registry."""
    return tool_executor.get_available_tools()