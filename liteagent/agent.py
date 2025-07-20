dev
"""
Core agent decorator and orchestration system for lite-agent framework.
Provides @agent(...) decorator for function-based agent definition.
"""

from typing import Any, Dict, List, Optional, Union, Callable, Type
from dataclasses import dataclass, field
import functools
import inspect
from datetime import datetime

from .state import AgentState
from .model import LLMBackend, LLMMessage, LLMResponse, generate, create_backend, LLMProvider
from .templates import TemplateBase, get_template
from .tools import Tool, ToolExecutor, ToolResult, tool_executor
from .parsers import OutputParser, create_parser, ParseResult


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    description: str = ""
    provider: Union[str, LLMProvider] = LLMProvider.OPENAI
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    system_prompt: str = ""
    template: Optional[Union[str, TemplateBase]] = None
    tools: List[Union[str, Tool, Callable]] = field(default_factory=list)
    output_parser: Optional[Union[str, Type, OutputParser]] = None
    backend: Optional[LLMBackend] = None
    max_retries: int = 3
    timeout: Optional[float] = None


@dataclass 
class AgentResult:
    """Result from agent execution."""
    success: bool
    result: Any
    raw_response: Optional[str] = None
    parsed_result: Optional[ParseResult] = None
    tool_results: List[ToolResult] = field(default_factory=list)
    error: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class Agent:
    """
    Core Agent class supporting multi-agent orchestration.
    Can be used directly or through the @agent decorator.
    """
    
    def __init__(
        self,
        func: Callable,
        config: AgentConfig,
        registry: Optional['AgentRegistry'] = None
    ):
        self.func = func
        self.config = config
        self.registry = registry
        
        # Initialize components
        self._setup_backend()
        self._setup_tools()
        self._setup_parser()
        
        # Agent metadata
        self.call_count = 0
        self.last_execution = None
        
        # Update function metadata
        functools.update_wrapper(self, func)
    
    def _setup_backend(self) -> None:
        """Setup LLM backend."""
        if self.config.backend is None:
            self.backend = create_backend(
                self.config.provider,
                model=self.config.model
            )
        else:
            self.backend = self.config.backend
    
    def _setup_tools(self) -> None:
        """Setup tool executor with agent-specific tools."""
        self.tool_executor = ToolExecutor()
        
        # Add agent-specific tools
        for tool in self.config.tools:
            if isinstance(tool, str):
                # Reference to global tool
                global_tool = tool_executor.registry.get_tool(tool)
                self.tool_executor.registry.register(global_tool)
            elif isinstance(tool, Tool):
                self.tool_executor.registry.register(tool)
            elif callable(tool):
                self.tool_executor.registry.register_function(tool)
    
    def _setup_parser(self) -> None:
        """Setup output parser."""
        if self.config.output_parser is None:
            self.parser = None
        elif isinstance(self.config.output_parser, str):
            self.parser = create_parser(self.config.output_parser)
        elif isinstance(self.config.output_parser, type):
            self.parser = create_parser(self.config.output_parser)
        else:
            self.parser = self.config.output_parser
    
    def _prepare_prompt(self, state: AgentState, *args, **kwargs) -> str:
        """Prepare the prompt for LLM generation."""
        # Get function signature to map args to kwargs
        sig = inspect.signature(self.func)
        bound_args = sig.bind(state, *args, **kwargs)
        bound_args.apply_defaults()
        
        # Remove state from variables for template
        template_vars = {k: v for k, v in bound_args.arguments.items() if k != 'state'}
        
        # Add state context to template variables
        template_vars.update({
            'memory': state.memory,
            'context': state.context,
            'history': state.get_history(),
            'current_agent': state.current_agent
        })
        
        # Prepare prompt based on configuration
        if self.config.template:
            if isinstance(self.config.template, str):
                # Template name - get from registry
                template = get_template(self.config.template)
                prompt = template.render(**template_vars)
            else:
                # Template instance
                prompt = self.config.template.render(**template_vars)
        else:
            # Use function docstring or system prompt
            base_prompt = self.config.system_prompt or self.func.__doc__ or ""
            
            # Add variable context
            if template_vars:
                var_context = "\n\nContext:\n" + "\n".join(
                    f"- {k}: {v}" for k, v in template_vars.items()
                )
                prompt = base_prompt + var_context
            else:
                prompt = base_prompt
        
        return prompt
    
    def _execute_tools(self, text: str, state: AgentState) -> List[ToolResult]:
        """Execute any tools mentioned in the response."""
        tool_results = self.tool_executor.execute_from_text(text)
        
        # Store tool results in state
        for result in tool_results:
            if result.success and hasattr(result, 'metadata') and result.metadata:
                tool_name = result.metadata.get('tool_name', 'unknown')
                state.store_tool_result(tool_name, result.result)
        
        return tool_results
    
    def run(self, state: AgentState, *args, **kwargs) -> AgentResult:
        """
        Execute the agent with given state and arguments.
        This is the main entry point for agent execution.
        """
        start_time = datetime.now()
        
        try:
            # Push agent onto execution stack
            state.push_agent(self.config.name)
            
            # Prepare prompt
            prompt = self._prepare_prompt(state, *args, **kwargs)
            
            # Add system prompt and conversation context
            messages = []
            
            if self.config.system_prompt:
                messages.append(LLMMessage(role="system", content=self.config.system_prompt))
            
            # Add recent conversation history
            for msg in state.get_history(limit=10):
                messages.append(LLMMessage(
                    role=msg["role"],
                    content=msg["content"]
                ))
            
            # Add current prompt
            messages.append(LLMMessage(role="user", content=prompt))
            
            # Generate response
            llm_response = self.backend.generate(
                messages=messages,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # Execute any tools mentioned in response
            tool_results = self._execute_tools(llm_response.content, state)
            
            # Parse output if parser is configured
            parsed_result = None
            final_result = llm_response.content
            
            if self.parser:
                parsed_result = self.parser.parse(llm_response.content)
                if parsed_result.success:
                    final_result = parsed_result.data
            
            # Update conversation history
            state.add_to_history("user", prompt)
            state.add_to_history("assistant", llm_response.content, {
                "agent": self.config.name,
                "model": self.config.model,
                "usage": llm_response.usage
            })
            
            # Update call count
            self.call_count += 1
            self.last_execution = datetime.now()
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResult(
                success=True,
                result=final_result,
                raw_response=llm_response.content,
                parsed_result=parsed_result,
                tool_results=tool_results,
                execution_time=execution_time,
                metadata={
                    "agent": self.config.name,
                    "model": self.config.model,
                    "call_count": self.call_count,
                    "usage": llm_response.usage
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResult(
                success=False,
                result=None,
                error=str(e),
                execution_time=execution_time,
                metadata={
                    "agent": self.config.name,
                    "call_count": self.call_count
                }
            )
        
        finally:
            # Pop agent from execution stack
            state.pop_agent()
    
    def __call__(self, state: AgentState, *args, **kwargs) -> AgentResult:
        """Make agent callable."""
        return self.run(state, *args, **kwargs)
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "name": self.config.name,
            "description": self.config.description,
            "provider": self.config.provider.value if hasattr(self.config.provider, 'value') else self.config.provider,
            "model": self.config.model,
            "tools": [tool.name if hasattr(tool, 'name') else str(tool) for tool in self.config.tools],
            "call_count": self.call_count,
            "last_execution": self.last_execution.isoformat() if self.last_execution else None
        }


class AgentRegistry:
    """Registry for managing agents and enabling multi-agent orchestration."""
    
    def __init__(self):
        self._agents: Dict[str, Agent] = {}
    
    def register(self, agent: Agent) -> None:
        """Register an agent."""
        self._agents[agent.config.name] = agent
        agent.registry = self
    
    def get_agent(self, name: str) -> Agent:
        """Get an agent by name."""
        if name not in self._agents:
            raise ValueError(f"Agent '{name}' not found in registry")
        return self._agents[name]
    
    def list_agents(self) -> List[str]:
        """List all registered agent names."""
        return list(self._agents.keys())
    
    def execute_agent(self, name: str, state: AgentState, *args, **kwargs) -> AgentResult:
        """Execute an agent by name."""
        agent = self.get_agent(name)
        return agent.run(state, *args, **kwargs)
    
    def get_agent_info(self, name: str) -> Dict[str, Any]:
        """Get agent information."""
        agent = self.get_agent(name)
        return agent.get_info()
    
    def get_all_agents_info(self) -> List[Dict[str, Any]]:
        """Get information about all agents."""
        return [agent.get_info() for agent in self._agents.values()]


# Global agent registry
agent_registry = AgentRegistry()


def agent(
    name: str,
    description: str = "",
    provider: Union[str, LLMProvider] = LLMProvider.OPENAI,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    system_prompt: str = "",
    template: Optional[Union[str, TemplateBase]] = None,
    tools: Optional[List[Union[str, Tool, Callable]]] = None,
    output_parser: Optional[Union[str, Type, OutputParser]] = None,
    backend: Optional[LLMBackend] = None,
    max_retries: int = 3,
    timeout: Optional[float] = None,
    register: bool = True
) -> Callable:
    """
    Agent-as-a-Function decorator.
    
    Defines agents with @agent(...) decorator, encouraging clean, functional design.
    
    Args:
        name: Agent name
        description: Agent description
        provider: LLM provider (OpenAI, Anthropic, etc.)
        model: Model name
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        system_prompt: System prompt for the agent
        template: Prompt template (name or instance)
        tools: List of tools available to the agent
        output_parser: Output parser for structured responses
        backend: Pre-configured LLM backend
        max_retries: Maximum retry attempts
        timeout: Execution timeout
        register: Whether to register in global registry
    
    Returns:
        Decorated function that becomes an Agent
    """
    def decorator(func: Callable) -> Agent:
        config = AgentConfig(
            name=name,
            description=description or func.__doc__ or f"Agent: {name}",
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            template=template,
            tools=tools or [],
            output_parser=output_parser,
            backend=backend,
            max_retries=max_retries,
            timeout=timeout
        )
        
        agent_instance = Agent(func, config)
        
        if register:
            agent_registry.register(agent_instance)
        
        return agent_instance
    
    return decorator


def get_agent(name: str) -> Agent:
    """Get an agent from the global registry."""
    return agent_registry.get_agent(name)


def list_agents() -> List[str]:
    """List all registered agents."""
    return agent_registry.list_agents()


def execute_agent(name: str, state: AgentState, *args, **kwargs) -> AgentResult:
    """Execute an agent by name."""
    return agent_registry.execute_agent(name, state, *args, **kwargs)


# Convenience function for quick agent creation
def create_simple_agent(
    name: str,
    prompt: str,
    provider: Union[str, LLMProvider] = LLMProvider.OPENAI,
    model: str = "gpt-3.5-turbo",
    **kwargs
) -> Agent:
    """Create a simple agent with just a prompt."""
    @agent(
        name=name,
        system_prompt=prompt,
        provider=provider,
        model=model,
        **kwargs
    )
    def simple_agent_func(state: AgentState, user_input: str) -> str:
        """Simple agent function."""
        return user_input
    
    return simple_agent_func

from functools import wraps

class Agent:
    def __init__(self,name,func,model,prompt_template = None, tools = None):
        self.name = name
        self.func = func
        self.model = model
        self.prompt_template = prompt_template 
        self.tools = tools or None

    def run(self,input_data,state):
        prompt = self.prompt_template.format(input_data,state)
        response = self.model.generate(prompt)
        return self.func(response,state)

def agent(name,model,prompt_template=None,tools=None):
    def decorator(func):
        a = Agent(name,func,model,prompt_template,tools)
        @wraps(func)
        def wrapper(input_data,state):
            return a.run(input_data,state)
        wrapper.agent = a
        return wrapper
    return decorator
main
