"""
lite-agent: A lightweight and simple agent framework for everyday use.

🚀 CORE FEATURES
- Agent-as-a-Function Decorator (@agent(...))
- LLM Integration (OpenAI, Claude, Mistral, Ollama)
- Unified generate() interface
- State-Based Memory (AgentState)
- Multi-Agent Orchestration
- Pluggable Tool Use
- Prompt Template System
- Output Parsers
- Middleware & Observability

🧩 EXTENSIBLE INFRASTRUCTURE
- Custom LLM Backends
- Tool Execution Framework
- Logging & Metrics
- Caching & Retry Logic
"""

__version__ = "0.1.0"
__author__ = "lite-agent"
__email__ = "hello@lite-agent.dev"

# Core components
from .state import AgentState
from .agent import (
    Agent,
    AgentConfig,
    AgentResult,
    AgentRegistry,
    agent,
    get_agent,
    list_agents,
    execute_agent,
    create_simple_agent,
    agent_registry
)

# LLM Integration
from .model import (
    LLMProvider,
    LLMBackend,
    LLMMessage,
    LLMResponse,
    OpenAIBackend,
    AnthropicBackend,
    OllamaBackend,
    create_backend,
    generate
)

# Templates
from .templates import (
    TemplateBase,
    SimpleTemplate,
    JinjaTemplate,
    ConversationTemplate,
    ResearchPrompt,
    CodeAnalysisPrompt,
    TaskPlanningPrompt,
    TemplateRegistry,
    template_registry,
    get_template,
    register_template,
    create_template
)

# Tools
from .tools import (
    Tool,
    ToolResult,
    FunctionTool,
    ToolRegistry,
    ToolExecutor,
    CalculatorTool,
    DateTimeTool,
    TextProcessingTool,
    RandomTool,
    tool_registry,
    tool_executor,
    register_tool,
    register_function_as_tool,
    execute_tool,
    get_available_tools
)

# Parsers
from .parsers import (
    OutputParser,
    ParseResult,
    JSONParser,
    MarkdownParser,
    CodeBlockParser,
    ListParser,
    PydanticParser,
    DataclassParser,
    ChainParser,
    create_parser,
    parse_json,
    parse_markdown,
    parse_code,
    parse_list
)

# Middleware
from .middleware import (
    Middleware,
    ExecutionContext,
    LoggingMiddleware,
    MetricsMiddleware,
    CachingMiddleware,
    RetryMiddleware,
    MiddlewareStack,
    global_middleware_stack,
    add_global_middleware,
    remove_global_middleware,
    setup_logging,
    setup_metrics,
    get_metrics
)

# Main exports for easy imports
__all__ = [
    # Core
    "AgentState",
    "Agent",
    "AgentConfig", 
    "AgentResult",
    "AgentRegistry",
    "agent",
    "get_agent",
    "list_agents",
    "execute_agent",
    "create_simple_agent",
    
    # LLM
    "LLMProvider",
    "LLMBackend",
    "LLMMessage",
    "LLMResponse",
    "OpenAIBackend",
    "AnthropicBackend", 
    "OllamaBackend",
    "create_backend",
    "generate",
    
    # Templates
    "TemplateBase",
    "SimpleTemplate",
    "JinjaTemplate",
    "ConversationTemplate",
    "ResearchPrompt",
    "CodeAnalysisPrompt",
    "TaskPlanningPrompt",
    "get_template",
    "register_template",
    "create_template",
    
    # Tools
    "Tool",
    "ToolResult",
    "FunctionTool",
    "CalculatorTool",
    "DateTimeTool",
    "TextProcessingTool", 
    "RandomTool",
    "register_tool",
    "register_function_as_tool",
    "execute_tool",
    "get_available_tools",
    
    # Parsers
    "OutputParser",
    "ParseResult",
    "JSONParser",
    "MarkdownParser",
    "CodeBlockParser",
    "ListParser",
    "PydanticParser",
    "DataclassParser",
    "ChainParser",
    "create_parser",
    "parse_json",
    "parse_markdown",
    "parse_code",
    "parse_list",
    
    # Middleware
    "Middleware",
    "ExecutionContext",
    "LoggingMiddleware",
    "MetricsMiddleware",
    "CachingMiddleware",
    "RetryMiddleware",
    "MiddlewareStack",
    "add_global_middleware",
    "remove_global_middleware",
    "setup_logging",
    "setup_metrics",
    "get_metrics"
]


def quick_start():
    """Quick start guide for lite-agent."""
    print("""
🚀 lite-agent Quick Start

1. Create your first agent:

```python
import liteagent as la

@la.agent(name="assistant", system_prompt="You are a helpful assistant.")
def my_agent(state: la.AgentState, question: str) -> str:
    \"\"\"Answer questions helpfully.\"\"\"
    return question

# Use the agent
state = la.AgentState()
result = my_agent.run(state, "What is Python?")
print(result.result)
```

2. Multi-agent orchestration:

```python
@la.agent(name="researcher", tools=["calculator", "datetime"])
def research_agent(state: la.AgentState, topic: str) -> str:
    \"\"\"Research a topic thoroughly.\"\"\"
    return f"Researching: {topic}"

@la.agent(name="writer")  
def writer_agent(state: la.AgentState, research: str) -> str:
    \"\"\"Write based on research.\"\"\"
    return f"Writing about: {research}"

# Orchestrate agents
state = la.AgentState()
research_result = research_agent.run(state, "AI trends")
final_result = writer_agent.run(state, research_result.result)
```

3. Add tools and structured output:

```python
from dataclasses import dataclass

@dataclass
class Summary:
    title: str
    key_points: list
    conclusion: str

@la.agent(
    name="summarizer",
    tools=["text_processing"],
    output_parser=Summary
)
def summarizer(state: la.AgentState, text: str) -> Summary:
    \"\"\"Summarize text into structured format.\"\"\"
    return text
```

For more examples, see the examples/ directory or visit the documentation.
    """)


def version_info():
    """Display version and feature information."""
    print(f"""
lite-agent v{__version__}

🚀 CORE FEATURES
✓ Agent-as-a-Function Decorator
✓ LLM Integration (OpenAI, Claude, Mistral, Ollama)
✓ State-Based Memory
✓ Multi-Agent Orchestration  
✓ Pluggable Tool Use
✓ Prompt Template System
✓ Output Parsers
✓ Middleware & Observability

🧩 EXTENSIBLE INFRASTRUCTURE
✓ Custom LLM Backends
✓ Tool Execution Framework
✓ Logging & Metrics
✓ Caching & Retry Logic

Registered Components:
- Agents: {len(agent_registry.list_agents())}
- Tools: {len(tool_registry.list_tools())}
- Templates: {len(template_registry.list_templates())}
    """)


# Convenience function for package testing
def test_installation():
    """Test that the package is properly installed."""
    try:
        state = AgentState()
        
        # Test basic functionality
        assert state is not None
        assert hasattr(state, 'memory')
        assert hasattr(state, 'context')
        
        # Test tool registry
        tools = get_available_tools()
        assert len(tools) > 0
        
        # Test templates
        templates = template_registry.list_templates()
        assert len(templates) > 0
        
        print("✅ lite-agent installation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False