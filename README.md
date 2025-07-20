# 🚀 lite-agent

A lightweight and simple agent framework for everyday use.

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/lite-agent/lite-agent)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)

## ✨ Overview

lite-agent is a modern, lightweight framework for building AI agents with clean, functional design patterns. It provides a simple yet powerful foundation for creating intelligent applications without the complexity of larger frameworks.

## 🎯 Core Features

### 🔧 Agent-as-a-Function Decorator
Define agents with a simple `@agent(...)` decorator that encourages clean, functional design:

```python
import liteagent as la

@la.agent(name="assistant", system_prompt="You are a helpful assistant.")
def my_agent(state: la.AgentState, question: str) -> str:
    """Answer questions helpfully."""
    return question

# Use the agent
state = la.AgentState()
result = my_agent.run(state, "What is Python?")
print(result.result)
```

### 🤖 LLM Integration
Plug-and-play support for multiple LLM providers with a unified interface:

- **OpenAI** (GPT-3.5, GPT-4, etc.)
- **Anthropic** (Claude models)
- **Mistral** (Mistral models)
- **Ollama** (Local models)
- **Custom backends** (Easy to extend)

```python
# Different providers, same interface
@la.agent(name="gpt_agent", provider="openai", model="gpt-4")
def gpt_agent(state, query): return query

@la.agent(name="claude_agent", provider="anthropic", model="claude-3-sonnet-20240229")
def claude_agent(state, query): return query

@la.agent(name="local_agent", provider="ollama", model="llama2")
def local_agent(state, query): return query
```

### 💾 State-Based Memory
AgentState object for managing short-term memory and context:

```python
state = la.AgentState()

# Memory management
state.set_memory("user_preference", "dark_mode")
state.update_memory({"session_id": "123", "language": "en"})

# Context and history
state.set_context("current_task", "data_analysis")
state.add_to_history("user", "I need help with Python")

# Agent execution tracking
state.push_agent("analyzer")
print(f"Current agent: {state.current_agent}")
```

### 🔀 Multi-Agent Orchestration
Call agents within agents for complex workflows:

```python
@la.agent(name="researcher", tools=["text_processing"])
def researcher(state: la.AgentState, topic: str) -> str:
    """Research a topic thoroughly."""
    return f"Research findings on: {topic}"

@la.agent(name="writer")
def writer(state: la.AgentState, research: str) -> str:
    """Write based on research."""
    return f"Article based on: {research}"

# Orchestrate agents
state = la.AgentState()
research_result = researcher.run(state, "AI trends")
article_result = writer.run(state, research_result.result)
```

### 🛠️ Pluggable Tool System
Built-in tools and easy custom tool registration:

```python
# Use built-in tools
@la.agent(name="math_helper", tools=["calculator", "datetime"])
def math_agent(state: la.AgentState, problem: str) -> str:
    """Solve math problems with tools."""
    return problem

# Create custom tools
def weather_api(location: str) -> str:
    """Get weather for a location."""
    return f"Weather in {location}: 22°C, sunny"

la.register_function_as_tool(weather_api, "weather", "Get weather information")

@la.agent(name="travel_agent", tools=["weather", "calculator"])
def travel_agent(state: la.AgentState, destination: str) -> str:
    """Plan travel with weather info."""
    return destination
```

### 📝 Prompt Template System
Class-based templates with complex formatting:

```python
# Simple templates
template = la.create_template(
    "You are a $role working on $project. Task: $task",
    role="engineer",
    project="lite-agent"
)

@la.agent(name="project_agent", template=template)
def project_agent(state: la.AgentState, task: str, role: str = "developer") -> str:
    return task

# Built-in templates
@la.agent(name="researcher", template="research")
def research_agent(state: la.AgentState, topic: str, context: str) -> str:
    return topic
```

### 📊 Output Parsers
Extract structured data from LLM responses:

```python
from dataclasses import dataclass
from typing import List

@dataclass
class TaskPlan:
    title: str
    steps: List[str]
    priority: str

@la.agent(
    name="planner",
    output_parser=TaskPlan,
    system_prompt="Create structured task plans."
)
def planning_agent(state: la.AgentState, goal: str) -> TaskPlan:
    """Create a structured plan for achieving a goal."""
    return goal

# Usage
result = planning_agent.run(state, "Build a web app")
plan = result.result  # TaskPlan object
print(f"Title: {plan.title}")
print(f"Steps: {plan.steps}")
```

## 🧩 Extensible Infrastructure

### 📈 Logging & Observability
Built-in logging and metrics collection:

```python
# Setup logging
la.setup_logging(level=logging.INFO, log_file="agents.log")

# Setup metrics
metrics = la.setup_metrics()

# Run agents...
agent.run(state, "input")

# View metrics
print(la.get_metrics())
```

### 🔄 Middleware System
Customize agent execution pipeline:

```python
# Add caching
cache_middleware = la.CachingMiddleware(cache_size=1000, ttl_seconds=3600)
la.add_global_middleware(cache_middleware)

# Add retry logic
retry_middleware = la.RetryMiddleware(max_retries=3, backoff_factor=1.5)
la.add_global_middleware(retry_middleware)
```

### 🎛️ Custom LLM Backends
Easy integration of new LLM providers:

```python
class CustomBackend(la.LLMBackend):
    def generate(self, messages, model, **kwargs):
        # Your custom implementation
        return la.LLMResponse(content="response", model=model, provider="custom")

# Use custom backend
backend = CustomBackend()
@la.agent(name="custom_agent", backend=backend)
def custom_agent(state, input_text): return input_text
```

## 🚀 Quick Start

### Installation

```bash
pip install lite-agent

# Or install with extras
pip install lite-agent[dev,web,local]
```

### Basic Usage

1. **Create your first agent:**

```python
import liteagent as la

@la.agent(name="assistant", system_prompt="You are a helpful assistant.")
def my_agent(state: la.AgentState, question: str) -> str:
    """Answer questions helpfully."""
    return question

state = la.AgentState()
result = my_agent.run(state, "What is Python?")
print(result.result)
```

2. **Multi-agent workflow:**

```python
@la.agent(name="researcher", tools=["calculator", "datetime"])
def research_agent(state: la.AgentState, topic: str) -> str:
    return f"Researching: {topic}"

@la.agent(name="writer")  
def writer_agent(state: la.AgentState, research: str) -> str:
    return f"Writing about: {research}"

# Orchestrate
state = la.AgentState()
research = research_agent.run(state, "AI trends")
article = writer_agent.run(state, research.result)
```

3. **Structured output:**

```python
from dataclasses import dataclass

@dataclass
class Summary:
    title: str
    key_points: list
    conclusion: str

@la.agent(name="summarizer", output_parser=Summary)
def summarizer(state: la.AgentState, text: str) -> Summary:
    return text
```

## 🛠️ CLI Interface

lite-agent includes a comprehensive CLI:

```bash
# Show version and features
lite-agent version

# Test installation
lite-agent test

# List components
lite-agent list-agents
lite-agent list-tools
lite-agent list-templates

# Get component info
lite-agent agent-info my_agent
lite-agent tool-info calculator

# Execute tools directly
lite-agent exec-tool calculator --params "expression=2+3*4"

# View metrics
lite-agent metrics

# Create example files
lite-agent create-examples
```

## 📚 Examples

Check out the `examples/` directory for comprehensive examples:

- **Basic Usage** (`examples/basic_usage.py`): Complete feature demonstration
- **Multi-Agent Systems**: Agent orchestration patterns
- **Custom Tools**: Building and integrating custom tools
- **Template Systems**: Advanced prompt engineering
- **Output Parsing**: Structured data extraction

Run examples:
```bash
python examples/basic_usage.py
```

Or create examples in your directory:
```bash
lite-agent create-examples
python lite_agent_examples/simple_agent.py
```

## 🏗️ Architecture

lite-agent is built with clean separation of concerns:

```
liteagent/
├── agent.py          # Core @agent decorator and Agent class
├── state.py          # AgentState for memory management  
├── model.py          # LLM integration and backends
├── templates.py      # Prompt template system
├── tools.py          # Tool execution framework
├── parsers.py        # Output parsing and extraction
├── middleware.py     # Execution pipeline and observability
└── __init__.py       # Clean public API
```

## 🎯 Use Cases

- **Chatbots & Virtual Assistants**
- **Content Generation Pipelines**
- **Data Analysis & Research Tools**
- **Task Automation & Workflows**
- **Multi-step Decision Making**
- **Code Generation & Review**
- **Document Processing**
- **Customer Support Automation**

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by modern agent frameworks
- Built with love for the AI community
- Thanks to all contributors and users

## 📞 Support

- **Documentation**: [docs.lite-agent.dev](https://docs.lite-agent.dev)
- **Issues**: [GitHub Issues](https://github.com/lite-agent/lite-agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/lite-agent/lite-agent/discussions)
- **Discord**: [Join our community](https://discord.gg/lite-agent)

---

**Made with ❤️ by the lite-agent team**

*Building the future of lightweight AI agents, one decorator at a time.*