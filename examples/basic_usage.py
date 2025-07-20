"""
Comprehensive Example: lite-agent Framework Features
This example demonstrates all major features of the lite-agent framework.
"""

import liteagent as la
from dataclasses import dataclass
from typing import List

# Setup logging and metrics
la.setup_logging()
metrics = la.setup_metrics()

print("🚀 lite-agent Framework Demo")
print("=" * 50)

# 1. Basic Agent Creation
print("\n1. Basic Agent with @agent decorator")

@la.agent(
    name="basic_assistant",
    description="A simple helpful assistant",
    system_prompt="You are a helpful assistant. Be concise and accurate.",
    temperature=0.7
)
def basic_agent(state: la.AgentState, question: str) -> str:
    """Answer questions helpfully."""
    return question

# Test basic agent
state = la.AgentState()
result = basic_agent.run(state, "What is Python?")
print(f"Q: What is Python?")
print(f"A: {result.result[:100]}...")

# 2. Agent with Tools
print("\n2. Agent with Built-in Tools")

@la.agent(
    name="calculator_agent",
    system_prompt="You are a math assistant. Use the calculator for computations.",
    tools=["calculator", "datetime"]
)
def math_agent(state: la.AgentState, problem: str) -> str:
    """Solve mathematical problems using tools."""
    return f"Solving: {problem}"

# Test tool usage
result = math_agent.run(state, "What is 15 * 23 + 7?")
print(f"Math problem result: {result.result}")
print(f"Tools used: {len(result.tool_results)}")

# 3. Custom Tools
print("\n3. Custom Tool Registration")

def sentiment_analysis(text: str) -> str:
    """Analyze sentiment of text (mock implementation)."""
    positive_words = ["good", "great", "excellent", "amazing", "wonderful"]
    negative_words = ["bad", "terrible", "awful", "horrible", "worst"]
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return "Positive"
    elif negative_count > positive_count:
        return "Negative"
    else:
        return "Neutral"

# Register custom tool
la.register_function_as_tool(sentiment_analysis, "sentiment", "Analyze text sentiment")

@la.agent(
    name="sentiment_agent",
    tools=["sentiment", "text_processing"]
)
def sentiment_agent(state: la.AgentState, text: str) -> str:
    """Analyze sentiment and text properties."""
    return f"Analyzing: {text}"

result = sentiment_agent.run(state, "This is a wonderful day!")
print(f"Sentiment analysis: {result.result}")

# 4. Structured Output with Parsers
print("\n4. Structured Output Parsing")

@dataclass
class TaskAnalysis:
    title: str
    priority: str
    estimated_time: str
    dependencies: List[str]
    
@la.agent(
    name="task_analyzer",
    system_prompt="Analyze tasks and return structured information.",
    output_parser=TaskAnalysis
)
def task_agent(state: la.AgentState, task_description: str) -> TaskAnalysis:
    """Analyze a task and return structured information."""
    return task_description

# Test structured output
result = task_agent.run(state, "Create a web API for user management")
print(f"Task analysis result type: {type(result.result)}")
if result.parsed_result and result.parsed_result.success:
    analysis = result.parsed_result.data
    print(f"Title: {analysis.title}")
    print(f"Priority: {analysis.priority}")

# 5. Prompt Templates
print("\n5. Prompt Templates")

# Create custom template
template = la.create_template(
    """You are a $role working on $project.
    
Task: $task
Context: $context

Please provide a detailed response following these guidelines:
- Be specific and actionable
- Consider the project context
- Provide examples where relevant""",
    name="project_template",
    role="software engineer",
    context="No additional context provided"
)

@la.agent(
    name="project_agent",
    template="project_template"
)
def project_agent(state: la.AgentState, task: str, project: str, role: str = "developer") -> str:
    """Handle project-related tasks with context."""
    return task

# Test template usage
result = project_agent.run(
    state, 
    task="Implement user authentication", 
    project="E-commerce platform",
    role="senior developer"
)
print(f"Project agent response: {result.result[:100]}...")

# 6. Multi-Agent Orchestration
print("\n6. Multi-Agent Orchestration")

@la.agent(
    name="planner",
    system_prompt="You are a project planner. Break down tasks into steps."
)
def planner_agent(state: la.AgentState, goal: str) -> str:
    """Plan how to achieve a goal."""
    # Store planning result in state
    state.set_memory("plan", f"Plan for: {goal}")
    return f"Created plan for: {goal}"

@la.agent(
    name="executor",
    system_prompt="You are a task executor. Execute based on plans."
)
def executor_agent(state: la.AgentState, step: str) -> str:
    """Execute a planned step."""
    plan = state.get_memory("plan", "No plan available")
    return f"Executing step '{step}' based on plan: {plan}"

# Multi-agent workflow
print("Multi-agent workflow:")
goal = "Build a todo app"

# Step 1: Plan
plan_result = planner_agent.run(state, goal)
print(f"1. Planner: {plan_result.result}")

# Step 2: Execute
exec_result = executor_agent.run(state, "Set up project structure")
print(f"2. Executor: {exec_result.result}")

# 7. State Management and Memory
print("\n7. State Management")

# Demonstrate state features
state.set_memory("user_preference", "dark_mode")
state.set_context("session_id", "user_123")
state.add_to_history("user", "I prefer dark mode", {"source": "preferences"})

print(f"Memory: {state.memory}")
print(f"Context: {state.context}")
print(f"History length: {len(state.get_history())}")
print(f"Current agent: {state.current_agent}")

# 8. Agent Registry and Info
print("\n8. Agent Registry")

agents = la.list_agents()
print(f"Registered agents: {agents}")

# Get agent info
if agents:
    info = la.agent_registry.get_agent_info(agents[0])
    print(f"Agent '{agents[0]}' info:")
    print(f"  - Description: {info['description']}")
    print(f"  - Call count: {info['call_count']}")
    print(f"  - Tools: {info['tools']}")

# 9. Available Tools
print("\n9. Available Tools")

tools = la.get_available_tools()
print(f"Available tools ({len(tools)}):")
for tool in tools[:3]:  # Show first 3
    print(f"  - {tool['name']}: {tool['description']}")

# 10. Execution Metrics
print("\n10. Execution Metrics")

final_metrics = la.get_metrics()
print("Final execution metrics:")
print(f"  - Total executions: {final_metrics.get('total_executions', 0)}")
print(f"  - Successful: {final_metrics.get('successful_executions', 0)}")
print(f"  - Failed: {final_metrics.get('failed_executions', 0)}")
print(f"  - Total time: {final_metrics.get('total_execution_time', 0):.2f}s")

if final_metrics.get('agent_stats'):
    print("  - Agent statistics:")
    for agent_name, stats in final_metrics['agent_stats'].items():
        print(f"    • {agent_name}: {stats['executions']} executions")

print("\n" + "=" * 50)
print("✅ Demo completed! All features demonstrated successfully.")
print("\nNext steps:")
print("- Explore the examples/ directory for more use cases")
print("- Check the CLI with: lite-agent --help")
print("- Visit documentation for advanced features")