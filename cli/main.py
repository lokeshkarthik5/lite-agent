#!/usr/bin/env python3
"""
CLI interface for lite-agent framework.
Provides commands for managing agents, tools, and templates.
"""

import argparse
import sys
import json
import os
from typing import Dict, Any, Optional

try:
    import liteagent as la
except ImportError:
    print("Error: lite-agent package not found. Please install it first.")
    sys.exit(1)


def cmd_version(args) -> None:
    """Show version information."""
    la.version_info()


def cmd_test(args) -> None:
    """Test installation."""
    success = la.test_installation()
    sys.exit(0 if success else 1)


def cmd_quick_start(args) -> None:
    """Show quick start guide."""
    la.quick_start()


def cmd_list_agents(args) -> None:
    """List all registered agents."""
    agents = la.list_agents()
    
    if not agents:
        print("No agents registered.")
        return
    
    print(f"Registered Agents ({len(agents)}):")
    for agent_name in agents:
        agent_info = la.agent_registry.get_agent_info(agent_name)
        print(f"  • {agent_name}: {agent_info.get('description', 'No description')}")


def cmd_list_tools(args) -> None:
    """List all available tools."""
    tools = la.get_available_tools()
    
    if not tools:
        print("No tools available.")
        return
    
    print(f"Available Tools ({len(tools)}):")
    for tool in tools:
        print(f"  • {tool['name']}: {tool['description']}")


def cmd_list_templates(args) -> None:
    """List all registered templates."""
    templates = la.template_registry.list_templates()
    
    if not templates:
        print("No templates registered.")
        return
    
    print(f"Registered Templates ({len(templates)}):")
    for template_name in templates:
        template = la.get_template(template_name)
        print(f"  • {template_name}: {getattr(template, 'description', 'No description')}")


def cmd_agent_info(args) -> None:
    """Show detailed information about an agent."""
    try:
        agent_info = la.agent_registry.get_agent_info(args.name)
        print(f"Agent: {args.name}")
        print(f"Description: {agent_info.get('description', 'No description')}")
        print(f"Provider: {agent_info.get('provider', 'Unknown')}")
        print(f"Model: {agent_info.get('model', 'Unknown')}")
        print(f"Tools: {', '.join(agent_info.get('tools', []))}")
        print(f"Call Count: {agent_info.get('call_count', 0)}")
        print(f"Last Execution: {agent_info.get('last_execution', 'Never')}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_tool_info(args) -> None:
    """Show detailed information about a tool."""
    try:
        tool_info = la.tool_registry.get_tool_info(args.name)
        print(f"Tool: {args.name}")
        print(f"Description: {tool_info.get('description', 'No description')}")
        
        if 'parameters' in tool_info:
            params = tool_info['parameters']
            if 'properties' in params:
                print("\nParameters:")
                for param_name, param_info in params['properties'].items():
                    required = param_name in params.get('required', [])
                    req_str = " (required)" if required else " (optional)"
                    print(f"  • {param_name}: {param_info.get('type', 'unknown')}{req_str}")
                    if 'description' in param_info:
                        print(f"    {param_info['description']}")
                        
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_execute_tool(args) -> None:
    """Execute a tool with parameters."""
    try:
        # Parse parameters from JSON or key=value format
        params = {}
        if args.params:
            if args.params.startswith('{'):
                # JSON format
                params = json.loads(args.params)
            else:
                # key=value format
                for param in args.params.split(','):
                    if '=' in param:
                        key, value = param.split('=', 1)
                        # Try to parse as JSON value, fallback to string
                        try:
                            params[key.strip()] = json.loads(value.strip())
                        except:
                            params[key.strip()] = value.strip()
        
        result = la.execute_tool(args.name, **params)
        
        if result.success:
            print(f"Result: {result.result}")
            if result.metadata:
                print(f"Metadata: {result.metadata}")
        else:
            print(f"Error: {result.error}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_metrics(args) -> None:
    """Show execution metrics."""
    metrics = la.get_metrics()
    
    if not metrics:
        print("No metrics available. Enable metrics with: la.setup_metrics()")
        return
    
    print("Execution Metrics:")
    print(f"  Total Executions: {metrics.get('total_executions', 0)}")
    print(f"  Successful: {metrics.get('successful_executions', 0)}")
    print(f"  Failed: {metrics.get('failed_executions', 0)}")
    print(f"  Total Time: {metrics.get('total_execution_time', 0):.2f}s")
    
    if 'agent_stats' in metrics and metrics['agent_stats']:
        print("\nAgent Statistics:")
        for agent_name, stats in metrics['agent_stats'].items():
            print(f"  {agent_name}:")
            print(f"    Executions: {stats.get('executions', 0)}")
            print(f"    Success Rate: {stats.get('successes', 0)}/{stats.get('executions', 0)}")
            print(f"    Total Time: {stats.get('total_time', 0):.2f}s")
    
    if 'error_counts' in metrics and metrics['error_counts']:
        print("\nError Counts:")
        for error_type, count in metrics['error_counts'].items():
            print(f"  {error_type}: {count}")


def cmd_create_example(args) -> None:
    """Create example agent files."""
    examples_dir = "lite_agent_examples"
    os.makedirs(examples_dir, exist_ok=True)
    
    # Simple agent example
    simple_example = '''
"""
Simple Agent Example
Run with: python simple_agent.py
"""

import liteagent as la

# Setup logging
la.setup_logging()

@la.agent(
    name="assistant", 
    system_prompt="You are a helpful assistant. Be concise and accurate.",
    temperature=0.7
)
def assistant_agent(state: la.AgentState, question: str) -> str:
    """Answer questions helpfully and concisely."""
    return question

if __name__ == "__main__":
    # Create state
    state = la.AgentState()
    
    # Ask questions
    questions = [
        "What is Python?",
        "How do I install packages?",
        "What are some best practices for coding?"
    ]
    
    for question in questions:
        print(f"\\nQ: {question}")
        result = assistant_agent.run(state, question)
        if result.success:
            print(f"A: {result.result}")
        else:
            print(f"Error: {result.error}")
'''
    
    # Multi-agent example
    multi_example = '''
"""
Multi-Agent Orchestration Example
Run with: python multi_agent.py
"""

import liteagent as la
from dataclasses import dataclass

# Setup
la.setup_logging()
la.setup_metrics()

@dataclass
class Research:
    topic: str
    key_findings: list
    sources: list

@la.agent(
    name="researcher",
    system_prompt="You are a thorough researcher. Provide detailed findings.",
    tools=["text_processing", "datetime"],
    output_parser=Research
)
def researcher_agent(state: la.AgentState, topic: str) -> Research:
    """Research a topic and return structured findings."""
    return f"Research topic: {topic}"

@la.agent(
    name="writer",
    system_prompt="You are a skilled writer. Create engaging content based on research."
)
def writer_agent(state: la.AgentState, research: Research) -> str:
    """Write content based on research findings."""
    return f"Write about research on: {research.topic}"

@la.agent(
    name="editor",
    system_prompt="You are an editor. Improve and refine written content."
)
def editor_agent(state: la.AgentState, content: str) -> str:
    """Edit and improve written content."""
    return f"Edit content: {content[:100]}..."

if __name__ == "__main__":
    # Create shared state
    state = la.AgentState()
    state.set_context("project", "AI Research Article")
    
    # Multi-agent workflow
    topic = "The future of artificial intelligence"
    
    print(f"Starting research on: {topic}")
    
    # Step 1: Research
    research_result = researcher_agent.run(state, topic)
    if not research_result.success:
        print(f"Research failed: {research_result.error}")
        exit(1)
    
    # Step 2: Write
    write_result = writer_agent.run(state, research_result.result)
    if not write_result.success:
        print(f"Writing failed: {write_result.error}")
        exit(1)
    
    # Step 3: Edit
    edit_result = editor_agent.run(state, write_result.result)
    if not edit_result.success:
        print(f"Editing failed: {edit_result.error}")
        exit(1)
    
    print(f"\\nFinal Result:\\n{edit_result.result}")
    
    # Show metrics
    print("\\nMetrics:")
    metrics = la.get_metrics()
    print(f"Total executions: {metrics.get('total_executions', 0)}")
    print(f"Total time: {metrics.get('total_execution_time', 0):.2f}s")
'''
    
    # Tool usage example
    tool_example = '''
"""
Tool Usage Example
Run with: python tool_usage.py
"""

import liteagent as la

# Setup
la.setup_logging()

# Custom tool
def weather_tool(location: str, units: str = "celsius") -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: 22°{units[0].upper()}, sunny"

# Register custom tool
la.register_function_as_tool(weather_tool, "weather", "Get weather information")

@la.agent(
    name="travel_assistant",
    system_prompt="You are a travel assistant. Help with travel planning.",
    tools=["weather", "datetime", "calculator"]
)
def travel_agent(state: la.AgentState, destination: str) -> str:
    """Help plan travel to a destination."""
    return f"Planning travel to {destination}"

if __name__ == "__main__":
    state = la.AgentState()
    
    # Test tools directly
    print("Testing tools directly:")
    
    # Calculator
    calc_result = la.execute_tool("calculator", expression="10 * 2 + 5")
    print(f"Calculator: {calc_result.result}")
    
    # Weather (custom tool)
    weather_result = la.execute_tool("weather", location="Paris", units="celsius")
    print(f"Weather: {weather_result.result}")
    
    # Date/time
    date_result = la.execute_tool("datetime", action="now")
    print(f"Current time: {date_result.result}")
    
    # Use agent with tools
    print("\\nUsing agent with tools:")
    result = travel_agent.run(state, "Tokyo")
    if result.success:
        print(f"Agent result: {result.result}")
        if result.tool_results:
            print(f"Tools used: {len(result.tool_results)}")
    else:
        print(f"Error: {result.error}")
'''
    
    # Write examples
    examples = {
        "simple_agent.py": simple_example,
        "multi_agent.py": multi_example,
        "tool_usage.py": tool_example
    }
    
    for filename, content in examples.items():
        filepath = os.path.join(examples_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content.strip())
        print(f"Created: {filepath}")
    
    print(f"\\nExample files created in {examples_dir}/")
    print("Run them with: python lite_agent_examples/<filename>")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="lite-agent CLI - Lightweight agent framework",
        epilog="For more information, visit: https://github.com/lite-agent/lite-agent"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Version command
    subparsers.add_parser('version', help='Show version information')
    
    # Test command
    subparsers.add_parser('test', help='Test installation')
    
    # Quick start command
    subparsers.add_parser('quick-start', help='Show quick start guide')
    
    # List commands
    subparsers.add_parser('list-agents', help='List all registered agents')
    subparsers.add_parser('list-tools', help='List all available tools')
    subparsers.add_parser('list-templates', help='List all registered templates')
    
    # Info commands
    agent_info_parser = subparsers.add_parser('agent-info', help='Show agent information')
    agent_info_parser.add_argument('name', help='Agent name')
    
    tool_info_parser = subparsers.add_parser('tool-info', help='Show tool information')
    tool_info_parser.add_argument('name', help='Tool name')
    
    # Execute tool command
    exec_tool_parser = subparsers.add_parser('exec-tool', help='Execute a tool')
    exec_tool_parser.add_argument('name', help='Tool name')
    exec_tool_parser.add_argument('--params', help='Tool parameters (JSON or key=value,key=value)')
    
    # Metrics command
    subparsers.add_parser('metrics', help='Show execution metrics')
    
    # Create examples command
    subparsers.add_parser('create-examples', help='Create example agent files')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    command_map = {
        'version': cmd_version,
        'test': cmd_test,
        'quick-start': cmd_quick_start,
        'list-agents': cmd_list_agents,
        'list-tools': cmd_list_tools,
        'list-templates': cmd_list_templates,
        'agent-info': cmd_agent_info,
        'tool-info': cmd_tool_info,
        'exec-tool': cmd_execute_tool,
        'metrics': cmd_metrics,
        'create-examples': cmd_create_example
    }
    
    try:
        command_map[args.command](args)
    except KeyboardInterrupt:
        print("\\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()