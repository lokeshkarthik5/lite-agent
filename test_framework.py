#!/usr/bin/env python3
"""
Simple test script to verify lite-agent framework functionality.
Run with: python test_framework.py
"""

import sys
import os
from pathlib import Path

# Add the package to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import liteagent as la
        print("✅ Main package imported successfully")
        
        # Test core components
        from liteagent.state import AgentState
        from liteagent.agent import agent, Agent
        from liteagent.model import LLMProvider, generate
        from liteagent.templates import TemplateBase, create_template
        from liteagent.tools import Tool, execute_tool
        from liteagent.parsers import parse_json
        from liteagent.middleware import setup_logging
        
        print("✅ All core modules imported successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_state_management():
    """Test AgentState functionality."""
    print("\nTesting state management...")
    
    try:
        import liteagent as la
        
        state = la.AgentState()
        
        # Test memory
        state.set_memory("test_key", "test_value")
        assert state.get_memory("test_key") == "test_value"
        
        # Test context
        state.set_context("session", "test_session")
        assert state.get_context("session") == "test_session"
        
        # Test history
        state.add_to_history("user", "Hello")
        history = state.get_history()
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        
        print("✅ State management tests passed")
        return True
        
    except Exception as e:
        print(f"❌ State management test failed: {e}")
        return False

def test_tools():
    """Test tool system."""
    print("\nTesting tools...")
    
    try:
        import liteagent as la
        
        # Test built-in tools
        tools = la.get_available_tools()
        assert len(tools) > 0
        
        # Test calculator tool
        result = la.execute_tool("calculator", expression="2 + 3")
        assert result.success
        assert result.result == 5
        
        # Test custom tool registration
        def test_tool(input_text: str) -> str:
            return f"Processed: {input_text}"
        
        la.register_function_as_tool(test_tool, "test", "Test tool")
        
        result = la.execute_tool("test", input_text="hello")
        assert result.success
        assert result.result == "Processed: hello"
        
        print("✅ Tool tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Tool test failed: {e}")
        return False

def test_templates():
    """Test template system."""
    print("\nTesting templates...")
    
    try:
        import liteagent as la
        
        # Test simple template
        template = la.create_template(
            "Hello $name, welcome to $place!",
            name="template_test",
            place="Earth"
        )
        
        result = template.render(name="Alice")
        assert "Hello Alice" in result
        assert "Earth" in result
        
        # Test template registry
        templates = la.template_registry.list_templates()
        assert len(templates) > 0
        
        print("✅ Template tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Template test failed: {e}")
        return False

def test_parsers():
    """Test output parsers."""
    print("\nTesting parsers...")
    
    try:
        import liteagent as la
        
        # Test JSON parser
        json_text = '{"name": "Alice", "age": 30}'
        result = la.parse_json(json_text)
        assert result.success
        assert result.data["name"] == "Alice"
        assert result.data["age"] == 30
        
        # Test markdown parser
        md_text = "# Title\nContent here\n## Section\nMore content"
        result = la.parse_markdown(md_text)
        assert result.success
        assert "Title" in result.data
        assert "Section" in result.data
        
        print("✅ Parser tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Parser test failed: {e}")
        return False

def test_agent_decorator():
    """Test agent decorator functionality."""
    print("\nTesting agent decorator...")
    
    try:
        import liteagent as la
        
        # Create a simple agent
        @la.agent(
            name="test_agent",
            system_prompt="You are a test agent",
            register=False  # Don't register globally for test
        )
        def test_agent(state: la.AgentState, message: str) -> str:
            """Test agent function."""
            return f"Received: {message}"
        
        # Test agent creation
        assert isinstance(test_agent, la.Agent)
        assert test_agent.config.name == "test_agent"
        
        # Test agent info
        info = test_agent.get_info()
        assert info["name"] == "test_agent"
        assert info["call_count"] == 0
        
        print("✅ Agent decorator tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Agent decorator test failed: {e}")
        return False

def test_agent_registry():
    """Test agent registry."""
    print("\nTesting agent registry...")
    
    try:
        import liteagent as la
        
        # Create and register an agent
        @la.agent(
            name="registry_test_agent",
            system_prompt="Test agent for registry"
        )
        def registry_agent(state: la.AgentState, input_text: str) -> str:
            return f"Registry test: {input_text}"
        
        # Test registry
        agents = la.list_agents()
        assert "registry_test_agent" in agents
        
        # Test getting agent info
        agent_info = la.agent_registry.get_agent_info("registry_test_agent")
        assert agent_info["name"] == "registry_test_agent"
        
        print("✅ Agent registry tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Agent registry test failed: {e}")
        return False

def test_middleware():
    """Test middleware system."""
    print("\nTesting middleware...")
    
    try:
        import liteagent as la
        
        # Test logging setup
        la.setup_logging()
        
        # Test metrics setup
        metrics_middleware = la.setup_metrics()
        assert metrics_middleware is not None
        
        # Test metrics retrieval
        metrics = la.get_metrics()
        assert isinstance(metrics, dict)
        
        print("✅ Middleware tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Middleware test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 lite-agent Framework Test Suite")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_state_management,
        test_tools,
        test_templates,
        test_parsers,
        test_agent_decorator,
        test_agent_registry,
        test_middleware
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Framework is working correctly.")
        
        # Test package info
        try:
            import liteagent as la
            la.version_info()
        except Exception as e:
            print(f"Warning: Could not display version info: {e}")
        
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)