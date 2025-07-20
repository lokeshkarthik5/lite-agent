#!/usr/bin/env python3
"""
Minimal test script for lite-agent framework.
Tests core functionality without external dependencies.
"""

import sys
from pathlib import Path

# Add the package to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_functionality():
    """Test core functionality without external dependencies."""
    print("🧪 Testing lite-agent core functionality...")
    
    try:
        # Test imports
        print("1. Testing imports...")
        from liteagent.state import AgentState
        from liteagent.templates import SimpleTemplate, create_template
        from liteagent.tools import CalculatorTool, execute_tool
        from liteagent.parsers import JSONParser, parse_json
        print("   ✅ Core imports successful")
        
        # Test AgentState
        print("2. Testing AgentState...")
        state = AgentState()
        state.set_memory("test", "value")
        assert state.get_memory("test") == "value"
        state.add_to_history("user", "Hello")
        assert len(state.get_history()) == 1
        print("   ✅ AgentState working")
        
        # Test Templates
        print("3. Testing templates...")
        template = create_template("Hello $name!", name="World")
        result = template.render(name="Alice")
        assert "Hello Alice!" == result
        print("   ✅ Templates working")
        
        # Test Tools
        print("4. Testing tools...")
        calc_tool = CalculatorTool()
        result = calc_tool.execute(expression="2 + 3")
        assert result.success
        assert result.result == 5
        print("   ✅ Tools working")
        
        # Test Parsers
        print("5. Testing parsers...")
        result = parse_json('{"name": "test", "value": 42}')
        assert result.success
        assert result.data["name"] == "test"
        assert result.data["value"] == 42
        print("   ✅ Parsers working")
        
        # Test Package Info
        print("6. Testing package structure...")
        import liteagent
        assert hasattr(liteagent, '__version__')
        assert hasattr(liteagent, 'AgentState')
        assert hasattr(liteagent, 'agent')
        print("   ✅ Package structure correct")
        
        print("\n🎉 All core tests passed!")
        print(f"lite-agent v{liteagent.__version__} is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_basic_usage():
    """Demonstrate basic usage without LLM calls."""
    print("\n📋 Basic Usage Demo")
    print("-" * 30)
    
    try:
        import liteagent as la
        
        # Demo 1: State Management
        print("Demo 1: State Management")
        state = la.AgentState()
        state.set_memory("user_name", "Alice")
        state.set_context("session_id", "demo_123")
        state.add_to_history("user", "Hello, I'm Alice!")
        
        print(f"  Memory: {state.memory}")
        print(f"  Context: {state.context}")
        print(f"  History: {len(state.get_history())} messages")
        
        # Demo 2: Tools
        print("\nDemo 2: Built-in Tools")
        calc_result = la.execute_tool("calculator", expression="15 * 8 + 7")
        print(f"  Calculator: 15 * 8 + 7 = {calc_result.result}")
        
        date_result = la.execute_tool("datetime", action="now")
        print(f"  Current time: {date_result.result}")
        
        text_result = la.execute_tool("text_processing", action="count_words", text="Hello world this is a test")
        print(f"  Word count: {text_result.result} words")
        
        # Demo 3: Templates
        print("\nDemo 3: Templates")
        template = la.create_template(
            "Hello $name! Welcome to $platform. Your role is $role.",
            name="User",
            platform="lite-agent",
            role="developer"
        )
        
        result = template.render(name="Alice", role="AI researcher")
        print(f"  Template output: {result}")
        
        # Demo 4: Parsers
        print("\nDemo 4: Parsers")
        json_data = '{"task": "Build an app", "priority": "high", "deadline": "2024-02-01"}'
        parsed = la.parse_json(json_data)
        if parsed.success:
            print(f"  Parsed JSON: {parsed.data}")
        
        # Demo 5: Available Tools
        print("\nDemo 5: Available Tools")
        tools = la.get_available_tools()
        print(f"  Available tools: {len(tools)}")
        for tool in tools[:3]:
            print(f"    - {tool['name']}: {tool['description']}")
        
        print("\n✅ Demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def main():
    """Run minimal tests and demo."""
    print("🚀 lite-agent Framework - Minimal Test & Demo")
    print("=" * 50)
    
    # Run tests
    test_success = test_basic_functionality()
    
    if test_success:
        # Run demo
        demo_success = demo_basic_usage()
        
        if demo_success:
            print("\n" + "=" * 50)
            print("🎉 SUCCESS: lite-agent framework is working correctly!")
            print("\nNext steps:")
            print("- Install dependencies: pip install openai anthropic")
            print("- Try the examples: python examples/basic_usage.py")
            print("- Use the CLI: python cli/main.py --help")
            return True
    
    print("\n❌ Tests failed. Please check the implementation.")
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)