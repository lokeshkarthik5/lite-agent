"""
Prompt template system for lite-agent framework.
Provides class-based prompt templates with complex formatting and defaults.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Type
from dataclasses import dataclass, field
import re
import json
from string import Template


class TemplateBase(ABC):
    """
    Abstract base class for prompt templates.
    Define input/output format structure and complex formatting logic.
    """
    
    # Template metadata
    name: str = ""
    description: str = ""
    version: str = "1.0"
    
    # Template configuration
    template: str = ""
    required_vars: List[str] = []
    optional_vars: Dict[str, Any] = {}
    
    @abstractmethod
    def format(self, **kwargs) -> str:
        """Format the template with provided variables."""
        pass
    
    def validate_inputs(self, **kwargs) -> None:
        """Validate that all required variables are provided."""
        missing = [var for var in self.required_vars if var not in kwargs]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
    
    def get_defaults(self) -> Dict[str, Any]:
        """Get default values for optional variables."""
        return self.optional_vars.copy()
    
    def render(self, **kwargs) -> str:
        """Render the template with validation and defaults."""
        # Apply defaults
        variables = self.get_defaults()
        variables.update(kwargs)
        
        # Validate required inputs
        self.validate_inputs(**variables)
        
        # Format template
        return self.format(**variables)


class SimpleTemplate(TemplateBase):
    """Simple string template using Python's string.Template."""
    
    def __init__(self, template: str, name: str = "", **defaults):
        self.template = template
        self.name = name or self.__class__.__name__
        self.optional_vars = defaults
        
        # Extract required variables from template
        self.required_vars = [
            match.group(1) for match in re.finditer(r'\$([a-zA-Z_][a-zA-Z0-9_]*)', template)
            if match.group(1) not in defaults
        ]
    
    def format(self, **kwargs) -> str:
        """Format using string.Template."""
        template = Template(self.template)
        return template.substitute(**kwargs)


class JinjaTemplate(TemplateBase):
    """Jinja2-based template for complex formatting."""
    
    def __init__(self, template: str, name: str = "", **defaults):
        try:
            import jinja2
        except ImportError:
            raise ImportError("jinja2 package is required for JinjaTemplate")
        
        self.jinja_env = jinja2.Environment()
        self.jinja_template = self.jinja_env.from_string(template)
        self.template = template
        self.name = name or self.__class__.__name__
        self.optional_vars = defaults
        
        # Extract required variables (simplified)
        var_pattern = r'\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)'
        self.required_vars = [
            match.group(1) for match in re.finditer(var_pattern, template)
            if match.group(1) not in defaults
        ]
    
    def format(self, **kwargs) -> str:
        """Format using Jinja2."""
        return self.jinja_template.render(**kwargs)


@dataclass
class ConversationTemplate(TemplateBase):
    """Template for conversation-style prompts with roles."""
    
    name: str = "ConversationTemplate"
    system_message: str = ""
    user_template: str = ""
    assistant_prefix: str = ""
    required_vars: List[str] = field(default_factory=list)
    optional_vars: Dict[str, Any] = field(default_factory=dict)
    
    def format(self, **kwargs) -> str:
        """Format as conversation with system, user, and assistant roles."""
        messages = []
        
        if self.system_message:
            system_formatted = Template(self.system_message).substitute(**kwargs)
            messages.append(f"System: {system_formatted}")
        
        if self.user_template:
            user_formatted = Template(self.user_template).substitute(**kwargs)
            messages.append(f"User: {user_formatted}")
        
        if self.assistant_prefix:
            assistant_formatted = Template(self.assistant_prefix).substitute(**kwargs)
            messages.append(f"Assistant: {assistant_formatted}")
        
        return "\n\n".join(messages)


class ResearchPrompt(TemplateBase):
    """Example research prompt template with structured input/output."""
    
    name = "ResearchPrompt"
    description = "Template for research tasks with structured output"
    version = "1.0"
    
    template = """
You are a research assistant tasked with analyzing the following topic: $topic

Research Context:
$context

Specific Questions:
$questions

Please provide a comprehensive analysis that includes:
1. Key findings
2. Supporting evidence
3. Potential limitations
4. Recommendations

Format your response as structured JSON with the following schema:
{
    "summary": "Brief overview of findings",
    "key_findings": ["finding1", "finding2", ...],
    "evidence": ["evidence1", "evidence2", ...],
    "limitations": ["limitation1", "limitation2", ...],
    "recommendations": ["rec1", "rec2", ...]
}
"""
    
    required_vars = ["topic", "questions"]
    optional_vars = {
        "context": "No additional context provided."
    }
    
    def format(self, **kwargs) -> str:
        """Format the research prompt."""
        template = Template(self.template)
        return template.substitute(**kwargs)


class CodeAnalysisPrompt(TemplateBase):
    """Template for code analysis tasks."""
    
    name = "CodeAnalysisPrompt"
    description = "Template for analyzing code with specific focus areas"
    version = "1.0"
    
    template = """
Analyze the following code for: $analysis_type

Code:
```$language
$code
```

Focus Areas:
$focus_areas

Additional Context:
$context

Please provide:
1. Overall assessment
2. Specific issues found
3. Improvement suggestions
4. Code quality score (1-10)

$output_format
"""
    
    required_vars = ["code", "analysis_type"]
    optional_vars = {
        "language": "python",
        "focus_areas": "- Code quality\n- Performance\n- Security\n- Maintainability",
        "context": "No additional context.",
        "output_format": "Format as markdown with clear sections."
    }
    
    def format(self, **kwargs) -> str:
        """Format the code analysis prompt."""
        template = Template(self.template)
        return template.substitute(**kwargs)


class TaskPlanningPrompt(TemplateBase):
    """Template for task decomposition and planning."""
    
    name = "TaskPlanningPrompt" 
    description = "Template for breaking down complex tasks into subtasks"
    version = "1.0"
    
    template = """
You need to create a detailed plan for the following task:

Main Task: $main_task

Context and Requirements:
$requirements

Constraints:
$constraints

Please create a step-by-step plan that includes:
1. Task breakdown into subtasks
2. Dependencies between tasks
3. Estimated timeline
4. Required resources
5. Risk assessment

Output Format:
```json
{
    "main_task": "$main_task",
    "subtasks": [
        {
            "id": 1,
            "name": "Subtask name",
            "description": "Detailed description",
            "dependencies": [],
            "estimated_time": "X hours/days",
            "resources_needed": [],
            "risks": []
        }
    ],
    "timeline": "Overall timeline estimate",
    "critical_path": [],
    "success_criteria": []
}
```
"""
    
    required_vars = ["main_task"]
    optional_vars = {
        "requirements": "Standard requirements apply.",
        "constraints": "No specific constraints."
    }
    
    def format(self, **kwargs) -> str:
        """Format the task planning prompt."""
        template = Template(self.template)
        return template.substitute(**kwargs)


class TemplateRegistry:
    """Registry for managing prompt templates."""
    
    def __init__(self):
        self._templates: Dict[str, Type[TemplateBase]] = {}
        self._instances: Dict[str, TemplateBase] = {}
        
        # Register built-in templates
        self.register("research", ResearchPrompt)
        self.register("code_analysis", CodeAnalysisPrompt)
        self.register("task_planning", TaskPlanningPrompt)
    
    def register(self, name: str, template_class: Type[TemplateBase]) -> None:
        """Register a template class."""
        self._templates[name] = template_class
    
    def get_template(self, name: str) -> TemplateBase:
        """Get a template instance by name."""
        if name not in self._instances:
            if name not in self._templates:
                raise ValueError(f"Template '{name}' not found in registry")
            self._instances[name] = self._templates[name]()
        return self._instances[name]
    
    def list_templates(self) -> List[str]:
        """List all registered template names."""
        return list(self._templates.keys())
    
    def create_simple_template(self, name: str, template: str, **defaults) -> SimpleTemplate:
        """Create and register a simple template."""
        template_instance = SimpleTemplate(template, name, **defaults)
        self._instances[name] = template_instance
        return template_instance
    
    def create_conversation_template(
        self,
        name: str,
        system_message: str = "",
        user_template: str = "",
        assistant_prefix: str = "",
        **defaults
    ) -> ConversationTemplate:
        """Create and register a conversation template."""
        template_instance = ConversationTemplate(
            name=name,
            system_message=system_message,
            user_template=user_template,
            assistant_prefix=assistant_prefix,
            optional_vars=defaults
        )
        self._instances[name] = template_instance
        return template_instance


# Global template registry
template_registry = TemplateRegistry()


def get_template(name: str) -> TemplateBase:
    """Get a template from the global registry."""
    return template_registry.get_template(name)


def register_template(name: str, template_class: Type[TemplateBase]) -> None:
    """Register a template in the global registry."""
    template_registry.register(name, template_class)


def create_template(template_str: str, name: str = "", **defaults) -> SimpleTemplate:
    """Create a simple template with optional registration."""
    template = SimpleTemplate(template_str, name, **defaults)
    if name:
        template_registry._instances[name] = template
    return template