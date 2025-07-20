"""
Output parsers for lite-agent framework.
Extract structured data from LLM responses with customizable parsing logic.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Type, get_type_hints
import re
import json
import yaml
from dataclasses import dataclass, fields, is_dataclass
try:
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    BaseModel = None
    ValidationError = None
    PYDANTIC_AVAILABLE = False
import ast


@dataclass
class ParseResult:
    """Result from parsing operation."""
    success: bool
    data: Any
    error: Optional[str] = None
    raw_text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class OutputParser(ABC):
    """Abstract base class for output parsers."""
    
    @abstractmethod
    def parse(self, text: str) -> ParseResult:
        """Parse text and extract structured data."""
        pass
    
    def format_instructions(self) -> str:
        """Return instructions for the LLM on how to format output."""
        return ""


class JSONParser(OutputParser):
    """Parse JSON from LLM output."""
    
    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        self.schema = schema
    
    def parse(self, text: str) -> ParseResult:
        """Extract and parse JSON from text."""
        try:
            # Try to find JSON in the text
            json_match = self._extract_json(text)
            if not json_match:
                return ParseResult(
                    success=False,
                    data=None,
                    error="No JSON found in text",
                    raw_text=text
                )
            
            # Parse JSON
            data = json.loads(json_match)
            
            # Validate against schema if provided
            if self.schema:
                self._validate_schema(data, self.schema)
            
            return ParseResult(
                success=True,
                data=data,
                raw_text=text,
                metadata={"json_start": text.find(json_match)}
            )
            
        except json.JSONDecodeError as e:
            return ParseResult(
                success=False,
                data=None,
                error=f"Invalid JSON: {str(e)}",
                raw_text=text
            )
        except Exception as e:
            return ParseResult(
                success=False,
                data=None,
                error=str(e),
                raw_text=text
            )
    
    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON string from text."""
        # Look for JSON code blocks
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        
        # Look for standalone JSON objects
        brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(brace_pattern, text, re.DOTALL)
        
        # Return the largest JSON-like string
        if matches:
            return max(matches, key=len)
        
        return None
    
    def _validate_schema(self, data: Any, schema: Dict[str, Any]) -> None:
        """Basic JSON schema validation."""
        if "type" in schema:
            expected_type = schema["type"]
            if expected_type == "object" and not isinstance(data, dict):
                raise ValueError(f"Expected object, got {type(data).__name__}")
            elif expected_type == "array" and not isinstance(data, list):
                raise ValueError(f"Expected array, got {type(data).__name__}")
        
        if "required" in schema and isinstance(data, dict):
            missing = [key for key in schema["required"] if key not in data]
            if missing:
                raise ValueError(f"Missing required fields: {missing}")
    
    def format_instructions(self) -> str:
        """Return JSON formatting instructions."""
        instructions = "Please format your response as valid JSON."
        if self.schema:
            instructions += f"\n\nRequired schema:\n```json\n{json.dumps(self.schema, indent=2)}\n```"
        return instructions


class MarkdownParser(OutputParser):
    """Parse structured data from Markdown."""
    
    def __init__(self, sections: Optional[List[str]] = None):
        self.sections = sections or []
    
    def parse(self, text: str) -> ParseResult:
        """Parse Markdown sections."""
        try:
            sections = self._extract_sections(text)
            
            # Filter to requested sections if specified
            if self.sections:
                sections = {k: v for k, v in sections.items() if k.lower() in [s.lower() for s in self.sections]}
            
            return ParseResult(
                success=True,
                data=sections,
                raw_text=text,
                metadata={"section_count": len(sections)}
            )
            
        except Exception as e:
            return ParseResult(
                success=False,
                data=None,
                error=str(e),
                raw_text=text
            )
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract sections from Markdown text."""
        sections = {}
        current_section = None
        current_content = []
        
        for line in text.split('\n'):
            # Check for headers
            if line.startswith('#'):
                # Save previous section
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = line.lstrip('#').strip()
                current_content = []
            else:
                if current_section:
                    current_content.append(line)
        
        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def format_instructions(self) -> str:
        """Return Markdown formatting instructions."""
        instructions = "Please format your response using Markdown headers for sections."
        if self.sections:
            section_list = '\n'.join(f"- {section}" for section in self.sections)
            instructions += f"\n\nRequired sections:\n{section_list}"
        return instructions


class CodeBlockParser(OutputParser):
    """Extract code blocks from text."""
    
    def __init__(self, language: Optional[str] = None):
        self.language = language
    
    def parse(self, text: str) -> ParseResult:
        """Extract code blocks."""
        try:
            blocks = self._extract_code_blocks(text)
            
            # Filter by language if specified
            if self.language:
                blocks = [block for block in blocks if block.get("language") == self.language]
            
            # Return single block if only one, otherwise list
            data = blocks[0] if len(blocks) == 1 else blocks
            
            return ParseResult(
                success=True,
                data=data,
                raw_text=text,
                metadata={"block_count": len(blocks)}
            )
            
        except Exception as e:
            return ParseResult(
                success=False,
                data=None,
                error=str(e),
                raw_text=text
            )
    
    def _extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """Extract code blocks with language info."""
        blocks = []
        
        # Pattern for fenced code blocks
        pattern = r'```(\w*)\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for language, code in matches:
            blocks.append({
                "language": language or "text",
                "code": code.strip()
            })
        
        return blocks
    
    def format_instructions(self) -> str:
        """Return code formatting instructions."""
        lang_spec = f" in {self.language}" if self.language else ""
        return f"Please format your code{lang_spec} using fenced code blocks with language specification."


class ListParser(OutputParser):
    """Parse lists from text."""
    
    def __init__(self, item_type: Type = str):
        self.item_type = item_type
    
    def parse(self, text: str) -> ParseResult:
        """Extract list items."""
        try:
            items = self._extract_list_items(text)
            
            # Convert items to specified type
            if self.item_type != str:
                items = [self._convert_item(item, self.item_type) for item in items]
            
            return ParseResult(
                success=True,
                data=items,
                raw_text=text,
                metadata={"item_count": len(items)}
            )
            
        except Exception as e:
            return ParseResult(
                success=False,
                data=None,
                error=str(e),
                raw_text=text
            )
    
    def _extract_list_items(self, text: str) -> List[str]:
        """Extract list items from text."""
        items = []
        
        # Look for numbered lists
        numbered_pattern = r'^\d+\.\s*(.+)$'
        numbered_matches = re.findall(numbered_pattern, text, re.MULTILINE)
        if numbered_matches:
            items.extend(numbered_matches)
        
        # Look for bullet lists
        bullet_pattern = r'^[-*•]\s*(.+)$'
        bullet_matches = re.findall(bullet_pattern, text, re.MULTILINE)
        if bullet_matches:
            items.extend(bullet_matches)
        
        # If no structured lists found, split by lines
        if not items:
            items = [line.strip() for line in text.split('\n') if line.strip()]
        
        return items
    
    def _convert_item(self, item: str, target_type: Type) -> Any:
        """Convert string item to target type."""
        if target_type == int:
            return int(item)
        elif target_type == float:
            return float(item)
        elif target_type == bool:
            return item.lower() in ('true', 'yes', '1', 'on')
        else:
            return item
    
    def format_instructions(self) -> str:
        """Return list formatting instructions."""
        return "Please format your response as a numbered or bulleted list."


class PydanticParser(OutputParser):
    """Parse output into Pydantic models."""
    
    def __init__(self, model_class: Type[BaseModel]):
        if not PYDANTIC_AVAILABLE:
            raise ImportError("pydantic package is required for PydanticParser")
        self.model_class = model_class
    
    def parse(self, text: str) -> ParseResult:
        """Parse text into Pydantic model."""
        try:
            # First try to extract JSON
            json_parser = JSONParser()
            json_result = json_parser.parse(text)
            
            if not json_result.success:
                return ParseResult(
                    success=False,
                    data=None,
                    error=f"Failed to extract JSON: {json_result.error}",
                    raw_text=text
                )
            
            # Create model instance
            model_instance = self.model_class(**json_result.data)
            
            return ParseResult(
                success=True,
                data=model_instance,
                raw_text=text,
                metadata={"model_type": self.model_class.__name__}
            )
            
        except ValidationError as e:
            return ParseResult(
                success=False,
                data=None,
                error=f"Validation error: {str(e)}",
                raw_text=text
            )
        except Exception as e:
            return ParseResult(
                success=False,
                data=None,
                error=str(e),
                raw_text=text
            )
    
    def format_instructions(self) -> str:
        """Return Pydantic model formatting instructions."""
        schema = self.model_class.model_json_schema()
        return f"Please format your response as JSON matching this schema:\n```json\n{json.dumps(schema, indent=2)}\n```"


class DataclassParser(OutputParser):
    """Parse output into dataclass instances."""
    
    def __init__(self, dataclass_type: Type):
        if not is_dataclass(dataclass_type):
            raise ValueError("Provided type is not a dataclass")
        self.dataclass_type = dataclass_type
    
    def parse(self, text: str) -> ParseResult:
        """Parse text into dataclass."""
        try:
            # Extract JSON
            json_parser = JSONParser()
            json_result = json_parser.parse(text)
            
            if not json_result.success:
                return ParseResult(
                    success=False,
                    data=None,
                    error=f"Failed to extract JSON: {json_result.error}",
                    raw_text=text
                )
            
            # Create dataclass instance
            instance = self.dataclass_type(**json_result.data)
            
            return ParseResult(
                success=True,
                data=instance,
                raw_text=text,
                metadata={"dataclass_type": self.dataclass_type.__name__}
            )
            
        except Exception as e:
            return ParseResult(
                success=False,
                data=None,
                error=str(e),
                raw_text=text
            )
    
    def format_instructions(self) -> str:
        """Return dataclass formatting instructions."""
        field_names = [f.name for f in fields(self.dataclass_type)]
        return f"Please format your response as JSON with these fields: {', '.join(field_names)}"


class ChainParser(OutputParser):
    """Chain multiple parsers together."""
    
    def __init__(self, parsers: List[OutputParser]):
        self.parsers = parsers
    
    def parse(self, text: str) -> ParseResult:
        """Try parsers in sequence until one succeeds."""
        errors = []
        
        for parser in self.parsers:
            result = parser.parse(text)
            if result.success:
                return result
            errors.append(f"{parser.__class__.__name__}: {result.error}")
        
        return ParseResult(
            success=False,
            data=None,
            error=f"All parsers failed: {'; '.join(errors)}",
            raw_text=text
        )
    
    def format_instructions(self) -> str:
        """Return combined formatting instructions."""
        instructions = []
        for parser in self.parsers:
            inst = parser.format_instructions()
            if inst:
                instructions.append(inst)
        return "\n\nOR\n\n".join(instructions)


def create_parser(
    output_type: Union[str, Type, Dict[str, Any]],
    **kwargs
) -> OutputParser:
    """Factory function to create parsers based on type specification."""
    
    if isinstance(output_type, str):
        if output_type.lower() == "json":
            return JSONParser(**kwargs)
        elif output_type.lower() == "markdown":
            return MarkdownParser(**kwargs)
        elif output_type.lower() == "code":
            return CodeBlockParser(**kwargs)
        elif output_type.lower() == "list":
            return ListParser(**kwargs)
        else:
            raise ValueError(f"Unknown parser type: {output_type}")
    
    elif isinstance(output_type, dict):
        # JSON schema
        return JSONParser(schema=output_type, **kwargs)
    
    elif isinstance(output_type, type):
        if PYDANTIC_AVAILABLE and BaseModel and issubclass(output_type, BaseModel):
            return PydanticParser(output_type, **kwargs)
        elif is_dataclass(output_type):
            return DataclassParser(output_type, **kwargs)
        else:
            raise ValueError(f"Unsupported type: {output_type}")
    
    else:
        raise ValueError(f"Invalid output_type: {output_type}")


# Convenience functions
def parse_json(text: str, schema: Optional[Dict[str, Any]] = None) -> ParseResult:
    """Parse JSON from text."""
    return JSONParser(schema).parse(text)


def parse_markdown(text: str, sections: Optional[List[str]] = None) -> ParseResult:
    """Parse Markdown sections from text."""
    return MarkdownParser(sections).parse(text)


def parse_code(text: str, language: Optional[str] = None) -> ParseResult:
    """Parse code blocks from text."""
    return CodeBlockParser(language).parse(text)


def parse_list(text: str, item_type: Type = str) -> ParseResult:
    """Parse list items from text."""
    return ListParser(item_type).parse(text)