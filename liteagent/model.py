"""
LLM integration module for lite-agent framework.
Provides unified generate() interface for multiple LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from dataclasses import dataclass
import os
from enum import Enum


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"  # Claude
    MISTRAL = "mistral"
    OLLAMA = "ollama"
    CUSTOM = "custom"


@dataclass
class LLMResponse:
    """Response from LLM generation."""
    content: str
    model: str
    provider: LLMProvider
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMMessage:
    """Standard message format for LLM interactions."""
    role: str  # "system", "user", "assistant"
    content: str
    metadata: Optional[Dict[str, Any]] = None


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    @abstractmethod
    def generate(
        self,
        messages: List[LLMMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    async def generate_async(
        self,
        messages: List[LLMMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM asynchronously."""
        pass
    
    @abstractmethod
    def stream(
        self,
        messages: List[LLMMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream response tokens from the LLM."""
        pass


class OpenAIBackend(LLMBackend):
    """OpenAI LLM backend."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        try:
            import openai
        except ImportError:
            raise ImportError("openai package is required for OpenAI backend")
        
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url
        )
    
    def generate(
        self,
        messages: List[LLMMessage],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using OpenAI API."""
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        response = self.client.chat.completions.create(
            model=model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=model,
            provider=LLMProvider.OPENAI,
            usage=response.usage.model_dump() if response.usage else None,
            metadata={"id": response.id, "created": response.created}
        )
    
    async def generate_async(
        self,
        messages: List[LLMMessage],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using OpenAI API asynchronously."""
        try:
            import openai
        except ImportError:
            raise ImportError("openai package is required for OpenAI backend")
        
        async_client = openai.AsyncOpenAI(
            api_key=self.client.api_key,
            base_url=self.client.base_url
        )
        
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        response = await async_client.chat.completions.create(
            model=model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=model,
            provider=LLMProvider.OPENAI,
            usage=response.usage.model_dump() if response.usage else None,
            metadata={"id": response.id, "created": response.created}
        )
    
    async def stream(
        self,
        messages: List[LLMMessage],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream response tokens from OpenAI."""
        try:
            import openai
        except ImportError:
            raise ImportError("openai package is required for OpenAI backend")
        
        async_client = openai.AsyncOpenAI(
            api_key=self.client.api_key,
            base_url=self.client.base_url
        )
        
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        stream = await async_client.chat.completions.create(
            model=model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicBackend(LLMBackend):
    """Anthropic (Claude) LLM backend."""
    
    def __init__(self, api_key: Optional[str] = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package is required for Anthropic backend")
        
        self.client = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )
    
    def generate(
        self,
        messages: List[LLMMessage],
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Anthropic API."""
        # Convert to Anthropic format
        system_message = None
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        request_kwargs = {
            "model": model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 1024,
            **kwargs
        }
        
        if system_message:
            request_kwargs["system"] = system_message
        
        response = self.client.messages.create(**request_kwargs)
        
        return LLMResponse(
            content=response.content[0].text,
            model=model,
            provider=LLMProvider.ANTHROPIC,
            usage=response.usage.model_dump() if hasattr(response, 'usage') else None,
            metadata={"id": response.id}
        )
    
    async def generate_async(
        self,
        messages: List[LLMMessage],
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Anthropic API asynchronously."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package is required for Anthropic backend")
        
        async_client = anthropic.AsyncAnthropic(
            api_key=self.client.api_key
        )
        
        # Convert to Anthropic format
        system_message = None
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        request_kwargs = {
            "model": model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 1024,
            **kwargs
        }
        
        if system_message:
            request_kwargs["system"] = system_message
        
        response = await async_client.messages.create(**request_kwargs)
        
        return LLMResponse(
            content=response.content[0].text,
            model=model,
            provider=LLMProvider.ANTHROPIC,
            usage=response.usage.model_dump() if hasattr(response, 'usage') else None,
            metadata={"id": response.id}
        )
    
    async def stream(
        self,
        messages: List[LLMMessage],
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream response tokens from Anthropic."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package is required for Anthropic backend")
        
        async_client = anthropic.AsyncAnthropic(
            api_key=self.client.api_key
        )
        
        # Convert to Anthropic format
        system_message = None
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        request_kwargs = {
            "model": model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 1024,
            "stream": True,
            **kwargs
        }
        
        if system_message:
            request_kwargs["system"] = system_message
        
        stream = await async_client.messages.create(**request_kwargs)
        
        async for chunk in stream:
            if chunk.type == "content_block_delta":
                yield chunk.delta.text


class OllamaBackend(LLMBackend):
    """Ollama local LLM backend."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        import requests
        self.session = requests.Session()
    
    def generate(
        self,
        messages: List[LLMMessage],
        model: str = "llama2",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Ollama API."""
        import requests
        
        ollama_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        payload = {
            "model": model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                **({"num_predict": max_tokens} if max_tokens else {}),
                **kwargs
            }
        }
        
        response = self.session.post(
            f"{self.base_url}/api/chat",
            json=payload
        )
        response.raise_for_status()
        
        result = response.json()
        
        return LLMResponse(
            content=result["message"]["content"],
            model=model,
            provider=LLMProvider.OLLAMA,
            metadata=result.get("eval_count", {})
        )
    
    async def generate_async(
        self,
        messages: List[LLMMessage],
        model: str = "llama2",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Ollama API asynchronously."""
        import aiohttp
        
        ollama_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        payload = {
            "model": model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                **({"num_predict": max_tokens} if max_tokens else {}),
                **kwargs
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/chat",
                json=payload
            ) as response:
                response.raise_for_status()
                result = await response.json()
        
        return LLMResponse(
            content=result["message"]["content"],
            model=model,
            provider=LLMProvider.OLLAMA,
            metadata=result.get("eval_count", {})
        )
    
    async def stream(
        self,
        messages: List[LLMMessage],
        model: str = "llama2",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream response tokens from Ollama."""
        import aiohttp
        import json
        
        ollama_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        payload = {
            "model": model,
            "messages": ollama_messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                **({"num_predict": max_tokens} if max_tokens else {}),
                **kwargs
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/chat",
                json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.content:
                    if line:
                        chunk = json.loads(line.decode())
                        if not chunk.get("done", False):
                            content = chunk.get("message", {}).get("content", "")
                            if content:
                                yield content


def create_backend(
    provider: Union[str, LLMProvider],
    **kwargs
) -> LLMBackend:
    """Factory function to create LLM backends."""
    if isinstance(provider, str):
        provider = LLMProvider(provider)
    
    if provider == LLMProvider.OPENAI:
        return OpenAIBackend(**kwargs)
    elif provider == LLMProvider.ANTHROPIC:
        return AnthropicBackend(**kwargs)
    elif provider == LLMProvider.OLLAMA:
        return OllamaBackend(**kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def generate(
    messages: Union[str, List[LLMMessage]],
    provider: Union[str, LLMProvider] = LLMProvider.OPENAI,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    backend: Optional[LLMBackend] = None,
    **kwargs
) -> LLMResponse:
    """
    Unified generate() interface for LLM providers.
    
    Args:
        messages: Either a string prompt or list of LLMMessage objects
        provider: LLM provider to use
        model: Model name
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        backend: Pre-configured backend instance
        **kwargs: Additional provider-specific arguments
    
    Returns:
        LLMResponse object with generated content
    """
    if backend is None:
        backend = create_backend(provider, **kwargs)
    
    # Convert string to message format
    if isinstance(messages, str):
        messages = [LLMMessage(role="user", content=messages)]
    
    return backend.generate(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )