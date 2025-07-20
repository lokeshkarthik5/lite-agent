"""
Middleware system for lite-agent framework.
Provides logging, observability, and execution pipeline customization.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
import time
import logging
import json
from datetime import datetime
from contextlib import contextmanager

from .state import AgentState
from .agent import AgentResult, AgentConfig


@dataclass
class ExecutionContext:
    """Context for agent execution with middleware support."""
    agent_name: str
    state: AgentState
    args: tuple
    kwargs: Dict[str, Any]
    config: AgentConfig
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Middleware(ABC):
    """Abstract base class for middleware."""
    
    @abstractmethod
    def before_execution(self, context: ExecutionContext) -> Optional[ExecutionContext]:
        """Called before agent execution. Return None to stop execution."""
        pass
    
    @abstractmethod
    def after_execution(self, context: ExecutionContext, result: AgentResult) -> AgentResult:
        """Called after agent execution."""
        pass
    
    @abstractmethod
    def on_error(self, context: ExecutionContext, error: Exception) -> Optional[AgentResult]:
        """Called when execution fails. Return AgentResult to override error."""
        pass


class LoggingMiddleware(Middleware):
    """Middleware for logging agent execution."""
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        log_level: int = logging.INFO,
        log_prompts: bool = True,
        log_responses: bool = True,
        log_tools: bool = True,
        log_timing: bool = True
    ):
        self.logger = logger or logging.getLogger("lite-agent")
        self.log_level = log_level
        self.log_prompts = log_prompts
        self.log_responses = log_responses
        self.log_tools = log_tools
        self.log_timing = log_timing
    
    def before_execution(self, context: ExecutionContext) -> Optional[ExecutionContext]:
        """Log execution start."""
        context.start_time = datetime.now()
        
        self.logger.log(
            self.log_level,
            f"Starting agent execution: {context.agent_name}",
            extra={
                "agent": context.agent_name,
                "args": str(context.args),
                "kwargs": str(context.kwargs),
                "timestamp": context.start_time.isoformat()
            }
        )
        
        return context
    
    def after_execution(self, context: ExecutionContext, result: AgentResult) -> AgentResult:
        """Log execution completion."""
        context.end_time = datetime.now()
        
        log_data = {
            "agent": context.agent_name,
            "success": result.success,
            "timestamp": context.end_time.isoformat()
        }
        
        if self.log_timing and result.execution_time:
            log_data["execution_time"] = result.execution_time
        
        if self.log_responses and result.raw_response:
            log_data["response_length"] = len(result.raw_response)
        
        if self.log_tools and result.tool_results:
            log_data["tools_used"] = len(result.tool_results)
        
        self.logger.log(
            self.log_level,
            f"Completed agent execution: {context.agent_name}",
            extra=log_data
        )
        
        return result
    
    def on_error(self, context: ExecutionContext, error: Exception) -> Optional[AgentResult]:
        """Log execution error."""
        self.logger.error(
            f"Agent execution failed: {context.agent_name}",
            extra={
                "agent": context.agent_name,
                "error": str(error),
                "error_type": type(error).__name__,
                "timestamp": datetime.now().isoformat()
            },
            exc_info=True
        )
        return None


class MetricsMiddleware(Middleware):
    """Middleware for collecting execution metrics."""
    
    def __init__(self):
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_execution_time": 0.0,
            "agent_stats": {},
            "error_counts": {}
        }
    
    def before_execution(self, context: ExecutionContext) -> Optional[ExecutionContext]:
        """Track execution start."""
        self.metrics["total_executions"] += 1
        
        if context.agent_name not in self.metrics["agent_stats"]:
            self.metrics["agent_stats"][context.agent_name] = {
                "executions": 0,
                "successes": 0,
                "failures": 0,
                "total_time": 0.0
            }
        
        self.metrics["agent_stats"][context.agent_name]["executions"] += 1
        return context
    
    def after_execution(self, context: ExecutionContext, result: AgentResult) -> AgentResult:
        """Track execution completion."""
        if result.success:
            self.metrics["successful_executions"] += 1
            self.metrics["agent_stats"][context.agent_name]["successes"] += 1
        else:
            self.metrics["failed_executions"] += 1
            self.metrics["agent_stats"][context.agent_name]["failures"] += 1
        
        if result.execution_time:
            self.metrics["total_execution_time"] += result.execution_time
            self.metrics["agent_stats"][context.agent_name]["total_time"] += result.execution_time
        
        return result
    
    def on_error(self, context: ExecutionContext, error: Exception) -> Optional[AgentResult]:
        """Track execution error."""
        error_type = type(error).__name__
        self.metrics["error_counts"][error_type] = self.metrics["error_counts"].get(error_type, 0) + 1
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        return self.metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_execution_time": 0.0,
            "agent_stats": {},
            "error_counts": {}
        }


class CachingMiddleware(Middleware):
    """Middleware for caching agent responses."""
    
    def __init__(self, cache_size: int = 1000, ttl_seconds: Optional[int] = None):
        self.cache_size = cache_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
    
    def _cache_key(self, context: ExecutionContext) -> str:
        """Generate cache key for execution context."""
        key_data = {
            "agent": context.agent_name,
            "args": context.args,
            "kwargs": context.kwargs,
            "memory": context.state.memory
        }
        return json.dumps(key_data, sort_keys=True, default=str)
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is still valid."""
        if key not in self.cache:
            return False
        
        if self.ttl_seconds is None:
            return True
        
        cache_time = self.access_times.get(key, 0)
        return (time.time() - cache_time) < self.ttl_seconds
    
    def _evict_oldest(self) -> None:
        """Evict oldest cache entry."""
        if not self.cache:
            return
        
        oldest_key = min(self.access_times.keys(), key=self.access_times.get)
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    def before_execution(self, context: ExecutionContext) -> Optional[ExecutionContext]:
        """Check cache before execution."""
        cache_key = self._cache_key(context)
        
        if self._is_cache_valid(cache_key):
            # Cache hit - store result in context for after_execution
            context.metadata["cache_hit"] = True
            context.metadata["cached_result"] = self.cache[cache_key]
            self.access_times[cache_key] = time.time()
        else:
            context.metadata["cache_hit"] = False
        
        return context
    
    def after_execution(self, context: ExecutionContext, result: AgentResult) -> AgentResult:
        """Store result in cache after execution."""
        if context.metadata.get("cache_hit"):
            # Return cached result
            cached_result = context.metadata["cached_result"]
            cached_result.metadata = cached_result.metadata or {}
            cached_result.metadata["cache_hit"] = True
            return cached_result
        
        # Store new result in cache
        cache_key = self._cache_key(context)
        
        # Evict if cache is full
        if len(self.cache) >= self.cache_size:
            self._evict_oldest()
        
        self.cache[cache_key] = result
        self.access_times[cache_key] = time.time()
        
        # Mark as cached for future reference
        result.metadata = result.metadata or {}
        result.metadata["cache_hit"] = False
        
        return result
    
    def on_error(self, context: ExecutionContext, error: Exception) -> Optional[AgentResult]:
        """Don't cache errors."""
        return None
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        self.cache.clear()
        self.access_times.clear()


class RetryMiddleware(Middleware):
    """Middleware for retrying failed agent executions."""
    
    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        retry_on: Optional[List[type]] = None
    ):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.retry_on = retry_on or [Exception]
    
    def before_execution(self, context: ExecutionContext) -> Optional[ExecutionContext]:
        """Initialize retry count."""
        context.metadata["retry_count"] = context.metadata.get("retry_count", 0)
        return context
    
    def after_execution(self, context: ExecutionContext, result: AgentResult) -> AgentResult:
        """No retry needed on success."""
        return result
    
    def on_error(self, context: ExecutionContext, error: Exception) -> Optional[AgentResult]:
        """Handle retry logic on error."""
        retry_count = context.metadata.get("retry_count", 0)
        
        # Check if we should retry this error type
        should_retry = any(isinstance(error, exc_type) for exc_type in self.retry_on)
        
        if should_retry and retry_count < self.max_retries:
            # Wait before retry
            if self.backoff_factor > 0:
                wait_time = self.backoff_factor * (2 ** retry_count)
                time.sleep(wait_time)
            
            context.metadata["retry_count"] = retry_count + 1
            
            # Signal that we want to retry (return None means continue with error)
            context.metadata["should_retry"] = True
            return None
        
        # Max retries reached or shouldn't retry
        return None


class MiddlewareStack:
    """Stack of middleware for agent execution."""
    
    def __init__(self, middlewares: Optional[List[Middleware]] = None):
        self.middlewares = middlewares or []
    
    def add_middleware(self, middleware: Middleware) -> None:
        """Add middleware to the stack."""
        self.middlewares.append(middleware)
    
    def remove_middleware(self, middleware_type: type) -> None:
        """Remove middleware of specific type."""
        self.middlewares = [m for m in self.middlewares if not isinstance(m, middleware_type)]
    
    @contextmanager
    def execute_with_middleware(self, context: ExecutionContext):
        """Execute with middleware pipeline."""
        # Before execution middleware
        for middleware in self.middlewares:
            try:
                result = middleware.before_execution(context)
                if result is None:
                    # Middleware cancelled execution
                    raise RuntimeError(f"Execution cancelled by {middleware.__class__.__name__}")
                context = result
            except Exception as e:
                # Handle middleware error
                for error_middleware in reversed(self.middlewares):
                    try:
                        error_result = error_middleware.on_error(context, e)
                        if error_result:
                            yield error_result
                            return
                    except:
                        pass
                raise e
        
        try:
            yield context
        except Exception as e:
            # After error middleware
            for middleware in reversed(self.middlewares):
                try:
                    error_result = middleware.on_error(context, e)
                    if error_result:
                        # Apply after_execution middleware to error result
                        for after_middleware in reversed(self.middlewares):
                            try:
                                error_result = after_middleware.after_execution(context, error_result)
                            except:
                                pass
                        yield error_result
                        return
                except:
                    pass
            raise e
    
    def apply_after_execution(self, context: ExecutionContext, result: AgentResult) -> AgentResult:
        """Apply after execution middleware."""
        for middleware in reversed(self.middlewares):
            try:
                result = middleware.after_execution(context, result)
            except Exception as e:
                # Log middleware error but don't fail execution
                logging.warning(f"Middleware {middleware.__class__.__name__} failed in after_execution: {e}")
        
        return result


# Global middleware stack
global_middleware_stack = MiddlewareStack()


def add_global_middleware(middleware: Middleware) -> None:
    """Add middleware to the global stack."""
    global_middleware_stack.add_middleware(middleware)


def remove_global_middleware(middleware_type: type) -> None:
    """Remove middleware from the global stack."""
    global_middleware_stack.remove_middleware(middleware_type)


def setup_logging(
    level: int = logging.INFO,
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file: Optional[str] = None
) -> None:
    """Setup logging for lite-agent."""
    logger = logging.getLogger("lite-agent")
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add logging middleware if not already present
    has_logging_middleware = any(
        isinstance(m, LoggingMiddleware) 
        for m in global_middleware_stack.middlewares
    )
    
    if not has_logging_middleware:
        add_global_middleware(LoggingMiddleware())


def setup_metrics() -> MetricsMiddleware:
    """Setup metrics collection."""
    # Remove existing metrics middleware
    remove_global_middleware(MetricsMiddleware)
    
    # Add new metrics middleware
    metrics_middleware = MetricsMiddleware()
    add_global_middleware(metrics_middleware)
    
    return metrics_middleware


def get_metrics() -> Dict[str, Any]:
    """Get metrics from global middleware stack."""
    for middleware in global_middleware_stack.middlewares:
        if isinstance(middleware, MetricsMiddleware):
            return middleware.get_metrics()
    return {}