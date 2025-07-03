import structlog
import logging
import sys
from typing import Any, Dict, Optional
from app.config import settings


def add_service_context(logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add service-specific context to all log entries."""
    event_dict["service"] = "email-ai-assistant"
    event_dict["version"] = getattr(settings, 'app_version', '1.0.0')
    event_dict["environment"] = getattr(settings, 'environment', 'development')
    return event_dict


def filter_sensitive_data(logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Filter sensitive information from log entries."""
    sensitive_keys = {'password', 'token', 'api_key', 'secret', 'auth'}
    
    # Check top-level keys
    for key in list(event_dict.keys()):
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            event_dict[key] = "***REDACTED***"
    
    # Redact email content if it's too long (for privacy)
    if 'email_content' in event_dict and len(str(event_dict['email_content'])) > 500:
        content = str(event_dict['email_content'])
        event_dict['email_content'] = content[:200] + "...[TRUNCATED]..." + content[-100:]
    
    # Mask email addresses for privacy (keep domain for debugging)
    for key, value in event_dict.items():
        if isinstance(value, str) and '@' in value and '.' in value:
            # user@domain.com -> u***@domain.com
            parts = value.split('@')
            if len(parts) == 2 and len(parts[0]) > 1:
                masked = parts[0][0] + '*' * (len(parts[0]) - 1) + '@' + parts[1]
                event_dict[key] = masked
    
    return event_dict


def configure_logging():
    """Configure structured logging for the email AI application."""
    
    # Configure stdlib logging
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        stream=sys.stdout  # Explicit stdout for container logging
    )
    
    # Silence noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Build processor chain
    processors = [
        structlog.stdlib.filter_by_level,
        add_service_context,  # Add service metadata
        filter_sensitive_data,  # Security filtering
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Choose renderer based on environment
    if settings.debug or getattr(settings, 'environment', 'development') == 'development':
        # Development: pretty console output
        processors.append(
            structlog.dev.ConsoleRenderer(colors=True)
        )
    else:
        # Production: JSON for log aggregation
        processors.append(
            structlog.processors.JSONRenderer(sort_keys=True)
        )
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__):
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name, typically __name__ from calling module
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


def get_request_logger(request_id: Optional[str] = None, user_email: Optional[str] = None):
    """
    Get a logger with request context pre-bound.
    
    Args:
        request_id: Unique request identifier
        user_email: User email (will be masked automatically)
        
    Returns:
        Logger with bound context
    """
    logger = get_logger()
    
    context = {}
    if request_id:
        context['request_id'] = request_id
    if user_email:
        context['user_email'] = user_email
    
    return logger.bind(**context)


# Convenience functions for common logging patterns
def log_email_processing_start(logger, user_email: str, email_type: str):
    """Log the start of email processing."""
    logger.info("Email processing started",
               user_email=user_email,
               email_type=email_type,
               processing_stage="start")


def log_email_processing_complete(logger, user_email: str, email_type: str, 
                                processing_time_ms: float, success: bool):
    """Log the completion of email processing."""
    logger.info("Email processing completed",
               user_email=user_email,
               email_type=email_type,
               processing_stage="complete",
               processing_time_ms=processing_time_ms,
               success=success)


def log_ai_generation(logger, user_email: str, prompt_length: int, 
                     response_length: int, confidence_score: float):
    """Log AI response generation details."""
    logger.info("AI response generated",
               user_email=user_email,
               prompt_length=prompt_length,
               response_length=response_length,
               confidence_score=confidence_score,
               processing_stage="ai_generation")


def log_profile_update(logger, user_email: str, sample_count: int, 
                      confidence_score: float, analysis_method: str):
    """Log writing profile updates."""
    logger.info("Writing profile updated",
               user_email=user_email,
               sample_count=sample_count,
               confidence_score=confidence_score,
               analysis_method=analysis_method,
               processing_stage="profile_update")


# Performance monitoring helpers
class LoggingTimer:
    """Context manager for timing operations with automatic logging."""
    
    def __init__(self, logger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.debug(f"{self.operation} started", **self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        if self.start_time is None:
            self.logger.error(f"{self.operation} timing error: start_time is None", **self.context)
            return
            
        duration_ms = (time.time() - self.start_time) * 1000
        
        if exc_type is None:
            self.logger.info(f"{self.operation} completed",
                           duration_ms=duration_ms,
                           success=True,
                           **self.context)
        else:
            self.logger.error(f"{self.operation} failed",
                            duration_ms=duration_ms,
                            success=False,
                            error_type=exc_type.__name__,
                            error_message=str(exc_val),
                            **self.context)