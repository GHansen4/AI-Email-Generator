from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import time
import asyncio
import signal
import sys
from typing import Dict, Any, Optional
import psutil

from app.config import settings
from app.database import create_tables, check_database_connection, async_engine
from app.api.webhooks import router as webhooks_router
from app.utils.logging import configure_logging, get_logger, LoggingTimer
from app.utils.security import SecurityManager

# Configure logging
configure_logging()
logger = get_logger(__name__)

# Global security manager
security_manager = SecurityManager()

# Application startup time for uptime calculation
app_start_time = time.time()

# Application state for health monitoring
app_state = {
    'startup_complete': False,
    'shutdown_initiated': False,
    'health_status': 'starting',
    'last_health_check': None,
    'service_metrics': {
        'total_requests': 0,
        'failed_requests': 0,
        'avg_response_time': 0.0,
        'peak_memory_mb': 0.0
    }
}

# Graceful shutdown handler
shutdown_event = asyncio.Event()

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown")
    app_state['shutdown_initiated'] = True
    shutdown_event.set()

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Enhanced lifespan context manager with comprehensive startup/shutdown handling.
    """
    # Startup
    logger.info("Starting AI Email Response Service")
    app_state['health_status'] = 'starting'
    
    try:
        # Validate critical configuration
        startup_checks = await perform_startup_checks()
        
        if not startup_checks['all_passed']:
            failed_checks = [check for check, passed in startup_checks['checks'].items() if not passed]
            raise Exception(f"Startup checks failed: {', '.join(failed_checks)}")
        
        # Initialize services
        await initialize_services()
        
        # Mark startup as complete
        app_state['startup_complete'] = True
        app_state['health_status'] = 'healthy'
        
        # Log startup configuration (sanitized)
        logger.info("Application started successfully", 
                   environment=getattr(settings, 'environment', 'development'),
                   debug=settings.debug,
                   log_level=settings.log_level,
                   database_type="postgresql" if "postgresql" in settings.database_url else "sqlite",
                   security_enabled=True,
                   rate_limiting_enabled=settings.api_rate_limit_enabled,
                   startup_time_seconds=round(time.time() - app_start_time, 2))
        
        # Start background tasks
        background_tasks = await start_background_tasks()
        
        yield  # Application runs here
        
    except Exception as e:
        logger.error("Failed to start application", error=str(e))
        app_state['health_status'] = 'failed'
        raise
    finally:
        # Shutdown
        logger.info("Shutting down AI Email Response Service")
        app_state['health_status'] = 'shutting_down'
        
        # Cancel background tasks
        if 'background_tasks' in locals():
            for task in background_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close database connections gracefully
        try:
            await async_engine.dispose()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error("Error during shutdown", error=str(e))
        
        app_state['health_status'] = 'stopped'
        logger.info("Shutdown complete")


async def perform_startup_checks() -> Dict[str, Any]:
    """Perform comprehensive startup checks."""
    checks = {}
    
    # Database check
    try:
        checks['database'] = await check_database_connection()
        if checks['database']:
            await create_tables()
            logger.info("Database initialization completed")
        else:
            logger.error("Database connection failed")
    except Exception as e:
        logger.error("Database check failed", error=str(e))
        checks['database'] = False
    
    # Configuration validation
    checks['config'] = bool(settings.database_url)
    if not settings.mailgun_api_key:
        logger.warning("MAILGUN_API_KEY not configured - email sending disabled")
        checks['mailgun'] = False
    else:
        checks['mailgun'] = True
    
    # Security manager check
    try:
        security_manager.generate_request_id()
        checks['security'] = True
    except Exception as e:
        logger.error("Security manager check failed", error=str(e))
        checks['security'] = False
    
    # Memory check
    memory_usage = psutil.virtual_memory()
    checks['memory'] = memory_usage.percent < 90  # Fail if >90% memory usage
    if not checks['memory']:
        logger.warning("High memory usage detected", usage_percent=memory_usage.percent)
    
    # Disk space check
    disk_usage = psutil.disk_usage('/')
    checks['disk'] = disk_usage.percent < 90  # Fail if >90% disk usage
    if not checks['disk']:
        logger.warning("Low disk space detected", usage_percent=disk_usage.percent)
    
    return {
        'all_passed': all(checks.values()),
        'checks': checks
    }


async def initialize_services():
    """Initialize application services."""
    # Initialize any additional services here
    # For example: AI model loading, cache warming, etc.
    logger.info("Services initialized successfully")


async def start_background_tasks():
    """Start background monitoring tasks."""
    tasks = []
    
    # Health monitoring task
    tasks.append(asyncio.create_task(health_monitor_task()))
    
    # Metrics collection task
    tasks.append(asyncio.create_task(metrics_collection_task()))
    
    # Cleanup task
    tasks.append(asyncio.create_task(cleanup_task()))
    
    logger.info(f"Started {len(tasks)} background tasks")
    return tasks


async def health_monitor_task():
    """Background task to monitor application health."""
    while not shutdown_event.is_set():
        try:
            # Update health status
            app_state['last_health_check'] = time.time()
            
            # Check memory usage
            memory_usage = psutil.virtual_memory()
            app_state['service_metrics']['peak_memory_mb'] = max(
                app_state['service_metrics']['peak_memory_mb'],
                memory_usage.used / 1024 / 1024
            )
            
            # Update health status based on system resources
            if memory_usage.percent > 90:
                app_state['health_status'] = 'degraded'
                logger.warning("High memory usage detected", usage_percent=memory_usage.percent)
            elif app_state['health_status'] == 'degraded' and memory_usage.percent < 80:
                app_state['health_status'] = 'healthy'
                logger.info("Memory usage normalized", usage_percent=memory_usage.percent)
            
            await asyncio.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            logger.error("Health monitor task error", error=str(e))
            await asyncio.sleep(60)  # Wait longer on error


async def metrics_collection_task():
    """Background task to collect application metrics."""
    while not shutdown_event.is_set():
        try:
            # Collect and log metrics periodically
            uptime = time.time() - app_start_time
            
            logger.info("Periodic metrics collection",
                       uptime_seconds=round(uptime, 2),
                       total_requests=app_state['service_metrics']['total_requests'],
                       failed_requests=app_state['service_metrics']['failed_requests'],
                       avg_response_time=app_state['service_metrics']['avg_response_time'],
                       peak_memory_mb=app_state['service_metrics']['peak_memory_mb'])
            
            await asyncio.sleep(300)  # Collect every 5 minutes
            
        except Exception as e:
            logger.error("Metrics collection task error", error=str(e))
            await asyncio.sleep(300)


async def cleanup_task():
    """Background task for periodic cleanup operations."""
    while not shutdown_event.is_set():
        try:
            # Clean up security manager old data
            try:
                # SecurityManager cleanup would go here when implemented
                pass
            except Exception as e:
                logger.error("Cleanup task error", error=str(e))
            
            logger.debug("Periodic cleanup completed")
            
            await asyncio.sleep(3600)  # Cleanup every hour
            
        except Exception as e:
            logger.error("Cleanup task error", error=str(e))
            await asyncio.sleep(3600)


# Create FastAPI app with enhanced configuration
app = FastAPI(
    title="AI Email Response Service",
    description="AI-powered email response generation service that learns your writing style",
    version=getattr(settings, 'app_version', '1.0.0'),
    docs_url="/docs" if getattr(settings, 'enable_docs', True) else None,
    redoc_url="/redoc" if getattr(settings, 'enable_docs', True) else None,
    openapi_url="/openapi.json" if getattr(settings, 'enable_docs', True) else None,
    lifespan=lifespan
)


# Enhanced security middleware with comprehensive monitoring
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Enhanced security middleware with performance tracking."""
    start_time = time.time()
    
    # Update request counter
    app_state['service_metrics']['total_requests'] += 1
    
    # Generate request ID for tracking
    request_id = security_manager.generate_request_id()
    
    # Get client info safely
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    method = request.method
    path = request.url.path
    
    # Log request with sanitized data
    logger.info("Request received",
               request_id=request_id,
               method=method,
               path=path,
               client_ip=client_ip,
               user_agent=user_agent[:200])
    
    try:
        # Check if application is ready
        if not app_state['startup_complete'] and path not in ['/health', '/']:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"error": "Service starting up, please try again shortly"}
            )
        
        # Check if shutdown is initiated
        if app_state['shutdown_initiated']:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"error": "Service is shutting down"}
            )
        
        # Security validation for suspicious patterns
        if len(path) > 2000:
            logger.warning("Suspicious long path detected", client_ip=client_ip, path_length=len(path))
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Invalid request path"}
            )
        
        # Rate limiting for API endpoints
        protected_paths = ["/api/", "/webhook"]
        if any(path.startswith(p) for p in protected_paths):
            rate_ok, rate_error = security_manager.check_rate_limits(client_ip, "api")
            if not rate_ok:
                logger.warning("Rate limit exceeded", 
                             client_ip=client_ip,
                             path=path,
                             user_agent=user_agent[:100])
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"error": "Rate limit exceeded", "detail": rate_error}
                )
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Update average response time
        current_avg = app_state['service_metrics']['avg_response_time']
        total_requests = app_state['service_metrics']['total_requests']
        app_state['service_metrics']['avg_response_time'] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )
        
        # Track failed requests
        if response.status_code >= 400:
            app_state['service_metrics']['failed_requests'] += 1
        
        # Log response
        logger.info("Request completed",
                   request_id=request_id,
                   status_code=response.status_code,
                   processing_time_ms=round(processing_time, 2))
        
        # Add comprehensive security headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-Service-Status"] = app_state['health_status']
        
        # Add HSTS header for production HTTPS
        if getattr(settings, 'is_production', lambda: 'production' in getattr(settings, 'environment', '').lower())():
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Add CSP header for enhanced security
        if getattr(settings, 'content_security_policy_enabled', False):
            response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
        
        return response
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        app_state['service_metrics']['failed_requests'] += 1
        
        logger.error("Request failed",
                    request_id=request_id,
                    error=str(e),
                    processing_time_ms=round(processing_time, 2),
                    client_ip=client_ip,
                    path=path)
        
        # Return generic error in production
        if getattr(settings, 'environment', 'development').lower() == 'production':
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Internal server error", "request_id": request_id}
            )
        raise


# Add trusted host middleware for production security
if getattr(settings, 'is_production', lambda: False)():
    allowed_hosts = getattr(settings, 'allowed_origins', ["*"])
    if allowed_hosts != ["*"]:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=allowed_hosts
        )

# Add CORS middleware with proper configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=getattr(settings, 'allowed_origins', ["*"]),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-RateLimit-Remaining", "X-RateLimit-Reset", "X-Service-Status"]
)

# Include routers
app.include_router(webhooks_router, prefix="/api/v1", tags=["webhooks"])


@app.get("/")
async def root():
    """Root endpoint with comprehensive service information."""
    uptime_seconds = time.time() - app_start_time
    
    return {
        "service": "AI Email Response Service",
        "version": getattr(settings, 'app_version', '1.0.0'),
        "status": app_state['health_status'],
        "environment": getattr(settings, 'environment', 'development'),
        "uptime_seconds": round(uptime_seconds, 2),
        "startup_complete": app_state['startup_complete'],
        "features": {
            "docs": "/docs" if getattr(settings, 'enable_docs', True) else "disabled",
            "metrics": "/metrics" if getattr(settings, 'metrics_enabled', True) else "disabled",
            "security": "enabled",
            "rate_limiting": settings.api_rate_limit_enabled
        },
        "timestamp": time.time()
    }


@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with comprehensive diagnostics."""
    try:
        # Check database connection
        db_healthy = await check_database_connection()
        
        # Check mailgun service if configured
        mailgun_healthy = True
        if settings.mailgun_api_key:
            try:
                # Import here to avoid circular imports
                from app.services.mailgun_client import MailgunClient
                mailgun_client = MailgunClient()
                if hasattr(mailgun_client, 'health_check'):
                    mailgun_status = await mailgun_client.health_check()
                    mailgun_healthy = mailgun_status.get('status') == 'healthy'
            except Exception as e:
                logger.warning("Mailgun health check failed", error=str(e))
                mailgun_healthy = False
        
        # System resource checks
        memory_usage = psutil.virtual_memory()
        disk_usage = psutil.disk_usage('/')
        
        # Calculate uptime
        uptime_seconds = time.time() - app_start_time
        
        # Determine overall health
        overall_healthy = (
            db_healthy and 
            mailgun_healthy and 
            memory_usage.percent < 90 and 
            disk_usage.percent < 90 and
            app_state['startup_complete'] and
            not app_state['shutdown_initiated']
        )
        
        health_status = {
            "status": "healthy" if overall_healthy else "unhealthy",
            "service": "ai-email-response",
            "version": getattr(settings, 'app_version', '1.0.0'),
            "environment": getattr(settings, 'environment', 'development'),
            "uptime_seconds": round(uptime_seconds, 2),
            "startup_complete": app_state['startup_complete'],
            "checks": {
                "database": "healthy" if db_healthy else "unhealthy",
                "mailgun": "healthy" if mailgun_healthy else "unhealthy",
                "security": "enabled",
                "rate_limiting": "enabled" if settings.api_rate_limit_enabled else "disabled",
                "logging": "enabled",
                "configuration": "valid"
            },
            "system": {
                "memory_usage_percent": round(memory_usage.percent, 1),
                "disk_usage_percent": round(disk_usage.percent, 1),
                "cpu_count": psutil.cpu_count(),
                "last_health_check": app_state['last_health_check']
            },
            "metrics": app_state['service_metrics'].copy(),
            "timestamp": time.time()
        }
        
        if not overall_healthy:
            logger.error("Health check failed", 
                        db_healthy=db_healthy,
                        mailgun_healthy=mailgun_healthy,
                        memory_percent=memory_usage.percent,
                        disk_percent=disk_usage.percent)
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=health_status
            )
        
        return health_status
        
    except Exception as e:
        logger.error("Health check error", error=str(e))
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": "Health check failed",
                "service": "ai-email-response",
                "timestamp": time.time()
            }
        )


@app.get("/metrics")
async def metrics():
    """Enhanced metrics endpoint with detailed performance data."""
    if not getattr(settings, 'metrics_enabled', True):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics endpoint disabled"
        )
    
    # Calculate uptime
    uptime_seconds = time.time() - app_start_time
    
    # System metrics
    memory_usage = psutil.virtual_memory()
    disk_usage = psutil.disk_usage('/')
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Calculate success rate
    total_requests = app_state['service_metrics']['total_requests']
    failed_requests = app_state['service_metrics']['failed_requests']
    success_rate = ((total_requests - failed_requests) / max(1, total_requests)) * 100
    
    metrics_data = {
        "service": "ai-email-response",
        "version": getattr(settings, 'app_version', '1.0.0'),
        "environment": getattr(settings, 'environment', 'development'),
        "uptime_seconds": round(uptime_seconds, 2),
        "database_type": "postgresql" if "postgresql" in settings.database_url else "sqlite",
        
        "performance": {
            "total_requests": total_requests,
            "failed_requests": failed_requests,
            "success_rate_percent": round(success_rate, 2),
            "avg_response_time_ms": round(app_state['service_metrics']['avg_response_time'], 2),
            "requests_per_second": round(total_requests / max(1, uptime_seconds), 2)
        },
        
        "system": {
            "memory_usage_percent": round(memory_usage.percent, 1),
            "memory_used_mb": round(memory_usage.used / 1024 / 1024, 1),
            "memory_total_mb": round(memory_usage.total / 1024 / 1024, 1),
            "peak_memory_mb": round(app_state['service_metrics']['peak_memory_mb'], 1),
            "disk_usage_percent": round(disk_usage.percent, 1),
            "cpu_usage_percent": round(cpu_percent, 1),
            "cpu_count": psutil.cpu_count()
        },
        
        "features": {
            "security_enabled": True,
            "rate_limiting_enabled": settings.api_rate_limit_enabled,
            "debug_mode": settings.debug,
            "docs_enabled": getattr(settings, 'enable_docs', True)
        },
        
        "timestamps": {
            "startup_time": app_start_time,
            "current_time": time.time(),
            "last_health_check": app_state['last_health_check']
        }
    }
    
    return metrics_data


@app.get("/ready")
async def readiness_check():
    """Kubernetes-style readiness probe."""
    if app_state['startup_complete'] and not app_state['shutdown_initiated']:
        return {"status": "ready"}
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not ready"}
        )


@app.get("/live")
async def liveness_check():
    """Kubernetes-style liveness probe."""
    return {"status": "alive", "timestamp": time.time()}


# Enhanced error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors with proper logging and response."""
    client_ip = request.client.host if request.client else "unknown"
    
    logger.warning("404 Not Found", 
                  path=request.url.path,
                  method=request.method,
                  client_ip=client_ip,
                  user_agent=request.headers.get("user-agent", "unknown")[:100])
    
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "Not found",
            "detail": "The requested resource was not found",
            "path": str(request.url.path),
            "timestamp": time.time()
        }
    )


@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    """Handle 500 errors with proper logging and sanitized responses."""
    client_ip = request.client.host if request.client else "unknown"
    
    logger.error("Internal server error",
                error=str(exc),
                path=request.url.path,
                method=request.method,
                client_ip=client_ip)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred" if getattr(settings, 'environment', 'development').lower() == 'production' else str(exc),
            "timestamp": time.time()
        }
    )


@app.exception_handler(429)
async def rate_limit_handler(request: Request, exc):
    """Handle rate limit errors with proper headers."""
    client_ip = request.client.host if request.client else "unknown"
    
    logger.warning("Rate limit exceeded",
                  client_ip=client_ip,
                  path=request.url.path)
    
    response = JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": "Rate limit exceeded",
            "detail": "Too many requests. Please try again later.",
            "retry_after": 60,
            "timestamp": time.time()
        }
    )
    response.headers["Retry-After"] = "60"
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper logging and response formatting."""
    client_ip = request.client.host if request.client else "unknown"
    
    if exc.status_code >= 500:
        logger.error("HTTP error",
                    status_code=exc.status_code,
                    detail=exc.detail,
                    path=request.url.path,
                    client_ip=client_ip)
    else:
        logger.warning("HTTP warning",
                      status_code=exc.status_code,
                      detail=exc.detail,
                      path=request.url.path,
                      client_ip=client_ip)
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )


# Development server configuration
if __name__ == "__main__":
    # Validate environment before starting
    if not settings.database_url:
        logger.error("DATABASE_URL is required")
        exit(1)
    
    # Development server with optimized configuration
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=getattr(settings, 'auto_reload', True) and getattr(settings, 'environment', 'development').lower() == 'development',
        log_level=settings.log_level.lower(),
        access_log=getattr(settings, 'enable_request_logging', True),
        workers=1 if getattr(settings, 'environment', 'development').lower() == 'development' else min(4, 2),
        use_colors=True,
        loop="asyncio"
    )