from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import time
import asyncio

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Replaces deprecated @app.on_event decorators.
    """
    # Startup
    logger.info("Starting AI Email Response Service")
    
    try:
        # Validate critical configuration
        if not settings.database_url:
            raise ValueError("DATABASE_URL is required")
        
        if not settings.mailgun_api_key:
            logger.warning("MAILGUN_API_KEY not configured - email sending disabled")
        
        # Check database connection
        if not await check_database_connection():
            raise Exception("Database connection failed")
        
        # Create database tables
        await create_tables()
        logger.info("Database initialization completed")
        
        # Log startup configuration (sanitized)
        logger.info("Application started successfully", 
                   environment=getattr(settings, 'environment', 'development'),
                   debug=settings.debug,
                   log_level=settings.log_level,
                   database_type="postgresql" if "postgresql" in settings.database_url else "sqlite",
                   security_enabled=True,
                   rate_limiting_enabled=settings.api_rate_limit_enabled)
        
        yield  # Application runs here
        
    except Exception as e:
        logger.error("Failed to start application", error=str(e))
        raise
    finally:
        # Shutdown
        logger.info("Shutting down AI Email Response Service")
        
        # Close database connections gracefully
        try:
            await async_engine.dispose()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error("Error during shutdown", error=str(e))


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
    """Enhanced security middleware for request validation, monitoring, and protection."""
    start_time = time.time()
    
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
               user_agent=user_agent[:200])  # Truncate long user agents
    
    try:
        # Security validation for suspicious patterns
        if len(path) > 2000:  # Extremely long paths
            logger.warning("Suspicious long path detected", client_ip=client_ip, path_length=len(path))
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Invalid request path"}
            )
        
        # Rate limiting for API endpoints (skip health/docs endpoints)
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
        
        # Add HSTS header for production HTTPS
        if getattr(settings, 'is_production', lambda: 'production' in getattr(settings, 'environment', '').lower())():
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Add CSP header for enhanced security
        if getattr(settings, 'content_security_policy_enabled', False):
            response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
        
        return response
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
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
    expose_headers=["X-Request-ID", "X-RateLimit-Remaining", "X-RateLimit-Reset"]
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
        "status": "running",
        "environment": getattr(settings, 'environment', 'development'),
        "uptime_seconds": round(uptime_seconds, 2),
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
    """Comprehensive health check endpoint with detailed status."""
    try:
        # Check database connection
        db_healthy = await check_database_connection()
        
        # Calculate uptime
        uptime_seconds = time.time() - app_start_time
        
        health_status = {
            "status": "healthy" if db_healthy else "unhealthy",
            "service": "ai-email-response",
            "version": getattr(settings, 'app_version', '1.0.0'),
            "environment": getattr(settings, 'environment', 'development'),
            "uptime_seconds": round(uptime_seconds, 2),
            "checks": {
                "database": "healthy" if db_healthy else "unhealthy",
                "security": "enabled",
                "rate_limiting": "enabled" if settings.api_rate_limit_enabled else "disabled",
                "logging": "enabled",
                "configuration": "valid"
            },
            "timestamp": time.time()
        }
        
        if not db_healthy:
            logger.error("Health check failed: database unhealthy")
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
    """Enhanced metrics endpoint for monitoring and observability."""
    if not getattr(settings, 'metrics_enabled', True):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics endpoint disabled"
        )
    
    # Calculate uptime
    uptime_seconds = time.time() - app_start_time
    
    # Basic metrics (can be enhanced with Prometheus metrics later)
    metrics_data = {
        "service": "ai-email-response",
        "version": getattr(settings, 'app_version', '1.0.0'),
        "environment": getattr(settings, 'environment', 'development'),
        "uptime_seconds": round(uptime_seconds, 2),
        "database_type": "postgresql" if "postgresql" in settings.database_url else "sqlite",
        "features": {
            "security_enabled": True,
            "rate_limiting_enabled": settings.api_rate_limit_enabled,
            "debug_mode": settings.debug,
            "docs_enabled": getattr(settings, 'enable_docs', True)
        },
        "system": {
            "startup_time": app_start_time,
            "current_time": time.time()
        }
    }
    
    return metrics_data


# Enhanced error handlers with proper logging and response formatting
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


# Development server configuration with enhanced settings
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
        workers=1 if getattr(settings, 'environment', 'development').lower() == 'development' else min(4, 2),  # Scale workers based on environment
        use_colors=True,
        loop="asyncio"
    )