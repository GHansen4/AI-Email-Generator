from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.config import settings
from app.database import create_tables
from app.api.webhooks import router as webhooks_router
from app.utils.logging import configure_logging, get_logger

# Configure logging
configure_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Email Response Service",
    description="AI-powered email response generation service that learns your writing style",
    version="1.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(webhooks_router, prefix="/api", tags=["webhooks"])


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting AI Email Response Service")
    
    try:
        # Create database tables
        create_tables()
        logger.info("Database tables created successfully")
        
        # Log configuration
        logger.info("Application started", 
                   debug=settings.debug,
                   log_level=settings.log_level)
        
    except Exception as e:
        logger.error("Failed to start application", error=str(e))
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on application shutdown."""
    logger.info("Shutting down AI Email Response Service")


@app.get("/")
async def root():
    """Root endpoint for health checks."""
    return {
        "message": "AI Email Response Service",
        "status": "running",
        "docs": "/docs" if settings.debug else "disabled"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ai-email-response",
        "version": "1.0.0"
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Not found", "detail": "The requested resource was not found"}


@app.exception_handler(500)
async def server_error_handler(request, exc):
    logger.error("Internal server error", error=str(exc))
    return {"error": "Internal server error", "detail": "An unexpected error occurred"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    ) 