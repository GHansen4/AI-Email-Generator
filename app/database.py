from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, DateTime, Text, Float, Integer, Boolean, text, TypeDecorator, CHAR
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
import uuid
from datetime import datetime
import json

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class UUID(TypeDecorator):
    """Platform-independent UUID type.
    
    Uses PostgreSQL's native UUID type when available,
    otherwise stores UUIDs as CHAR(36) strings in other databases.
    """
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PostgreSQLUUID(as_uuid=True))
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return value
        else:
            if not isinstance(value, uuid.UUID):
                value = uuid.UUID(value)
            return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return value
        else:
            if not isinstance(value, uuid.UUID):
                value = uuid.UUID(value)
            return value


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


def get_async_database_url(database_url: str) -> str:
    """Convert database URL to async format."""
    if database_url.startswith("postgresql://"):
        # Convert to asyncpg for PostgreSQL
        return database_url.replace("postgresql://", "postgresql+asyncpg://")
    elif database_url.startswith("sqlite:///"):
        # Convert to aiosqlite for SQLite
        return database_url.replace("sqlite:///", "sqlite+aiosqlite:///")
    else:
        # Return as-is for other formats
        return database_url


def get_database_config() -> dict:
    """Get database configuration based on database type."""
    database_url = settings.database_url
    
    if database_url.startswith("postgresql"):
        # PostgreSQL configuration
        return {
            "pool_size": 10,
            "max_overflow": 20,
            "pool_pre_ping": True,
            "pool_recycle": 3600,  # 1 hour
        }
    elif database_url.startswith("sqlite"):
        # SQLite configuration  
        return {
            "pool_pre_ping": True,
            "connect_args": {"check_same_thread": False},
        }
    else:
        # Default configuration
        return {
            "pool_pre_ping": True,
        }


# Create async engine with proper configuration
async_engine = create_async_engine(
    get_async_database_url(settings.database_url),
    **get_database_config(),
    echo=settings.debug,  # Log SQL queries in debug mode
    future=True,  # Use SQLAlchemy 2.0+ style
)

# Create async session maker
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Async database dependency for FastAPI.
    
    Usage:
        @app.get("/users")
        async def get_users(db: AsyncSession = Depends(get_db)):
            # Use db here
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions.
    
    Usage:
        async with get_db_session() as db:
            user = await db.get(User, user_id)
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_tables():
    """Create all database tables."""
    try:
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error("Failed to create database tables", error=str(e))
        raise


async def drop_tables():
    """Drop all database tables (useful for testing)."""
    try:
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.info("Database tables dropped successfully")
    except Exception as e:
        logger.error("Failed to drop database tables", error=str(e))
        raise


async def check_database_connection():
    """Check if database connection is working."""
    try:
        async with async_engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error("Database connection failed", error=str(e))
        return False


# Database Models for Email AI Service
class User(Base):
    """User model for tracking email users and their profiles."""
    __tablename__ = "users"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # Profile statistics
    sample_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    last_profile_update: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<User(email='{self.email}', samples={self.sample_count})>"


class WritingProfile(Base):
    """Writing profile model storing user's linguistic patterns."""
    __tablename__ = "writing_profiles"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID, nullable=False, index=True)
    
    # Core profile data (encrypted JSON)
    profile_data: Mapped[str] = mapped_column(Text, nullable=False)  # Encrypted JSON
    fingerprint_data: Mapped[str] = mapped_column(Text, nullable=True)  # Encrypted comprehensive fingerprint
    
    # Writing style metrics
    avg_sentence_length: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    avg_paragraph_length: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    
    # Tone and style scores
    formality_score: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)  # 0 = very casual, 1 = very formal
    enthusiasm_score: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)  # 0 = neutral, 1 = very enthusiastic
    
    # Common patterns (JSON strings)
    common_greetings: Mapped[str] = mapped_column(Text, nullable=True)  # JSON list of common greetings
    common_closings: Mapped[str] = mapped_column(Text, nullable=True)   # JSON list of common closings
    common_phrases: Mapped[str] = mapped_column(Text, nullable=True)    # JSON list of frequently used phrases
    vocabulary_level: Mapped[str] = mapped_column(String(20), default="medium", nullable=False)  # simple, medium, advanced
    
    # Email specific patterns
    signature_pattern: Mapped[str] = mapped_column(Text, nullable=True)  # User's typical email signature
    preferred_response_time: Mapped[str] = mapped_column(String(50), nullable=True)  # "immediate", "same_day", "next_day"
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    
    # Sample count for confidence
    sample_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    
    def __repr__(self):
        return f"<WritingProfile(user_id='{self.user_id}', samples={self.sample_count}, confidence={self.confidence_score})>"


class EmailLog(Base):
    """Log of processed emails for monitoring and debugging."""
    __tablename__ = "email_logs"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID, nullable=True, index=True)
    
    # Email details
    sender_email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    subject: Mapped[str] = mapped_column(String(500), nullable=True)
    email_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)  # 'profile', 'response_request'
    
    # Processing details
    processing_status: Mapped[str] = mapped_column(String(50), nullable=False, index=True)  # 'success', 'failed', 'processing'
    confidence_score: Mapped[float] = mapped_column(Float, nullable=True)
    processing_time_ms: Mapped[float] = mapped_column(Float, nullable=True)
    
    # Error tracking
    error_message: Mapped[str] = mapped_column(Text, nullable=True)
    error_type: Mapped[str] = mapped_column(String(100), nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    processed_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<EmailLog(sender='{self.sender_email}', type='{self.email_type}', status='{self.processing_status}')>"


class SecurityEvent(Base):
    """Security events and violations log."""
    __tablename__ = "security_events"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    
    # Event details
    event_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(20), nullable=False, index=True)  # 'low', 'medium', 'high', 'critical'
    
    # Source information
    source_ip: Mapped[str] = mapped_column(String(45), nullable=True, index=True)  # IPv6 support
    source_email: Mapped[str] = mapped_column(String(255), nullable=True, index=True)
    user_agent: Mapped[str] = mapped_column(String(500), nullable=True)
    
    # Event data
    event_data: Mapped[str] = mapped_column(Text, nullable=True)  # JSON data
    description: Mapped[str] = mapped_column(Text, nullable=True)
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<SecurityEvent(type='{self.event_type}', severity='{self.severity}')>"


class ForwardedEmail(Base):
    """Forwarded email model for tracking original emails extracted from forwards."""
    __tablename__ = "forwarded_emails"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    email_log_id: Mapped[uuid.UUID] = mapped_column(UUID, nullable=False, index=True)  # Foreign key to email_logs
    
    # Original email details (extracted from forwarded email)
    original_from: Mapped[str] = mapped_column(String(255), nullable=True)
    original_to: Mapped[str] = mapped_column(String(255), nullable=True)
    original_subject: Mapped[str] = mapped_column(String(500), nullable=True)
    original_date: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    original_content: Mapped[str] = mapped_column(Text, nullable=True)
    
    # Forwarding details
    forwarded_by: Mapped[str] = mapped_column(String(255), nullable=True)  # Who forwarded it
    forward_type: Mapped[str] = mapped_column(String(20), nullable=True)   # gmail, outlook, apple_mail, other
    
    # Analysis results
    email_category: Mapped[str] = mapped_column(String(50), nullable=True)  # business, personal, support, etc.
    urgency_level: Mapped[str] = mapped_column(String(20), nullable=True)   # low, medium, high, urgent
    sentiment: Mapped[str] = mapped_column(String(20), nullable=True)       # positive, neutral, negative
    key_topics: Mapped[str] = mapped_column(Text, nullable=True)  # JSON string of extracted topics/keywords
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<ForwardedEmail(id='{self.id}', original_from='{self.original_from}')>"


# Database utility functions
async def get_user_by_email(db: AsyncSession, email: str) -> User | None:
    """Get user by email address."""
    from sqlalchemy import select
    
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def create_user(db: AsyncSession, email: str) -> User:
    """Create a new user."""
    user = User(email=email)
    db.add(user)
    await db.flush()  # Get the ID without committing
    await db.refresh(user)
    return user


async def get_or_create_user(db: AsyncSession, email: str) -> User:
    """Get existing user or create new one."""
    user = await get_user_by_email(db, email)
    if not user:
        user = await create_user(db, email)
        logger.info("Created new user", user_email=email)
    return user


async def log_email_processing(
    db: AsyncSession,
    sender_email: str,
    subject: str,
    email_type: str,
    status: str,
    confidence_score: Optional[float] = None,
    processing_time_ms: Optional[float] = None,
    error_message: Optional[str] = None,
    error_type: Optional[str] = None
) -> EmailLog:
    """Log email processing event."""
    
    # Get user ID if exists
    user = await get_user_by_email(db, sender_email)
    user_id = user.id if user else None
    
    email_log = EmailLog(
        user_id=user_id,
        sender_email=sender_email,
        subject=subject,
        email_type=email_type,
        processing_status=status,
        confidence_score=confidence_score,
        processing_time_ms=processing_time_ms,
        error_message=error_message,
        error_type=error_type,
        processed_at=datetime.utcnow() if status in ['success', 'failed'] else None
    )
    
    db.add(email_log)
    await db.flush()
    return email_log


async def log_security_event(
    db: AsyncSession,
    event_type: str,
    severity: str,
    source_ip: Optional[str] = None,
    source_email: Optional[str] = None,
    event_data: Optional[dict] = None,
    description: Optional[str] = None
) -> SecurityEvent:
    """Log security event."""
    
    security_event = SecurityEvent(
        event_type=event_type,
        severity=severity,
        source_ip=source_ip,
        source_email=source_email,
        event_data=json.dumps(event_data) if event_data else None,
        description=description
    )
    
    db.add(security_event)
    await db.flush()
    return security_event