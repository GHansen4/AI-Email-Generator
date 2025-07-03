from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List
import sys


class Settings(BaseSettings):
    # Database
    database_url: str
    
    # Mailgun
    mailgun_api_key: str
    mailgun_domain: str
    mailgun_webhook_signing_key: str
    
    # Claude API
    claude_api_key: str
    
    # Application
    secret_key: str
    service_email: str
    debug: bool = False
    
    # Logging
    log_level: str = "INFO"
    
    # Security Settings
    encryption_key: Optional[str] = None  # For sensitive data encryption
    blocked_domains: Optional[List[str]] = None  # Additional blocked email domains
    max_requests_per_minute: int = 30  # Rate limiting
    max_failed_attempts_per_hour: int = 10  # Abuse prevention
    max_email_size: int = 1024 * 1024  # 1MB email size limit
    webhook_timestamp_tolerance: int = 300  # 5 minutes for webhook freshness
    
    # API Rate Limiting
    api_rate_limit_enabled: bool = True
    api_burst_limit: int = 100  # Burst requests allowed
    
    # Content Security
    content_sanitization_enabled: bool = True
    malicious_content_detection: bool = True
    
    model_config = SettingsConfigDict(env_file=".env")


# Initialize settings with proper error handling
try:
    settings = Settings()  # type: ignore  # pydantic-settings loads from env vars
except Exception as e:
    print(f"Error loading configuration: {e}")
    print("Please ensure all required environment variables are set in your .env file:")
    print("\nREQUIRED:")
    print("- DATABASE_URL")
    print("- MAILGUN_API_KEY") 
    print("- MAILGUN_DOMAIN")
    print("- MAILGUN_WEBHOOK_SIGNING_KEY")
    print("- CLAUDE_API_KEY")
    print("- SECRET_KEY")
    print("- SERVICE_EMAIL")
    print("\nOPTIONAL SECURITY:")
    print("- ENCRYPTION_KEY (for data encryption)")
    print("- BLOCKED_DOMAINS (comma-separated list)")
    print("- MAX_REQUESTS_PER_MINUTE (default: 30)")
    print("- MAX_EMAIL_SIZE (default: 1048576 bytes)")
    sys.exit(1) 