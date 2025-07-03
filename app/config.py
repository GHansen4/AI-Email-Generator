from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
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
    
    model_config = SettingsConfigDict(env_file=".env")


# Initialize settings with proper error handling
try:
    settings = Settings()  # type: ignore  # pydantic-settings loads from env vars
except Exception as e:
    print(f"Error loading configuration: {e}")
    print("Please ensure all required environment variables are set in your .env file:")
    print("- DATABASE_URL")
    print("- MAILGUN_API_KEY") 
    print("- MAILGUN_DOMAIN")
    print("- MAILGUN_WEBHOOK_SIGNING_KEY")
    print("- CLAUDE_API_KEY")
    print("- SECRET_KEY")
    print("- SERVICE_EMAIL")
    sys.exit(1) 