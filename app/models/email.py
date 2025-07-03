from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON
from sqlalchemy.sql import func
from app.database import Base


class EmailLog(Base):
    __tablename__ = "email_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)  # Foreign key to users
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Email metadata
    message_id = Column(String(255), unique=True, index=True)  # Mailgun message ID
    from_email = Column(String(255), index=True)
    to_email = Column(String(255))
    subject = Column(String(500))
    
    # Processing info
    email_type = Column(String(20))  # "profile" or "response_request"
    processing_status = Column(String(20), default="received")  # received, processing, completed, failed
    error_message = Column(Text)
    
    # Email content (stored for debugging/reprocessing)
    raw_content = Column(Text)  # Original email content
    parsed_content = Column(Text)  # Extracted original email content
    
    # AI response info
    ai_response = Column(Text)  # Generated response
    prompt_used = Column(Text)  # Prompt sent to OpenAI
    openai_model = Column(String(50))  # Model used (gpt-3.5-turbo, gpt-4, etc.)
    tokens_used = Column(Integer)  # Token consumption
    response_time_ms = Column(Integer)  # Processing time in milliseconds


class ForwardedEmail(Base):
    __tablename__ = "forwarded_emails"
    
    id = Column(Integer, primary_key=True, index=True)
    email_log_id = Column(Integer, index=True)  # Foreign key to email_logs
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Original email details (extracted from forwarded email)
    original_from = Column(String(255))
    original_to = Column(String(255))
    original_subject = Column(String(500))
    original_date = Column(DateTime(timezone=True))
    original_content = Column(Text)
    
    # Forwarding details
    forwarded_by = Column(String(255))  # Who forwarded it
    forward_type = Column(String(20))   # gmail, outlook, apple_mail, other
    
    # Analysis results
    email_category = Column(String(50))  # business, personal, support, etc.
    urgency_level = Column(String(20))   # low, medium, high, urgent
    sentiment = Column(String(20))       # positive, neutral, negative
    key_topics = Column(JSON, default=list)  # Extracted topics/keywords 