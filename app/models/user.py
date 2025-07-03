from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, Float
from sqlalchemy.sql import func
from app.database import Base


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Basic profile info
    first_name = Column(String(100))
    last_name = Column(String(100))
    is_active = Column(Integer, default=1)  # 1 for active, 0 for inactive


class WritingProfile(Base):
    __tablename__ = "writing_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)  # Foreign key to users
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Writing style metrics
    avg_sentence_length = Column(Float, default=0.0)
    avg_paragraph_length = Column(Float, default=0.0)
    
    # Tone and style
    formality_score = Column(Float, default=0.5)  # 0 = very casual, 1 = very formal
    enthusiasm_score = Column(Float, default=0.5)  # 0 = neutral, 1 = very enthusiastic
    
    # Common patterns (JSON fields)
    common_greetings = Column(JSON, default=list)  # ["Hi", "Hello", "Hey"]
    common_closings = Column(JSON, default=list)   # ["Best", "Thanks", "Cheers"]
    common_phrases = Column(JSON, default=list)    # Frequently used phrases
    vocabulary_level = Column(String(20), default="medium")  # simple, medium, advanced
    
    # Email specific patterns
    signature_pattern = Column(Text)  # User's typical email signature
    preferred_response_time = Column(String(50))  # "immediate", "same_day", "next_day"
    
    # Sample count for confidence
    sample_count = Column(Integer, default=0)  # Number of emails analyzed
    confidence_score = Column(Float, default=0.0)  # How confident we are in the profile 