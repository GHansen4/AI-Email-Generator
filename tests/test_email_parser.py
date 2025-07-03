import pytest
from app.services.email_parser import EmailParser, ParsedEmail


class TestEmailParser:
    """Test email parsing functionality."""
    
    def setup_method(self):
        self.parser = EmailParser()
    
    def test_gmail_forward_parsing(self):
        """Test parsing Gmail forwarded emails."""
        gmail_forward = """Here's the email you asked me to forward:

---------- Forwarded message ---------
From: sender@example.com
Date: Mon, Jan 15, 2024 at 2:30 PM
Subject: Test Subject
To: recipient@example.com

This is the original email content that was forwarded.
It should be extracted properly.

Best regards,
Original Sender"""
        
        result = self.parser.parse_email(gmail_forward, "forwarder@example.com", "Fwd: Test Subject")
        
        assert isinstance(result, ParsedEmail)
        assert result.email_type == "response_request"
        assert result.confidence_score > 0
        assert result.original_content is not None
        assert len(result.original_content) > 0
    
    def test_profile_content_extraction(self):
        """Test extracting content for profile analysis."""
        profile_email = """Hi there!

This is how I write emails. I'm usually friendly and casual.
I like to use exclamation points and keep things brief.

Thanks!
Best,
User"""
        
        result = self.parser.parse_email(profile_email, "user@example.com", "PROFILE: My writing style")
        
        assert isinstance(result, ParsedEmail)
        assert result.email_type == "profile"
        assert result.user_content is not None
        assert 'friendly and casual' in result.user_content
        assert 'exclamation points' in result.user_content
    
    def test_regular_email_parsing(self):
        """Test handling of regular emails without forward patterns."""
        regular_email = """This is just a regular email without any forward markers.
It should not be parsed as a forwarded email."""
        
        result = self.parser.parse_email(regular_email, "sender@example.com", "Regular Email")
        
        assert isinstance(result, ParsedEmail)
        assert result.email_type == "response_request"
        assert result.confidence_score >= 0
        assert result.original_content is not None 