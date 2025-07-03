import pytest
from app.services.email_parser import EmailParser


class TestEmailParser:
    """Test email parsing functionality."""
    
    def setup_method(self):
        self.parser = EmailParser()
    
    def test_gmail_forward_parsing(self):
        """Test parsing Gmail forwarded emails."""
        gmail_forward = """
        Here's the email you asked me to forward:

        ---------- Forwarded message ---------
        From: sender@example.com
        Date: Mon, Jan 15, 2024 at 2:30 PM
        Subject: Test Subject
        To: recipient@example.com

        This is the original email content that was forwarded.
        It should be extracted properly.

        Best regards,
        Original Sender
        """
        
        result = self.parser.parse_forwarded_email(gmail_forward)
        
        assert result['success'] is True
        assert result['forward_type'] == 'gmail'
        assert 'original email content' in result['original_content'].lower()
        assert result['original_from'] == 'sender@example.com'
        assert result['original_subject'] == 'Test Subject'
    
    def test_profile_content_extraction(self):
        """Test extracting content for profile analysis."""
        profile_email = """
        Subject: PROFILE: My writing style
        
        Hi there!
        
        This is how I write emails. I'm usually friendly and casual.
        I like to use exclamation points and keep things brief.
        
        Thanks!
        Best,
        User
        """
        
        content = self.parser.extract_profile_content(profile_email, "PROFILE: My writing style")
        
        assert content is not None
        assert 'friendly and casual' in content
        assert 'exclamation points' in content
    
    def test_invalid_forward_pattern(self):
        """Test handling of emails that don't match forward patterns."""
        regular_email = """
        This is just a regular email without any forward markers.
        It should not be parsed as a forwarded email.
        """
        
        result = self.parser.parse_forwarded_email(regular_email)
        
        assert result['success'] is False
        assert result['forward_type'] == 'unknown'
        assert result['original_content'] == regular_email 