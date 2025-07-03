import httpx
import re
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from app.config import settings
from app.utils.logging import get_logger
from app.utils.security import validate_email_address, SecurityManager

logger = get_logger(__name__)


class MailgunClient:
    """Enhanced client for sending emails via Mailgun API with comprehensive security integration."""
    
    def __init__(self):
        self.api_key = settings.mailgun_api_key
        self.domain = settings.mailgun_domain
        self.base_url = f"https://api.mailgun.net/v3/{self.domain}"
        self.service_email = settings.service_email
        
        # Initialize security manager
        self.security_manager = SecurityManager()
        
        # Rate limiting: track sends per email address
        self._rate_limits = defaultdict(list)
        self._max_emails_per_hour = getattr(settings, 'max_emails_per_hour', 20)
        
    def _validate_email_security(self, email: str) -> bool:
        """Validate email address using enhanced security validation."""
        return validate_email_address(email)
    
    async def _check_rate_limit(self, email: str, request_id: Optional[str] = None) -> bool:
        """Check if email address is within rate limits with security logging."""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        # Clean old entries
        self._rate_limits[email] = [
            timestamp for timestamp in self._rate_limits[email] 
            if timestamp > hour_ago
        ]
        
        # Check limit
        if len(self._rate_limits[email]) >= self._max_emails_per_hour:
            logger.warning("Email rate limit exceeded", 
                          email=email, 
                          count=len(self._rate_limits[email]),
                          request_id=request_id,
                          security_event=True)
            self.security_manager.log_failed_attempt(email, "email_rate_limit_exceeded")
            return False
            
        self._rate_limits[email].append(now)
        return True
    
    async def _send_email_request(self, data: Dict[str, str], request_id: Optional[str] = None) -> Dict[str, Any]:
        """Send email request with retry logic and security tracking."""
        max_retries = 3
        retry_delay = 1
        
        start_time = datetime.now()
        logger.debug("Starting mailgun API request", request_id=request_id, to_email=data.get('to'))
        
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/messages",
                        auth=("api", self.api_key),
                        data=data,
                        timeout=30.0
                    )
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    logger.info("Email sent successfully via Mailgun",
                               to_email=data.get('to'),
                               message_id=result.get('id'),
                               request_id=request_id)
                    
                    return {
                        'success': True,
                        'message_id': result.get('id'),
                        'message': result.get('message', 'Email sent successfully'),
                        'request_id': request_id
                    }
                    
            except httpx.HTTPStatusError as e:
                if e.response.status_code in [429, 503, 504] and attempt < max_retries - 1:
                    logger.warning(f"Retrying email send (attempt {attempt + 1})", 
                                 status_code=e.response.status_code,
                                 request_id=request_id)
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                    continue
                else:
                    error_msg = f"HTTP {e.response.status_code}: {str(e)}"
                    logger.error("HTTP error sending email", 
                               status_code=e.response.status_code,
                               error=str(e),
                               to_email=data.get('to'),
                               request_id=request_id,
                               security_event=True)
                    self.security_manager.log_failed_attempt(data.get('to', 'unknown'), f"mailgun_http_error: {e.response.status_code}")
                    return {
                        'success': False,
                        'error': error_msg,
                        'request_id': request_id
                    }
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Retrying email send due to error (attempt {attempt + 1})", 
                                 error=str(e),
                                 request_id=request_id)
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                    continue
                else:
                    logger.error("Error sending email", 
                               error=str(e),
                               to_email=data.get('to'),
                               request_id=request_id,
                               security_event=True)
                    self.security_manager.log_failed_attempt(data.get('to', 'unknown'), f"mailgun_error: {str(e)}")
                    return {
                        'success': False,
                        'error': str(e),
                        'request_id': request_id
                    }
        
        return {
            'success': False, 
            'error': 'Max retries exceeded',
            'request_id': request_id
        }
    
    async def send_email(self, to_email: str, subject: str, content: str, 
                        from_email: Optional[str] = None, request_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Send a plain text email via Mailgun with comprehensive security checks.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            content: Email content (plain text)
            from_email: Sender email (defaults to service email)
            request_id: Optional request ID for tracking
            
        Returns:
            Dict with success status and message details
        """
        # Generate request ID if not provided
        if not request_id:
            request_id = self.security_manager.generate_request_id()
        
        # Enhanced email validation using security module
        if not self._validate_email_security(to_email):
            logger.error("Email security validation failed", 
                        email=to_email,
                        request_id=request_id,
                        security_event=True)
            self.security_manager.log_failed_attempt(to_email, "email_validation_failed")
            return {
                'success': False, 
                'error': 'Email address failed security validation',
                'request_id': request_id
            }
        
        # Rate limiting with security logging
        if not await self._check_rate_limit(to_email, request_id):
            return {
                'success': False, 
                'error': f'Rate limit exceeded. Max {self._max_emails_per_hour} emails per hour.',
                'request_id': request_id
            }
        
        # Content sanitization
        if settings.content_sanitization_enabled:
            from app.utils.security import sanitize_content
            content = sanitize_content(content)
            subject = sanitize_content(subject)
        
        sender = from_email or self.service_email
        
        data = {
            "from": f"AI Email Assistant <{sender}>",
            "to": to_email,
            "subject": subject,
            "text": content
        }
        
        logger.info("Sending email via Mailgun", 
                   to=to_email, 
                   subject=subject,
                   request_id=request_id)
        return await self._send_email_request(data, request_id)
    
    async def send_ai_response_draft(self, to_email: str, original_subject: str, 
                                   ai_response: str, original_email_preview: str,
                                   confidence_score: float = 0.0) -> Dict[str, Any]:
        """
        Send AI-generated response draft to user for review.
        
        Args:
            to_email: User's email address
            original_subject: Subject of original email
            ai_response: AI-generated response content
            original_email_preview: Preview of original email for context
            confidence_score: AI confidence in style matching (0-1)
            
        Returns:
            Dict with success status and message details
        """
        subject = f"AI Draft Response: {original_subject}"
        
        confidence_text = ""
        if confidence_score > 0:
            confidence_percent = int(confidence_score * 100)
            confidence_text = f"\nStyle Matching Confidence: {confidence_percent}%"
        
        formatted_content = f"""Hi there!

I've generated a response draft for your email. Please review and edit before sending:

{'-' * 50}
DRAFT RESPONSE:

{ai_response}
{'-' * 50}

{confidence_text}

ORIGINAL EMAIL CONTEXT:
Subject: {original_subject}
Preview: {original_email_preview[:200]}{'...' if len(original_email_preview) > 200 else ''}

---
This is an AI-generated draft. Please review, edit as needed, and send manually to the original sender.

Best regards,
Your AI Email Assistant
"""
        
        return await self.send_email(to_email, subject, formatted_content)
    
    async def send_profile_confirmation(self, to_email: str, sample_count: int, 
                                      confidence_score: float) -> Dict[str, Any]:
        """
        Send confirmation that profile training email was processed.
        
        Args:
            to_email: User's email address
            sample_count: Total number of training samples
            confidence_score: Current profile confidence (0-1)
            
        Returns:
            Dict with success status and message details
        """
        subject = "Writing Profile Updated ✓"
        
        confidence_percent = int(confidence_score * 100)
        
        # Provide guidance on profile quality
        if sample_count < 5:
            guidance = "Send 3-5 more emails with 'PROFILE:' in the subject to improve accuracy."
        elif sample_count < 10:
            guidance = "Your profile is getting better! A few more samples will help fine-tune your style."
        else:
            guidance = "Your writing profile is well-established and ready for high-quality responses."
        
        content = f"""Profile Training Update

✓ Successfully added email to your writing profile
✓ Total training samples: {sample_count}
✓ Profile confidence: {confidence_percent}%

{guidance}

To continue training, forward emails you've written with 'PROFILE:' in the subject line.
To get AI responses, simply forward emails normally (without 'PROFILE:').

Best regards,
Your AI Email Assistant"""
        
        return await self.send_email(to_email, subject, content)
    
    async def send_error_notification(self, to_email: str, error_type: str, 
                                    error_details: str, support_contact: Optional[str] = None) -> Dict[str, Any]:
        """
        Send error notification to user when email processing fails.
        
        Args:
            to_email: User's email address
            error_type: Type of error (parsing, analysis, generation, etc.)
            error_details: Human-readable error description
            support_contact: Optional support contact info
            
        Returns:
            Dict with success status and message details
        """
        subject = f"Email Processing Issue - {error_type}"
        
        support_text = ""
        if support_contact:
            support_text = f"\n\nNeed help? Contact us at {support_contact}"
        
        content = f"""Email Processing Error

We encountered an issue processing your email:

Error Type: {error_type}
Details: {error_details}

What you can try:
• Make sure your forwarded email has clear, readable content
• For profile training, ensure the email contains your original writing
• Check that email content is in English
• Try forwarding the email again

We're working to improve our service. This error has been logged for review.{support_text}

Best regards,
Your AI Email Assistant"""
        
        return await self.send_email(to_email, subject, content)
    
    async def send_welcome_email(self, to_email: str) -> Dict[str, Any]:
        """
        Send welcome email to new users explaining how the service works.
        
        Args:
            to_email: New user's email address
            
        Returns:
            Dict with success status and message details
        """
        subject = "Welcome to AI Email Assistant! 🎉"
        
        content = f"""Welcome to AI Email Assistant!

Your personal email writing AI is ready to help you respond in your unique style.

HOW TO GET STARTED:

1. TRAIN YOUR PROFILE (Recommended):
   • Forward 5-10 emails you've written to {self.service_email}
   • Add 'PROFILE:' to the subject line
   • These help the AI learn your writing style

2. GET AI RESPONSES:
   • Forward any email you want to respond to
   • We'll send back a draft written in your style
   • Review, edit, and send as needed

EXAMPLE:
Forward an email with subject: "PROFILE: Great example of my casual tone"
Forward an email normally to get an AI response draft

Your responses will be indistinguishable from your own writing!

Questions? Just reply to this email.

Best regards,
Your AI Email Assistant Team"""
        
        return await self.send_email(to_email, subject, content)