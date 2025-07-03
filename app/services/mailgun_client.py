import httpx
import re
import asyncio
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from app.config import settings
from app.utils.logging import get_logger
from app.utils.security import validate_email_address, SecurityManager

logger = get_logger(__name__)

class EmailType(Enum):
    """Email type enumeration for template management."""
    AI_RESPONSE_DRAFT = "ai_response_draft"
    PROFILE_CONFIRMATION = "profile_confirmation"
    ERROR_NOTIFICATION = "error_notification"
    WELCOME = "welcome"
    GENERIC = "generic"

@dataclass
class EmailTemplate:
    """Email template structure."""
    subject_template: str
    content_template: str
    required_variables: List[str]
    max_length: Optional[int] = None
    include_signature: bool = True

class EnhancedMailgunClient:
    """Enhanced client for sending emails via Mailgun API with comprehensive features."""
    
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
        
        # Email size limits
        self.max_subject_length = 200
        self.max_content_length = 100000  # 100KB
        
        # Delivery tracking
        self._delivery_metrics = {
            'sent': 0,
            'failed': 0,
            'rate_limited': 0,
            'validation_failed': 0
        }
        
        # Initialize email templates
        self._email_templates = self._initialize_templates()
        
        # Email validation patterns
        self._content_validation_patterns = {
            'spam_indicators': [
                r'click here now', r'limited time offer', r'act now',
                r'congratulations.*won', r'urgent.*action.*required'
            ],
            'suspicious_links': [
                r'bit\.ly', r'tinyurl\.com', r'goo\.gl',
                r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'
            ]
        }
    
    def _initialize_templates(self) -> Dict[EmailType, EmailTemplate]:
        """Initialize email templates for different message types."""
        return {
            EmailType.AI_RESPONSE_DRAFT: EmailTemplate(
                subject_template="AI Draft Response: {original_subject}",
                content_template="""Hi there!

I've generated a response draft for your email. Please review and edit before sending:

{separator}
DRAFT RESPONSE:

{ai_response}
{separator}

{confidence_text}

ORIGINAL EMAIL CONTEXT:
Subject: {original_subject}
Preview: {original_email_preview}

---
This is an AI-generated draft. Please review, edit as needed, and send manually to the original sender.

Best regards,
Your AI Email Assistant""",
                required_variables=['original_subject', 'ai_response', 'separator', 'confidence_text', 'original_email_preview']
            ),
            
            EmailType.PROFILE_CONFIRMATION: EmailTemplate(
                subject_template="Writing Profile Updated ✓",
                content_template="""Profile Training Update

✓ Successfully added email to your writing profile
✓ Total training samples: {sample_count}
✓ Profile confidence: {confidence_percent}%

{guidance}

To continue training, forward emails you've written with 'PROFILE:' in the subject line.
To get AI responses, simply forward emails normally (without 'PROFILE:').

Best regards,
Your AI Email Assistant""",
                required_variables=['sample_count', 'confidence_percent', 'guidance']
            ),
            
            EmailType.ERROR_NOTIFICATION: EmailTemplate(
                subject_template="Email Processing Issue - {error_type}",
                content_template="""Email Processing Error

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
Your AI Email Assistant""",
                required_variables=['error_type', 'error_details', 'support_text']
            ),
            
            EmailType.WELCOME: EmailTemplate(
                subject_template="Welcome to AI Email Assistant! 🎉",
                content_template="""Welcome to AI Email Assistant!

Your personal email writing AI is ready to help you respond in your unique style.

HOW TO GET STARTED:

1. TRAIN YOUR PROFILE (Recommended):
   • Forward 5-10 emails you've written to {service_email}
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
Your AI Email Assistant Team""",
                required_variables=['service_email']
            )
        }
    
    def _validate_email_content(self, subject: str, content: str) -> Tuple[bool, Optional[str]]:
        """Validate email content for security and quality."""
        
        # Length validation
        if len(subject) > self.max_subject_length:
            return False, f"Subject too long (max {self.max_subject_length} characters)"
        
        if len(content) > self.max_content_length:
            return False, f"Content too long (max {self.max_content_length} characters)"
        
        # Content validation
        if not subject.strip():
            return False, "Subject cannot be empty"
        
        if not content.strip():
            return False, "Content cannot be empty"
        
        # Spam detection
        content_lower = content.lower()
        for pattern in self._content_validation_patterns['spam_indicators']:
            if re.search(pattern, content_lower):
                logger.warning("Potential spam content detected", pattern=pattern)
                return False, "Content contains suspicious patterns"
        
        # Suspicious link detection
        for pattern in self._content_validation_patterns['suspicious_links']:
            if re.search(pattern, content):
                logger.warning("Suspicious link detected", pattern=pattern)
                return False, "Content contains suspicious links"
        
        return True, None
    
    def _render_template(self, email_type: EmailType, variables: Dict[str, Any]) -> Tuple[str, str]:
        """Render email template with provided variables."""
        
        template = self._email_templates.get(email_type)
        if not template:
            raise ValueError(f"Template not found for email type: {email_type}")
        
        # Check required variables
        missing_vars = [var for var in template.required_variables if var not in variables]
        if missing_vars:
            raise ValueError(f"Missing required variables: {', '.join(missing_vars)}")
        
        # Render subject and content
        try:
            subject = template.subject_template.format(**variables)
            content = template.content_template.format(**variables)
            
            # Apply length limits if specified
            if template.max_length and len(content) > template.max_length:
                content = content[:template.max_length] + "..."
            
            return subject, content
            
        except KeyError as e:
            raise ValueError(f"Template variable error: {str(e)}")
    
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
            self._delivery_metrics['rate_limited'] += 1
            logger.warning("Email rate limit exceeded", 
                          email=email, 
                          count=len(self._rate_limits[email]),
                          request_id=request_id,
                          security_event=True)
            self.security_manager.log_failed_attempt(email, "email_rate_limit_exceeded")
            return False
            
        self._rate_limits[email].append(now)
        return True
    
    async def _send_email_request(self, data: Dict[str, str], request_id: Optional[str] = None, 
                                 email_type: EmailType = EmailType.GENERIC) -> Dict[str, Any]:
        """Send email request with retry logic and comprehensive tracking."""
        max_retries = 3
        retry_delay = 1
        
        start_time = datetime.now()
        logger.debug("Starting mailgun API request", 
                    request_id=request_id, 
                    to_email=data.get('to'),
                    email_type=email_type.value)
        
        for attempt in range(max_retries):
            try:
                # Add tracking tags
                tracking_data = data.copy()
                tracking_data['o:tag'] = f"{email_type.value},attempt_{attempt + 1}"
                tracking_data['o:tracking'] = 'yes'
                tracking_data['o:tracking-clicks'] = 'no'  # Disable click tracking for privacy
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/messages",
                        auth=("api", self.api_key),
                        data=tracking_data,
                        timeout=30.0
                    )
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    # Update metrics
                    self._delivery_metrics['sent'] += 1
                    
                    # Calculate delivery time
                    delivery_time = (datetime.now() - start_time).total_seconds()
                    
                    logger.info("Email sent successfully via Mailgun",
                               to_email=data.get('to'),
                               message_id=result.get('id'),
                               email_type=email_type.value,
                               delivery_time_seconds=delivery_time,
                               request_id=request_id)
                    
                    return {
                        'success': True,
                        'message_id': result.get('id'),
                        'message': result.get('message', 'Email sent successfully'),
                        'delivery_time_seconds': delivery_time,
                        'email_type': email_type.value,
                        'request_id': request_id
                    }
                    
            except httpx.HTTPStatusError as e:
                if e.response.status_code in [429, 503, 504] and attempt < max_retries - 1:
                    logger.warning(f"Retrying email send (attempt {attempt + 1})", 
                                 status_code=e.response.status_code,
                                 email_type=email_type.value,
                                 request_id=request_id)
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                    continue
                else:
                    self._delivery_metrics['failed'] += 1
                    error_msg = f"HTTP {e.response.status_code}: {str(e)}"
                    logger.error("HTTP error sending email", 
                               status_code=e.response.status_code,
                               error=str(e),
                               to_email=data.get('to'),
                               email_type=email_type.value,
                               request_id=request_id,
                               security_event=True)
                    self.security_manager.log_failed_attempt(data.get('to', 'unknown'), f"mailgun_http_error: {e.response.status_code}")
                    return {
                        'success': False,
                        'error': error_msg,
                        'email_type': email_type.value,
                        'request_id': request_id
                    }
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Retrying email send due to error (attempt {attempt + 1})", 
                                 error=str(e),
                                 email_type=email_type.value,
                                 request_id=request_id)
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                    continue
                else:
                    self._delivery_metrics['failed'] += 1
                    logger.error("Error sending email", 
                               error=str(e),
                               to_email=data.get('to'),
                               email_type=email_type.value,
                               request_id=request_id,
                               security_event=True)
                    self.security_manager.log_failed_attempt(data.get('to', 'unknown'), f"mailgun_error: {str(e)}")
                    return {
                        'success': False,
                        'error': str(e),
                        'email_type': email_type.value,
                        'request_id': request_id
                    }
        
        return {
            'success': False, 
            'error': 'Max retries exceeded',
            'email_type': email_type.value,
            'request_id': request_id
        }
    
    async def send_email(self, to_email: str, subject: str, content: str, 
                        from_email: Optional[str] = None, request_id: Optional[str] = None,
                        email_type: EmailType = EmailType.GENERIC) -> Dict[str, Any]:
        """
        Send a plain text email via Mailgun with comprehensive validation and tracking.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            content: Email content (plain text)
            from_email: Sender email (defaults to service email)
            request_id: Optional request ID for tracking
            email_type: Type of email for metrics and tracking
            
        Returns:
            Dict with success status and message details
        """
        # Generate request ID if not provided
        if not request_id:
            request_id = self.security_manager.generate_request_id()
        
        # Enhanced email validation using security module
        if not validate_email_address(to_email):
            self._delivery_metrics['validation_failed'] += 1
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
        
        # Content validation
        content_valid, content_error = self._validate_email_content(subject, content)
        if not content_valid:
            self._delivery_metrics['validation_failed'] += 1
            logger.error("Email content validation failed", 
                        error=content_error,
                        request_id=request_id)
            return {
                'success': False, 
                'error': content_error,
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
                   email_type=email_type.value,
                   request_id=request_id)
        
        return await self._send_email_request(data, request_id, email_type)
    
    async def send_ai_response_draft(self, to_email: str, original_subject: str, 
                                   ai_response: str, original_email_preview: str,
                                   confidence_score: float = 0.0) -> Dict[str, Any]:
        """Send AI-generated response draft using template system."""
        
        # Prepare template variables
        confidence_text = ""
        if confidence_score > 0:
            confidence_percent = int(confidence_score * 100)
            confidence_text = f"\nStyle Matching Confidence: {confidence_percent}%"
        
        # Truncate preview if too long
        preview = original_email_preview[:200]
        if len(original_email_preview) > 200:
            preview += "..."
        
        variables = {
            'original_subject': original_subject,
            'ai_response': ai_response,
            'separator': '-' * 50,
            'confidence_text': confidence_text,
            'original_email_preview': preview
        }
        
        try:
            subject, content = self._render_template(EmailType.AI_RESPONSE_DRAFT, variables)
            return await self.send_email(to_email, subject, content, 
                                       email_type=EmailType.AI_RESPONSE_DRAFT)
        except ValueError as e:
            logger.error("Template rendering failed for AI response draft", error=str(e))
            return {
                'success': False,
                'error': f'Template error: {str(e)}'
            }
    
    async def send_profile_confirmation(self, to_email: str, sample_count: int, 
                                      confidence_score: float) -> Dict[str, Any]:
        """Send profile confirmation using template system."""
        
        confidence_percent = int(confidence_score * 100)
        
        # Generate guidance based on sample count
        if sample_count < 5:
            guidance = "Send 3-5 more emails with 'PROFILE:' in the subject to improve accuracy."
        elif sample_count < 10:
            guidance = "Your profile is getting better! A few more samples will help fine-tune your style."
        else:
            guidance = "Your writing profile is well-established and ready for high-quality responses."
        
        variables = {
            'sample_count': sample_count,
            'confidence_percent': confidence_percent,
            'guidance': guidance
        }
        
        try:
            subject, content = self._render_template(EmailType.PROFILE_CONFIRMATION, variables)
            return await self.send_email(to_email, subject, content,
                                       email_type=EmailType.PROFILE_CONFIRMATION)
        except ValueError as e:
            logger.error("Template rendering failed for profile confirmation", error=str(e))
            return {
                'success': False,
                'error': f'Template error: {str(e)}'
            }
    
    async def send_error_notification(self, to_email: str, error_type: str, 
                                    error_details: str, support_contact: Optional[str] = None) -> Dict[str, Any]:
        """Send error notification using template system."""
        
        support_text = ""
        if support_contact:
            support_text = f"\n\nNeed help? Contact us at {support_contact}"
        
        variables = {
            'error_type': error_type,
            'error_details': error_details,
            'support_text': support_text
        }
        
        try:
            subject, content = self._render_template(EmailType.ERROR_NOTIFICATION, variables)
            return await self.send_email(to_email, subject, content,
                                       email_type=EmailType.ERROR_NOTIFICATION)
        except ValueError as e:
            logger.error("Template rendering failed for error notification", error=str(e))
            return {
                'success': False,
                'error': f'Template error: {str(e)}'
            }
    
    async def send_welcome_email(self, to_email: str) -> Dict[str, Any]:
        """Send welcome email using template system."""
        
        variables = {
            'service_email': self.service_email
        }
        
        try:
            subject, content = self._render_template(EmailType.WELCOME, variables)
            return await self.send_email(to_email, subject, content,
                                       email_type=EmailType.WELCOME)
        except ValueError as e:
            logger.error("Template rendering failed for welcome email", error=str(e))
            return {
                'success': False,
                'error': f'Template error: {str(e)}'
            }
    
    def get_delivery_metrics(self) -> Dict[str, Any]:
        """Get email delivery metrics for monitoring."""
        total_attempts = sum(self._delivery_metrics.values())
        
        return {
            'total_attempts': total_attempts,
            'success_rate': self._delivery_metrics['sent'] / max(1, total_attempts),
            'failure_rate': self._delivery_metrics['failed'] / max(1, total_attempts),
            'rate_limited_rate': self._delivery_metrics['rate_limited'] / max(1, total_attempts),
            'validation_failure_rate': self._delivery_metrics['validation_failed'] / max(1, total_attempts),
            'detailed_metrics': self._delivery_metrics.copy(),
            'rate_limit_status': {
                email: len(timestamps) for email, timestamps in self._rate_limits.items()
            }
        }
    
    def reset_metrics(self) -> None:
        """Reset delivery metrics (useful for testing or periodic resets)."""
        self._delivery_metrics = {
            'sent': 0,
            'failed': 0,
            'rate_limited': 0,
            'validation_failed': 0
        }
        logger.info("Email delivery metrics reset")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Mailgun service."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/domains/{self.domain}",
                    auth=("api", self.api_key),
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    domain_info = response.json()
                    return {
                        'status': 'healthy',
                        'domain_verified': domain_info.get('domain', {}).get('state') == 'active',
                        'api_accessible': True,
                        'response_time_ms': response.elapsed.total_seconds() * 1000
                    }
                else:
                    return {
                        'status': 'unhealthy',
                        'error': f"HTTP {response.status_code}",
                        'api_accessible': False
                    }
                    
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'api_accessible': False
            }

# Backward compatibility alias
MailgunClient = EnhancedMailgunClient