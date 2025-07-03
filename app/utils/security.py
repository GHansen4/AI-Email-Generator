import hmac
import hashlib
import re
import time
import base64
import secrets
from typing import Optional, Dict, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from cryptography.fernet import Fernet
from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class SecurityManager:
    """Comprehensive security manager for email AI service."""
    
    def __init__(self):
        # Rate limiting tracking
        self._request_counts = defaultdict(list)
        self._failed_attempts = defaultdict(list)
        
        # Initialize encryption
        self._encryption_key = self._get_or_create_encryption_key()
        self._cipher = Fernet(self._encryption_key)
        
        # Security configuration
        self.max_requests_per_minute = 30
        self.max_failed_attempts_per_hour = 10
        self.webhook_timestamp_tolerance = 300  # 5 minutes
        self.max_email_size = 1024 * 1024  # 1MB
        self.blocked_domains = self._load_blocked_domains()
        
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for sensitive data."""
        if hasattr(settings, 'encryption_key') and getattr(settings, 'encryption_key', None):
            encryption_key = getattr(settings, 'encryption_key')
            return encryption_key.encode() if isinstance(encryption_key, str) else encryption_key
        else:
            # Generate a key (in production, store this securely)
            key = Fernet.generate_key()
            logger.warning("Generated new encryption key - store this securely!")
            return key
    
    def _load_blocked_domains(self) -> Set[str]:
        """Load list of blocked email domains."""
        # Default blocked domains (disposable email services, etc.)
        defaults = {
            '10minutemail.com', 'tempmail.org', 'guerrillamail.com',
            'mailinator.com', 'throwaway.email', 'temp-mail.org'
        }
        
        # Add any custom blocked domains from settings
        custom = getattr(settings, 'blocked_domains', None)
        if custom is None:
            return defaults
        
        # If custom is a list/set, convert to set and union
        if isinstance(custom, (list, tuple)):
            custom = set(custom)
        elif isinstance(custom, str):
            # If it's a comma-separated string, split it
            custom = set(domain.strip() for domain in custom.split(',') if domain.strip())
        elif not isinstance(custom, set):
            # If it's some other type, convert to set
            custom = set(custom) if custom else set()
        
        return defaults.union(custom)

    def verify_mailgun_webhook(self, timestamp: str, token: str, signature: str) -> bool:
        """
        Verify Mailgun webhook signature with enhanced security checks.
        
        Args:
            timestamp: Timestamp from Mailgun webhook
            token: Token from Mailgun webhook  
            signature: Signature from Mailgun webhook
            
        Returns:
            bool: True if signature is valid and recent
        """
        try:
            # Check timestamp freshness (prevent replay attacks)
            try:
                webhook_time = int(timestamp)
                current_time = int(time.time())
                
                if abs(current_time - webhook_time) > self.webhook_timestamp_tolerance:
                    logger.warning("Webhook timestamp too old or future",
                                 webhook_time=webhook_time,
                                 current_time=current_time,
                                 difference=abs(current_time - webhook_time))
                    return False
            except ValueError:
                logger.warning("Invalid webhook timestamp format", timestamp=timestamp)
                return False
            
            # Create the signature string
            signature_string = f"{timestamp}{token}"
            
            # Create HMAC-SHA256 signature
            expected_signature = hmac.new(
                key=settings.mailgun_webhook_signing_key.encode(),
                msg=signature_string.encode(),
                digestmod=hashlib.sha256
            ).hexdigest()
            
            # Compare signatures (timing-attack safe)
            is_valid = hmac.compare_digest(signature, expected_signature)
            
            if not is_valid:
                logger.warning("Invalid webhook signature",
                             expected_prefix=expected_signature[:8] + "...",
                             received_prefix=signature[:8] + "...")
                self._log_security_event("webhook_signature_invalid", {
                    'timestamp': timestamp,
                    'signature_valid': False
                })
            
            return is_valid
            
        except Exception as e:
            logger.error("Error verifying webhook signature", error=str(e))
            self._log_security_event("webhook_verification_error", {'error': str(e)})
            return False

    def validate_email_security(self, email: str, content: str = "") -> Tuple[bool, str]:
        """
        Comprehensive email security validation.
        
        Args:
            email: Email address to validate
            content: Email content to check
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Basic format validation
        if not self._validate_email_format(email):
            return False, "Invalid email format"
        
        # Check email length
        if len(email) > 254:
            return False, "Email address too long"
        
        # Domain validation
        domain = email.split('@')[1].lower()
        
        # Check blocked domains
        if domain in self.blocked_domains:
            logger.warning("Blocked domain attempted", domain=domain, email=email)
            self._log_security_event("blocked_domain_attempt", {
                'domain': domain,
                'email': email
            })
            return False, "Email domain not allowed"
        
        # Check for suspicious patterns
        if self._has_suspicious_patterns(email):
            return False, "Email contains suspicious patterns"
        
        # Content size validation
        if len(content) > self.max_email_size:
            return False, f"Email content too large (max {self.max_email_size} bytes)"
        
        # Content security checks
        if content and self._has_malicious_content(content):
            return False, "Email content contains suspicious elements"
        
        return True, ""

    def check_rate_limits(self, identifier: str, request_type: str = "general") -> Tuple[bool, str]:
        """
        Check rate limits for requests.
        
        Args:
            identifier: IP address, email, or other identifier
            request_type: Type of request for separate limits
            
        Returns:
            Tuple of (is_allowed, error_message)
        """
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        
        # Clean old entries
        key = f"{identifier}:{request_type}"
        self._request_counts[key] = [
            timestamp for timestamp in self._request_counts[key]
            if timestamp > minute_ago
        ]
        
        # Check request rate
        current_requests = len(self._request_counts[key])
        if current_requests >= self.max_requests_per_minute:
            logger.warning("Rate limit exceeded",
                         identifier=identifier,
                         request_type=request_type,
                         current_count=current_requests)
            self._log_security_event("rate_limit_exceeded", {
                'identifier': identifier,
                'request_type': request_type,
                'count': current_requests
            })
            return False, f"Rate limit exceeded. Max {self.max_requests_per_minute} requests per minute."
        
        # Track this request
        self._request_counts[key].append(now)
        return True, ""

    def log_failed_attempt(self, identifier: str, reason: str) -> bool:
        """
        Log failed authentication/validation attempt.
        
        Args:
            identifier: IP, email, or other identifier
            reason: Reason for failure
            
        Returns:
            bool: True if identifier should be temporarily blocked
        """
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        # Clean old failures
        self._failed_attempts[identifier] = [
            timestamp for timestamp in self._failed_attempts[identifier]
            if timestamp > hour_ago
        ]
        
        # Add this failure
        self._failed_attempts[identifier].append(now)
        
        failure_count = len(self._failed_attempts[identifier])
        
        logger.warning("Failed attempt logged",
                      identifier=identifier,
                      reason=reason,
                      failure_count=failure_count)
        
        self._log_security_event("failed_attempt", {
            'identifier': identifier,
            'reason': reason,
            'failure_count': failure_count
        })
        
        # Check if should be blocked
        if failure_count >= self.max_failed_attempts_per_hour:
            logger.error("Identifier temporarily blocked due to repeated failures",
                        identifier=identifier,
                        failure_count=failure_count)
            self._log_security_event("identifier_blocked", {
                'identifier': identifier,
                'failure_count': failure_count
            })
            return True
        
        return False

    def encrypt_sensitive_data(self, data: str) -> str:
        """
        Encrypt sensitive data for storage.
        
        Args:
            data: Sensitive data to encrypt
            
        Returns:
            Encrypted data (base64 encoded)
        """
        try:
            encrypted = self._cipher.encrypt(data.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            logger.error("Error encrypting data", error=str(e))
            raise

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """
        Decrypt sensitive data from storage.
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            
        Returns:
            Decrypted data
        """
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted = self._cipher.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error("Error decrypting data", error=str(e))
            raise

    def sanitize_email_content(self, content: str) -> str:
        """
        Sanitize email content for safe processing.
        
        Args:
            content: Raw email content
            
        Returns:
            Sanitized content
        """
        # Remove potentially dangerous HTML/scripts
        content = re.sub(r'<script.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<iframe.*?</iframe>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'javascript:', '', content, flags=re.IGNORECASE)
        
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        # Limit content length
        if len(content) > self.max_email_size:
            content = content[:self.max_email_size] + "...[TRUNCATED]"
        
        return content

    def generate_request_id(self) -> str:
        """Generate secure random request ID."""
        return secrets.token_urlsafe(16)

    def _validate_email_format(self, email: str) -> bool:
        """Validate email format with comprehensive regex."""
        if not email or '@' not in email:
            return False
        
        # More comprehensive email validation
        pattern = r'^[a-zA-Z0-9.!#$%&\'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$'
        return bool(re.match(pattern, email))

    def _has_suspicious_patterns(self, email: str) -> bool:
        """Check for suspicious email patterns."""
        suspicious_patterns = [
            r'(.)\1{5,}',  # Repeated characters
            r'[0-9]{10,}',  # Long number sequences
            r'test.*test',  # Multiple "test" words
            r'admin.*admin',  # Multiple "admin" words
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, email, re.IGNORECASE):
                return True
        return False

    def _has_malicious_content(self, content: str) -> bool:
        """Check for potentially malicious content patterns."""
        malicious_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'data:text/html',
            r'vbscript:',
            r'onload\s*=',
            r'onerror\s*=',
            r'eval\s*\(',
            r'document\.cookie',
        ]
        
        for pattern in malicious_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                logger.warning("Malicious content pattern detected", pattern=pattern)
                return True
        return False

    def _log_security_event(self, event_type: str, details: Dict):
        """Log security events for monitoring."""
        logger.warning("Security event",
                      event_type=event_type,
                      security_event=True,
                      **details)


# Convenience functions
def verify_mailgun_webhook(timestamp: str, token: str, signature: str) -> bool:
    """Verify Mailgun webhook signature."""
    security_manager = SecurityManager()
    return security_manager.verify_mailgun_webhook(timestamp, token, signature)

def verify_webhook_signature(timestamp: str, token: str, signature: str) -> bool:
    """Verify Mailgun webhook signature."""
    security_manager = SecurityManager()
    return security_manager.verify_mailgun_webhook(timestamp, token, signature)

def validate_email_address(email: str) -> bool:
    """
    Validate email address format and security.
    
    Args:
        email: Email address to validate
        
    Returns:
        bool: True if email is valid and safe
    """
    security_manager = SecurityManager()
    is_valid, _ = security_manager.validate_email_security(email, "")
    return is_valid

def validate_email_request(email: str, content: str = "", identifier: Optional[str] = None) -> Tuple[bool, str]:
    """
    Validate email request with security checks.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    security_manager = SecurityManager()
    
    # Email validation
    email_valid, email_error = security_manager.validate_email_security(email, content)
    if not email_valid:
        return False, email_error
    
    # Rate limiting (if identifier provided)
    if identifier:
        rate_ok, rate_error = security_manager.check_rate_limits(identifier, "email_processing")
        if not rate_ok:
            return False, rate_error
    
    return True, ""


def sanitize_content(content: str) -> str:
    """Sanitize email content for safe processing."""
    security_manager = SecurityManager()
    return security_manager.sanitize_email_content(content)