import hmac
import hashlib
from typing import Optional
from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


def verify_mailgun_webhook(timestamp: str, token: str, signature: str) -> bool:
    """
    Verify Mailgun webhook signature for security.
    
    Args:
        timestamp: Timestamp from Mailgun webhook
        token: Token from Mailgun webhook
        signature: Signature from Mailgun webhook
        
    Returns:
        bool: True if signature is valid, False otherwise
    """
    try:
        # Create the signature string
        signature_string = f"{timestamp}{token}"
        
        # Create HMAC-SHA256 signature
        expected_signature = hmac.new(
            key=settings.mailgun_webhook_signing_key.encode(),
            msg=signature_string.encode(),
            digestmod=hashlib.sha256
        ).hexdigest()
        
        # Compare signatures
        is_valid = hmac.compare_digest(signature, expected_signature)
        
        if not is_valid:
            logger.warning(
                "Invalid webhook signature",
                expected=expected_signature,
                received=signature
            )
        
        return is_valid
        
    except Exception as e:
        logger.error("Error verifying webhook signature", error=str(e))
        return False


def validate_email_address(email: str) -> bool:
    """
    Basic email validation.
    
    Args:
        email: Email address to validate
        
    Returns:
        bool: True if email is valid format
    """
    import re
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email)) 