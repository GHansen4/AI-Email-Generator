import re
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from bs4 import BeautifulSoup
from app.utils.logging import get_logger

logger = get_logger(__name__)


class EmailParser:
    """Parse forwarded emails to extract original content and metadata."""
    
    def __init__(self):
        # Patterns for different email client forward formats
        self.forward_patterns = {
            'gmail': [
                r'---------- Forwarded message ---------\s*\n',
                r'Begin forwarded message:',
            ],
            'outlook': [
                r'From:\s*.*?\nSent:\s*.*?\nTo:\s*.*?\nSubject:\s*.*?\n',
                r'________________________________\s*\n',
                r'From:.*?Sent:.*?To:.*?Subject:.*?\n',
            ],
            'apple_mail': [
                r'Begin forwarded message:\s*\n',
                r'From:\s*.*?\nSubject:\s*.*?\nDate:\s*.*?\nTo:\s*.*?\n',
            ]
        }
    
    def parse_forwarded_email(self, email_content: str, is_html: bool = False) -> Dict[str, Any]:
        """
        Parse a forwarded email to extract the original email content.
        
        Args:
            email_content: The full forwarded email content
            is_html: Whether the content is HTML format
            
        Returns:
            Dict containing parsed email data
        """
        try:
            if is_html:
                return self._parse_html_forwarded_email(email_content)
            else:
                return self._parse_text_forwarded_email(email_content)
                
        except Exception as e:
            logger.error("Error parsing forwarded email", error=str(e))
            return {
                'original_content': email_content,
                'original_from': None,
                'original_to': None,
                'original_subject': None,
                'original_date': None,
                'forward_type': 'unknown',
                'success': False,
                'error': str(e)
            }
    
    def _parse_html_forwarded_email(self, html_content: str) -> Dict[str, Any]:
        """Parse HTML forwarded email."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Convert to text for pattern matching
        text_content = soup.get_text()
        
        # Try to parse as text first
        text_result = self._parse_text_forwarded_email(text_content)
        
        # If text parsing failed, try HTML-specific parsing
        if not text_result.get('success', False):
            return self._extract_from_html_structure(soup)
        
        return text_result
    
    def _parse_text_forwarded_email(self, text_content: str) -> Dict[str, Any]:
        """Parse plain text forwarded email."""
        
        # Detect forward type and extract content
        forward_type, original_content = self._detect_forward_type_and_extract(text_content)
        
        if not original_content:
            return {
                'original_content': text_content,
                'original_from': None,
                'original_to': None,
                'original_subject': None,
                'original_date': None,
                'forward_type': 'unknown',
                'success': False,
                'error': 'Could not detect forward pattern'
            }
        
        # Extract metadata from the forwarded email headers
        metadata = self._extract_forwarded_metadata(text_content, forward_type)
        
        return {
            'original_content': original_content.strip(),
            'original_from': metadata.get('from'),
            'original_to': metadata.get('to'),
            'original_subject': metadata.get('subject'),
            'original_date': metadata.get('date'),
            'forward_type': forward_type,
            'success': True
        }
    
    def _detect_forward_type_and_extract(self, content: str) -> Tuple[str, Optional[str]]:
        """Detect the email client type and extract original content."""
        
        for client_type, patterns in self.forward_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                if match:
                    # Extract content after the forward pattern
                    original_content = content[match.end():]
                    
                    # Clean up common footer patterns
                    original_content = self._clean_forwarded_content(original_content)
                    
                    return client_type, original_content
        
        return 'unknown', None
    
    def _extract_forwarded_metadata(self, content: str, forward_type: str) -> Dict[str, Optional[str]]:
        """Extract metadata (from, to, subject, date) from forwarded email headers."""
        
        metadata: Dict[str, Optional[str]] = {
            'from': None,
            'to': None,
            'subject': None,
            'date': None
        }
        
        # Common patterns for extracting metadata
        patterns = {
            'from': r'From:\s*([^\n\r]+)',
            'to': r'To:\s*([^\n\r]+)',
            'subject': r'Subject:\s*([^\n\r]+)',
            'date': r'(?:Date|Sent):\s*([^\n\r]+)'
        }
        
        for field, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                metadata[field] = match.group(1).strip()
        
        return metadata
    
    def _clean_forwarded_content(self, content: str) -> str:
        """Clean up forwarded content by removing headers and footers."""
        
        # Remove common email headers from the beginning
        header_patterns = [
            r'^From:.*?\n',
            r'^To:.*?\n',
            r'^Sent:.*?\n',
            r'^Date:.*?\n',
            r'^Subject:.*?\n',
            r'^Cc:.*?\n',
            r'^Bcc:.*?\n',
        ]
        
        for pattern in header_patterns:
            content = re.sub(pattern, '', content, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove multiple consecutive newlines
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content.strip()
    
    def _extract_from_html_structure(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract email content from HTML structure (fallback method)."""
        
        # Look for common HTML email structures
        # This is a basic implementation - could be enhanced based on specific email clients
        
        # Try to find the main content area
        main_content = soup.find('body')
        if main_content:
            text_content = main_content.get_text()
            
            return {
                'original_content': text_content.strip(),
                'original_from': None,
                'original_to': None,
                'original_subject': None,
                'original_date': None,
                'forward_type': 'html',
                'success': True
            }
        
        return {
            'original_content': soup.get_text(),
            'original_from': None,
            'original_to': None,
            'original_subject': None,
            'original_date': None,
            'forward_type': 'html',
            'success': False,
            'error': 'Could not parse HTML structure'
        }
    
    def extract_profile_content(self, email_content: str, subject: str) -> Optional[str]:
        """
        Extract writing style content from profile emails.
        
        Args:
            email_content: Email content
            subject: Email subject
            
        Returns:
            Cleaned content for profile analysis
        """
        try:
            # Remove the "PROFILE:" prefix and any forward headers
            content = email_content
            
            # If it's a forwarded email, try to extract the original
            if any(pattern in content.lower() for patterns in self.forward_patterns.values() for pattern in patterns):
                parsed = self.parse_forwarded_email(content)
                if parsed.get('success'):
                    content = parsed['original_content']
            
            # Clean up the content for profile analysis
            content = self._clean_profile_content(content)
            
            return content if content.strip() else None
            
        except Exception as e:
            logger.error("Error extracting profile content", error=str(e))
            return None
    
    def _clean_profile_content(self, content: str) -> str:
        """Clean content for writing style analysis."""
        
        # Remove email signatures (common patterns)
        signature_patterns = [
            r'\n--\s*\n.*',  # Standard email signature delimiter
            r'\nBest regards,.*',
            r'\nSincerely,.*',
            r'\nThanks,.*',
            r'\n\w+\s+\w+\s*\n[^\n]*@[^\n]*\n.*',  # Name + email pattern
        ]
        
        for pattern in signature_patterns:
            content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove quoted text (lines starting with >)
        lines = content.split('\n')
        cleaned_lines = [line for line in lines if not line.strip().startswith('>')]
        content = '\n'.join(cleaned_lines)
        
        # Remove excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' {2,}', ' ', content)
        
        return content.strip() 