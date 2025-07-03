import re
import email
from email.message import EmailMessage
from email.policy import default
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import html2text
import logging
from bs4 import BeautifulSoup
import quopri
import base64

# Try to import optional libraries with graceful fallbacks
try:
    import mailparser
    MAILPARSER_AVAILABLE = True
except ImportError:
    mailparser = None
    MAILPARSER_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ParsedEmail:
    """Structured representation of a parsed email."""
    original_sender: Optional[str] = None
    original_recipient: Optional[str] = None
    original_subject: Optional[str] = None
    original_content: Optional[str] = None
    original_date: Optional[datetime] = None
    forward_chain: List[str] = field(default_factory=list)
    user_content: Optional[str] = None  # For profile emails
    email_type: str = "unknown"  # "profile", "response_request", "unclear"
    confidence_score: float = 0.0
    parsing_method: str = ""
    raw_content: str = ""

class RobustEmailParser:
    """Multi-strategy email parser for handling various forward formats."""
    
    def __init__(self):
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = True
        self.h2t.ignore_images = True
        
        # Comprehensive forward indicators by email client
        self.forward_patterns = {
            'gmail': [
                r'---------- Forwarded message ---------',
                r'Begin forwarded message:',
                r'From:.*?Subject:.*?Date:.*?To:'
            ],
            'outlook': [
                r'From:.*?Sent:.*?To:.*?Subject:',
                r'________________________________',
                r'From:.*?Date:.*?Subject:.*?To:'
            ],
            'apple_mail': [
                r'Begin forwarded message:',
                r'From:.*?Subject:.*?Date:.*?To:',
                r'On .* wrote:'
            ],
            'thunderbird': [
                r'-------- Original Message --------',
                r'Subject:.*?Date:.*?From:.*?To:'
            ]
        }
        
        # Quote patterns for reply detection
        self.quote_patterns = [
            r'^>.*$',  # Standard quote
            r'^\s*On .+ wrote:',  # Gmail/Outlook reply
            r'^\s*Am .+ schrieb',  # German
            r'^\s*Le .+ a écrit',  # French
            r'^\|\s.*$',  # Some clients use |
        ]
        
        # Signature patterns
        self.signature_patterns = [
            r'--\s*$',  # Standard signature delimiter
            r'Best regards?,?\s*$',
            r'Sincerely,?\s*$',
            r'Thanks?,?\s*$',
            r'Sent from my \w+',
            r'Get Outlook for \w+',
            r'^Virus-free\.',
        ]
    
    def parse_email(self, raw_email: str, sender: str, subject: str) -> ParsedEmail:
        """
        Main parsing method that tries multiple strategies.
        
        Args:
            raw_email: Raw email content from webhook
            sender: Sender's email address
            subject: Email subject line
            
        Returns:
            ParsedEmail object with extracted information
        """
        
        result = ParsedEmail(raw_content=raw_email)
        
        # Determine email intent from subject
        result.email_type = self._classify_email_intent(subject)
        
        try:
            # Strategy 1: Try mailparser library (most robust)
            if MAILPARSER_AVAILABLE:
                mailparser_result = self._parse_with_mailparser(raw_email)
                if mailparser_result and mailparser_result.confidence_score > 0.7:
                    return mailparser_result
            
            # Strategy 2: Native email library parsing
            native_result = self._parse_with_email_lib(raw_email)
            if native_result and native_result.confidence_score > 0.6:
                return native_result
            
            # Strategy 3: Pattern-based parsing
            pattern_result = self._parse_with_patterns(raw_email, subject)
            if pattern_result and pattern_result.confidence_score > 0.5:
                return pattern_result
            
            # Strategy 4: Fallback heuristic parsing
            return self._parse_with_heuristics(raw_email, sender, subject)
            
        except Exception as e:
            logger.error(f"Email parsing failed: {e}")
            result.confidence_score = 0.1
            result.parsing_method = "error_fallback"
            return result
    
    def _parse_with_mailparser(self, raw_email: str) -> Optional[ParsedEmail]:
        """Parse using the mailparser library (most accurate for complex emails)."""
        try:
            if not MAILPARSER_AVAILABLE:
                return None
                
            mail = mailparser.parse_from_string(raw_email)  # type: ignore
            
            result = ParsedEmail()
            result.parsing_method = "mailparser"
            
            # Extract basic headers
            result.original_sender = mail.from_[0][1] if mail.from_ else None
            
            # Handle subject which can be str, list, or None
            if isinstance(mail.subject, list):
                result.original_subject = mail.subject[0] if mail.subject else None
            else:
                result.original_subject = str(mail.subject) if mail.subject else None
                
            result.original_date = mail.date
            
            # Get clean text content
            if mail.text_plain:
                content = mail.text_plain[0] if isinstance(mail.text_plain, list) else mail.text_plain
            elif mail.text_html:
                html_content = mail.text_html[0] if isinstance(mail.text_html, list) else mail.text_html
                content = self.h2t.handle(html_content)
            else:
                content = ""
            
            # Try to extract forwarded content vs user content
            if result.original_subject and "PROFILE:" in result.original_subject.upper():
                result.user_content = self._extract_user_content(content)
                result.email_type = "profile"
            else:
                result.original_content = self._extract_original_content(content)
                result.email_type = "response_request"
            
            result.confidence_score = 0.8 if result.original_content or result.user_content else 0.3
            return result
            
        except Exception as e:
            logger.warning(f"Mailparser failed: {e}")
            return None
    
    def _parse_with_email_lib(self, raw_email: str) -> Optional[ParsedEmail]:
        """Parse using Python's built-in email library."""
        try:
            msg = email.message_from_string(raw_email, policy=default)
            
            result = ParsedEmail()
            result.parsing_method = "email_lib"
            
            # Extract headers
            result.original_sender = msg.get('From')
            result.original_subject = msg.get('Subject')
            
            # Get body content
            if msg.is_multipart():
                body = ""
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        if isinstance(payload, bytes):
                            body += payload.decode('utf-8', errors='ignore')
                        else:
                            body += str(payload)
                    elif part.get_content_type() == "text/html":
                        payload = part.get_payload(decode=True)
                        if isinstance(payload, bytes):
                            html_body = payload.decode('utf-8', errors='ignore')
                        else:
                            html_body = str(payload)
                        body += self.h2t.handle(html_body)
            else:
                payload = msg.get_payload()
                if msg.get_content_type() == "text/html":
                    body = self.h2t.handle(str(payload))
                else:
                    body = str(payload)
            
            # Process content based on email type
            if "PROFILE:" in (result.original_subject or "").upper():
                result.user_content = self._extract_user_content(body)
                result.email_type = "profile"
            else:
                result.original_content = self._extract_original_content(body)
                result.email_type = "response_request"
            
            result.confidence_score = 0.7 if result.original_content or result.user_content else 0.2
            return result
            
        except Exception as e:
            logger.warning(f"Email lib parsing failed: {e}")
            return None
    
    def _parse_with_patterns(self, raw_email: str, subject: str) -> Optional[ParsedEmail]:
        """Parse using regex patterns for different email clients."""
        try:
            result = ParsedEmail()
            result.parsing_method = "pattern_based"
            
            # Clean the email content
            content = self._clean_email_content(raw_email)
            
            # Detect forward pattern and extract accordingly
            forward_info = self._detect_forward_pattern(content)
            if forward_info:
                client_type, pattern_match = forward_info
                
                if "PROFILE:" in subject.upper():
                    result.user_content = self._extract_user_content_by_pattern(content, pattern_match)
                    result.email_type = "profile"
                else:
                    result.original_content = self._extract_original_content_by_pattern(content, pattern_match)
                    result.email_type = "response_request"
                
                result.confidence_score = 0.6 if result.original_content or result.user_content else 0.2
            else:
                result.confidence_score = 0.1
            
            return result
            
        except Exception as e:
            logger.warning(f"Pattern parsing failed: {e}")
            return None
    
    def _extract_original_content(self, content: str) -> str:
        """Extract the original email content from a forward."""
        lines = content.split('\n')
        
        # Find the start of the forwarded content
        start_idx = 0
        for i, line in enumerate(lines):
            # Look for forward indicators
            if any(pattern in line for pattern_list in self.forward_patterns.values() 
                   for pattern in pattern_list):
                start_idx = i + 1
                break
            # Look for header patterns
            if re.match(r'From:.*?Subject:.*?Date:', line.replace('\n', ' ')):
                start_idx = i + 1
                break
        
        # Find the end (signature or next forward)
        end_idx = len(lines)
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            if any(re.match(pattern, line) for pattern in self.signature_patterns):
                end_idx = i
                break
        
        # Extract and clean the content
        original_lines = lines[start_idx:end_idx]
        
        # Remove quote markers
        cleaned_lines = []
        for line in original_lines:
            if not any(re.match(pattern, line) for pattern in self.quote_patterns):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _extract_user_content(self, content: str) -> str:
        """Extract user's original writing from a profile email."""
        # For profile emails, we want the user's actual email content
        # This is typically everything before the forward headers
        
        lines = content.split('\n')
        
        # Find where the forward starts
        forward_start = len(lines)
        for i, line in enumerate(lines):
            if any(pattern in line for pattern_list in self.forward_patterns.values() 
                   for pattern in pattern_list):
                forward_start = i
                break
            if re.match(r'From:.*?Subject:.*?Date:', line.replace('\n', ' ')):
                forward_start = i
                break
        
        # Take everything before the forward
        user_lines = lines[:forward_start]
        
        # Remove common forward prefixes from user content
        cleaned_lines = []
        for line in user_lines:
            line = line.strip()
            if line and not line.startswith(('Fwd:', 'Re:', '>')):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _classify_email_intent(self, subject: str) -> str:
        """Classify email intent based on subject line."""
        subject_upper = subject.upper()
        
        if any(prefix in subject_upper for prefix in ['PROFILE:', 'TRAIN:']):
            return "profile"
        elif any(prefix in subject_upper for prefix in ['REPLY:', 'RESPOND:']):
            return "response_request"
        elif subject_upper.startswith('FWD:') or subject_upper.startswith('RE:'):
            return "response_request"
        else:
            return "response_request"  # Default assumption
    
    def _clean_email_content(self, content: str) -> str:
        """Clean raw email content."""
        # Decode quoted-printable if present
        if '=' in content and re.search(r'=[0-9A-F]{2}', content):
            try:
                content = quopri.decodestring(content).decode('utf-8', errors='ignore')
            except:
                pass
        
        # Convert HTML to text if needed
        if '<html' in content.lower() or '<body' in content.lower():
            soup = BeautifulSoup(content, 'html.parser')
            content = soup.get_text()
        
        # Normalize line endings
        content = re.sub(r'\r\n', '\n', content)
        content = re.sub(r'\r', '\n', content)
        
        return content
    
    def _detect_forward_pattern(self, content: str) -> Optional[Tuple[str, str]]:
        """Detect which email client forward pattern is present."""
        for client, patterns in self.forward_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content, re.MULTILINE | re.DOTALL):
                    return (client, pattern)
        return None
    
    def _extract_user_content_by_pattern(self, content: str, pattern_match: str) -> str:
        """Extract user content based on detected pattern."""
        # Split content at the pattern match
        parts = re.split(pattern_match, content, 1)
        if len(parts) > 1:
            return parts[0].strip()
        return content.strip()
    
    def _extract_original_content_by_pattern(self, content: str, pattern_match: str) -> str:
        """Extract original content based on detected pattern."""
        # Split content at the pattern match and take the second part
        parts = re.split(pattern_match, content, 1)
        if len(parts) > 1:
            return parts[1].strip()
        return content.strip()
    
    def _parse_with_heuristics(self, raw_email: str, sender: str, subject: str) -> ParsedEmail:
        """Fallback heuristic parsing when other methods fail."""
        result = ParsedEmail()
        result.parsing_method = "heuristic_fallback"
        result.original_sender = sender
        result.original_subject = subject
        
        # Clean content
        content = self._clean_email_content(raw_email)
        
        # Simple heuristic: take the first substantial paragraph
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if paragraphs:
            if "PROFILE:" in subject.upper():
                result.user_content = paragraphs[0]
                result.email_type = "profile"
            else:
                result.original_content = paragraphs[0]
                result.email_type = "response_request"
            
            result.confidence_score = 0.3
        else:
            result.confidence_score = 0.1
        
        return result

# Additional utility for validation
class EmailParsingValidator:
    """Validate parsing quality and provide feedback."""
    
    def validate_parsing_result(self, result: ParsedEmail) -> Dict[str, Any]:
        """Validate and score parsing quality."""
        validation = {
            'is_valid': False,
            'confidence': result.confidence_score,
            'issues': [],
            'recommendations': []
        }
        
        # Check for essential content
        if result.email_type == "profile" and not result.user_content:
            validation['issues'].append("No user content extracted for profile email")
        elif result.email_type == "response_request" and not result.original_content:
            validation['issues'].append("No original content extracted for response request")
        
        # Check content quality
        if result.user_content and len(result.user_content.split()) < 5:
            validation['issues'].append("Extracted content is very short")
        
        # Overall validation
        validation['is_valid'] = len(validation['issues']) == 0 and result.confidence_score > 0.4
        
        return validation