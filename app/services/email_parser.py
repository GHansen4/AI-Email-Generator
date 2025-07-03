import re
import email
from email.message import EmailMessage
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import html2text
import logging
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

@dataclass
class ParsedEmail:
    """Enhanced structured representation of a parsed email."""
    original_sender: Optional[str] = None
    original_recipient: Optional[str] = None
    original_subject: Optional[str] = None
    original_content: Optional[str] = None
    original_date: Optional[datetime] = None
    forward_chain: List[str] = field(default_factory=list)
    user_content: Optional[str] = None
    email_type: str = "unknown"
    confidence_score: float = 0.0
    parsing_method: str = ""
    raw_content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class EnhancedEmailParser:
    """Advanced multi-strategy email parser with improved accuracy."""
    
    def __init__(self):
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = True
        self.h2t.ignore_images = True
        self.h2t.body_width = 0  # Don't wrap lines
        
        # Enhanced forward detection patterns
        self.forward_indicators = {
            'gmail': [
                r'---------- Forwarded message ---------',
                r'Begin forwarded message:',
                r'From:.*?(?=Subject:|Date:|To:)',
                r'On.*?wrote:'
            ],
            'outlook': [
                r'From:.*?Sent:.*?To:.*?Subject:',
                r'________________________________',
                r'From:.*?Date:.*?Subject:.*?To:',
                r'-----Original Message-----'
            ],
            'apple_mail': [
                r'Begin forwarded message:',
                r'From:.*?Subject:.*?Date:.*?To:',
                r'On .* wrote:'
            ],
            'thunderbird': [
                r'-------- Original Message --------',
                r'Subject:.*?Date:.*?From:.*?To:'
            ],
            'generic': [
                r'Subject:\s*.*?\n.*?From:\s*.*?\n',
                r'Date:\s*.*?\n.*?From:\s*.*?\n',
                r'>.*?From:\s*.*?\n.*?To:\s*.*?\n'
            ]
        }
        
        # Enhanced quote detection
        self.quote_patterns = [
            r'^>\s*.*$',  # Standard quotes
            r'^\s*On .+ wrote:.*$',  # Reply headers
            r'^\s*Am .+ schrieb.*$',  # German
            r'^\s*Le .+ a écrit.*$',  # French
            r'^\|\s*.*$',  # Some clients
            r'^From:.*?(?=Subject:|Date:|To:)',  # Header quotes
        ]
        
        # Improved signature detection
        self.signature_patterns = [
            r'--\s*$',
            r'^\s*Best regards?,?\s*$',
            r'^\s*Sincerely,?\s*$',
            r'^\s*Thanks?,?\s*$',
            r'^\s*Cheers,?\s*$',
            r'Sent from my \w+',
            r'Get Outlook for \w+',
            r'Virus-free\.',
            r'This email has been checked for viruses',
            r'^\w+\s+\w+$',  # Two words (likely name)
            r'^\w+\s+\w+\s+\|\s+.*$'  # Name | Title format
        ]
    
    def parse_email(self, raw_email: str, sender: str, subject: str) -> ParsedEmail:
        """Enhanced parsing with multiple strategies and confidence scoring."""
        result = ParsedEmail(raw_content=raw_email)
        result.email_type = self._classify_email_intent(subject)
        
        try:
            # Strategy 1: Structured email library parsing
            structured_result = self._parse_structured_email(raw_email)
            if structured_result and structured_result.confidence_score > 0.7:
                return self._finalize_result(structured_result, sender, subject)
            
            # Strategy 2: Pattern-based parsing with ML-like scoring
            pattern_result = self._parse_with_enhanced_patterns(raw_email, subject)
            if pattern_result and pattern_result.confidence_score > 0.6:
                return self._finalize_result(pattern_result, sender, subject)
            
            # Strategy 3: Content-based heuristic parsing
            heuristic_result = self._parse_with_content_analysis(raw_email, sender, subject)
            return self._finalize_result(heuristic_result, sender, subject)
            
        except Exception as e:
            logger.error(f"Email parsing failed: {e}")
            result.confidence_score = 0.1
            result.parsing_method = "error_fallback"
            return result
    
    def _parse_structured_email(self, raw_email: str) -> Optional[ParsedEmail]:
        """Parse using email library with enhanced header extraction."""
        try:
            msg = email.message_from_string(raw_email)
            result = ParsedEmail()
            result.parsing_method = "structured"
            
            # Extract headers with fallbacks
            result.original_sender = self._extract_header(msg, 'From')
            result.original_subject = self._extract_header(msg, 'Subject')
            result.original_date = self._parse_date(msg.get('Date'))
            
            # Get body with improved multipart handling
            body = self._extract_body_content(msg)
            
            if body:
                # Analyze content structure for better extraction
                content_analysis = self._analyze_content_structure(body)
                result.metadata.update(content_analysis)
                
                if result.original_subject and "PROFILE:" in result.original_subject.upper():
                    result.user_content = self._extract_user_content_advanced(body, content_analysis)
                    result.email_type = "profile"
                else:
                    result.original_content = self._extract_original_content_advanced(body, content_analysis)
                    result.email_type = "response_request"
                
                # Calculate confidence based on extraction quality
                result.confidence_score = self._calculate_extraction_confidence(result, content_analysis)
                return result
            
        except Exception as e:
            logger.warning(f"Structured parsing failed: {e}")
        
        return None
    
    def _parse_with_enhanced_patterns(self, raw_email: str, subject: str) -> Optional[ParsedEmail]:
        """Enhanced pattern-based parsing with better accuracy."""
        try:
            result = ParsedEmail()
            result.parsing_method = "enhanced_patterns"
            
            # Clean and normalize content
            content = self._clean_and_normalize(raw_email)
            
            # Multi-stage pattern detection
            forward_info = self._detect_forward_patterns_advanced(content)
            
            if forward_info:
                client_type, pattern_match, split_point = forward_info
                result.metadata['detected_client'] = client_type
                result.metadata['pattern_match'] = pattern_match
                
                if "PROFILE:" in subject.upper():
                    result.user_content = self._extract_content_by_split(content, split_point, 'before')
                    result.email_type = "profile"
                else:
                    result.original_content = self._extract_content_by_split(content, split_point, 'after')
                    result.email_type = "response_request"
                
                # Enhanced confidence calculation
                result.confidence_score = self._calculate_pattern_confidence(
                    result, forward_info, content
                )
                
                return result
                
        except Exception as e:
            logger.warning(f"Pattern parsing failed: {e}")
        
        return None
    
    def _analyze_content_structure(self, content: str) -> Dict[str, Any]:
        """Analyze email content structure for better parsing decisions."""
        lines = content.split('\n')
        
        analysis = {
            'total_lines': len(lines),
            'empty_lines': sum(1 for line in lines if not line.strip()),
            'header_lines': [],
            'quoted_lines': [],
            'signature_start': None,
            'forward_indicators': [],
            'content_density': 0.0
        }
        
        # Analyze line by line
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Detect header patterns
            if re.match(r'^(From|To|Subject|Date|Sent):\s*', line, re.IGNORECASE):
                analysis['header_lines'].append(i)
            
            # Detect quoted content
            if any(re.match(pattern, line) for pattern in self.quote_patterns):
                analysis['quoted_lines'].append(i)
            
            # Detect signatures
            if not analysis['signature_start']:
                if any(re.match(pattern, line_stripped, re.IGNORECASE) 
                       for pattern in self.signature_patterns):
                    analysis['signature_start'] = i
            
            # Detect forward indicators
            for client, patterns in self.forward_indicators.items():
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE | re.MULTILINE):
                        analysis['forward_indicators'].append({
                            'line': i,
                            'client': client,
                            'pattern': pattern
                        })
        
        # Calculate content density
        content_lines = [line for line in lines if line.strip() and 
                        not any(re.match(p, line) for p in self.quote_patterns)]
        analysis['content_density'] = len(content_lines) / max(1, len(lines))
        
        return analysis
    
    def _extract_user_content_advanced(self, content: str, analysis: Dict[str, Any]) -> str:
        """Advanced user content extraction using structure analysis."""
        lines = content.split('\n')
        
        # Find the best split point for user content
        split_point = len(lines)
        
        # Use forward indicators if available
        if analysis['forward_indicators']:
            earliest_forward = min(fi['line'] for fi in analysis['forward_indicators'])
            split_point = min(split_point, earliest_forward)
        
        # Use header patterns
        if analysis['header_lines']:
            earliest_header = min(analysis['header_lines'])
            if earliest_header > 5:  # Skip headers at the very beginning
                split_point = min(split_point, earliest_header)
        
        # Extract user content
        user_lines = lines[:split_point]
        
        # Clean user content
        cleaned_lines = []
        for line in user_lines:
            line = line.strip()
            # Skip obvious forward markers and empty lines
            if (line and 
                not line.startswith(('Fwd:', 'Re:', '>', 'From:', 'To:', 'Subject:')) and
                not re.match(r'^-+$', line)):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _extract_original_content_advanced(self, content: str, analysis: Dict[str, Any]) -> str:
        """Advanced original content extraction using structure analysis."""
        lines = content.split('\n')
        
        # Find the best starting point for original content
        start_point = 0
        end_point = len(lines)
        
        # Use forward indicators
        if analysis['forward_indicators']:
            latest_forward = max(fi['line'] for fi in analysis['forward_indicators'])
            start_point = latest_forward + 1
        
        # Use signature detection for end point
        if analysis['signature_start'] is not None:
            end_point = analysis['signature_start']
        
        # Extract and clean original content
        original_lines = lines[start_point:end_point]
        
        cleaned_lines = []
        for line in original_lines:
            # Skip quoted lines and headers
            if (line.strip() and 
                not any(re.match(pattern, line) for pattern in self.quote_patterns) and
                not re.match(r'^(From|To|Subject|Date|Sent):\s*', line, re.IGNORECASE)):
                cleaned_lines.append(line.rstrip())
        
        return '\n'.join(cleaned_lines).strip()
    
    def _calculate_extraction_confidence(self, result: ParsedEmail, analysis: Dict[str, Any]) -> float:
        """Calculate confidence score based on extraction quality."""
        confidence = 0.0
        
        # Content length factor
        content = result.user_content or result.original_content or ""
        if len(content) > 50:
            confidence += 0.3
        elif len(content) > 20:
            confidence += 0.2
        
        # Structure indicators
        if analysis['forward_indicators']:
            confidence += 0.3
        
        if analysis['header_lines']:
            confidence += 0.2
        
        # Content density
        if analysis['content_density'] > 0.3:
            confidence += 0.2
        
        return min(1.0, confidence)
    
    def _detect_forward_patterns_advanced(self, content: str) -> Optional[Tuple[str, str, int]]:
        """Advanced forward pattern detection with split point calculation."""
        best_match = None
        best_score = 0
        
        for client, patterns in self.forward_indicators.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE))
                if matches:
                    # Score based on pattern specificity and position
                    for match in matches:
                        score = len(pattern) / 100  # Pattern specificity
                        position_factor = 1.0 - (match.start() / len(content))  # Earlier is better
                        total_score = score * position_factor
                        
                        if total_score > best_score:
                            best_score = total_score
                            split_point = content[:match.start()].count('\n')
                            best_match = (client, pattern, split_point)
        
        return best_match
    
    def _clean_and_normalize(self, content: str) -> str:
        """Enhanced content cleaning and normalization."""
        # Handle different encodings
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='ignore')
        
        # Convert HTML to text if needed
        if '<html' in content.lower() or '<body' in content.lower():
            soup = BeautifulSoup(content, 'html.parser')
            content = soup.get_text()
        
        # Normalize line endings and spacing
        content = re.sub(r'\r\n|\r', '\n', content)
        content = re.sub(r'\n{3,}', '\n\n', content)  # Reduce excessive newlines
        
        return content
    
    def _extract_content_by_split(self, content: str, split_point: int, direction: str) -> str:
        """Extract content before or after split point."""
        lines = content.split('\n')
        
        if direction == 'before':
            return '\n'.join(lines[:split_point]).strip()
        else:
            return '\n'.join(lines[split_point:]).strip()
    
    def _calculate_pattern_confidence(self, result: ParsedEmail, forward_info: Tuple, content: str) -> float:
        """Calculate confidence for pattern-based parsing."""
        confidence = 0.5  # Base confidence
        
        client_type, pattern, split_point = forward_info
        
        # Pattern specificity bonus
        if len(pattern) > 30:
            confidence += 0.2
        
        # Client-specific patterns are more reliable
        if client_type != 'generic':
            confidence += 0.1
        
        # Content quality check
        extracted_content = result.user_content or result.original_content or ""
        if len(extracted_content) > 30:
            confidence += 0.2
        
        return min(1.0, confidence)
    
    def _parse_with_content_analysis(self, raw_email: str, sender: str, subject: str) -> ParsedEmail:
        """Fallback parsing using content analysis heuristics."""
        result = ParsedEmail()
        result.parsing_method = "content_analysis"
        result.original_sender = sender
        result.original_subject = subject
        
        content = self._clean_and_normalize(raw_email)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if paragraphs:
            if "PROFILE:" in subject.upper():
                # For profile emails, take substantial paragraphs with more flexible threshold
                # First try paragraphs > 50 chars, then fall back to > 20 chars if nothing found
                substantial_paragraphs = [p for p in paragraphs if len(p) > 50]
                if not substantial_paragraphs:
                    # Fall back to smaller threshold for shorter content
                    substantial_paragraphs = [p for p in paragraphs if len(p) > 20]
                if not substantial_paragraphs:
                    # Take all non-empty paragraphs as last resort
                    substantial_paragraphs = paragraphs
                
                result.user_content = '\n\n'.join(substantial_paragraphs[:3])  # First 3 paragraphs
                result.email_type = "profile"
            else:
                # For response requests, try to find the original email
                # Look for paragraphs that don't start with common forward markers
                original_paragraphs = []
                for p in paragraphs:
                    if not p.startswith(('From:', 'To:', 'Subject:', 'Date:', '>', 'Fwd:', 'Re:')):
                        original_paragraphs.append(p)
                
                result.original_content = '\n\n'.join(original_paragraphs[:5])  # First 5 paragraphs
                result.email_type = "response_request"
            
            # Calculate confidence based on content quality
            content_length = len(result.user_content or result.original_content or "")
            result.confidence_score = min(0.6, content_length / 200)  # Max 0.6 for heuristic
        else:
            result.confidence_score = 0.1
        
        return result
    
    def _finalize_result(self, result: ParsedEmail, sender: str, subject: str) -> ParsedEmail:
        """Finalize parsing result with sender and subject info."""
        if not result.original_sender:
            result.original_sender = sender
        if not result.original_subject:
            result.original_subject = subject
        
        return result
    
    def _classify_email_intent(self, subject: str) -> str:
        """Enhanced email intent classification."""
        subject_upper = subject.upper()
        
        profile_indicators = ['PROFILE:', 'TRAIN:', 'LEARNING:', 'SAMPLE:']
        if any(indicator in subject_upper for indicator in profile_indicators):
            return "profile"
        
        response_indicators = ['FWD:', 'RE:', 'REPLY:', 'RESPOND:']
        if any(indicator in subject_upper for indicator in response_indicators):
            return "response_request"
        
        return "response_request"  # Default assumption
    
    def _extract_header(self, msg, header: str) -> Optional[str]:
        """Safely extract email header."""
        try:
            return msg.get(header)
        except Exception:
            return None
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse email date with multiple format support."""
        if not date_str:
            return None
        
        try:
            import email.utils
            parsed_date = email.utils.parsedate_tz(date_str)
            if parsed_date:
                return datetime.fromtimestamp(email.utils.mktime_tz(parsed_date))
            else:
                return None
        except Exception:
            return None
    
    def _extract_body_content(self, msg) -> str:
        """Extract body content with improved multipart handling."""
        body = ""
        
        try:
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    
                    if content_type == "text/plain":
                        payload = part.get_payload(decode=True)
                        if isinstance(payload, bytes):
                            body += payload.decode('utf-8', errors='ignore')
                        else:
                            body += str(payload)
                    elif content_type == "text/html" and not body:
                        # Use HTML as fallback if no plain text
                        payload = part.get_payload(decode=True)
                        if isinstance(payload, bytes):
                            html_content = payload.decode('utf-8', errors='ignore')
                        else:
                            html_content = str(payload)
                        body += self.h2t.handle(html_content)
            else:
                payload = msg.get_payload()
                if msg.get_content_type() == "text/html":
                    body = self.h2t.handle(str(payload))
                else:
                    body = str(payload)
        
        except Exception as e:
            logger.warning(f"Body extraction failed: {e}")
        
        return body

# Backward compatibility alias
EmailParser = EnhancedEmailParser