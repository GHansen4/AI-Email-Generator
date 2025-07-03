import anthropic
from typing import Dict, Any, Optional, List
import json
import re
import time
from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

class AIResponseGenerator:
    """Enhanced AI response generator with improved prompting and quality control."""
    
    def __init__(self):
        """Initialize the AI response generator with Claude client."""
        self.client = anthropic.Anthropic(api_key=settings.claude_api_key) if settings.claude_api_key else None
        self.default_model = "claude-3-sonnet-20240229"
        self.max_tokens = 800
        
        # Quality thresholds
        self.min_response_length = 20
        self.max_response_length = 1000
        
        # Response quality patterns
        self.spam_patterns = [
            r'click here', r'limited time', r'act now', r'free gift',
            r'congratulations', r'you\'ve won', r'claim now'
        ]
        
    def generate_response(self, original_email: str, user_profile: Dict[str, Any], 
                         context: Optional[str] = None) -> Dict[str, Any]:
        """Generate AI response with enhanced quality control and multiple attempts."""
        
        try:
            # Validate inputs
            if not original_email or len(original_email.strip()) < 10:
                return self._error_response("Original email is too short or empty")
            
            if not self.client:
                return self._error_response("Claude API key not configured")
            
            # Determine prompt strategy based on profile quality
            profile_confidence = user_profile.get('confidence_score', 0)
            has_advanced_profile = 'comprehensive_fingerprint' in user_profile
            
            logger.info("Starting AI response generation", 
                       profile_confidence=profile_confidence,
                       has_advanced_profile=has_advanced_profile,
                       email_length=len(original_email))
            
            # Generate response with multiple attempts for quality
            best_response = None
            best_score = 0
            
            max_attempts = 2 if profile_confidence > 0.7 else 1
            
            for attempt in range(max_attempts):
                try:
                    response_result = self._generate_single_response(
                        original_email, user_profile, context, attempt
                    )
                    
                    if response_result['success']:
                        quality_score = self._evaluate_response_quality(
                            response_result['response'], user_profile, original_email
                        )
                        
                        if quality_score > best_score:
                            best_score = quality_score
                            best_response = response_result
                            
                        # If we get a high-quality response, use it
                        if quality_score > 0.8:
                            break
                            
                except Exception as e:
                    logger.warning(f"Response generation attempt {attempt + 1} failed: {e}")
                    continue
            
            if best_response:
                best_response['quality_score'] = best_score
                logger.info("AI response generated successfully", 
                           quality_score=best_score,
                           response_length=len(best_response['response']))
                return best_response
            else:
                return self._error_response("Failed to generate acceptable response after all attempts")
                
        except Exception as e:
            logger.error(f"Error in enhanced response generation: {e}")
            return self._error_response(str(e))
    
    def _generate_single_response(self, original_email: str, user_profile: Dict[str, Any], 
                                 context: Optional[str], attempt: int) -> Dict[str, Any]:
        """Generate a single response attempt."""
        
        # Ensure client is available
        if not self.client:
            return {'success': False, 'error': 'Claude API client not initialized'}
        
        # Choose prompt strategy
        if user_profile.get('comprehensive_fingerprint'):
            prompt_strategy = "advanced"
            system_prompt = self._get_advanced_system_prompt(user_profile)
            user_prompt = self._build_advanced_user_prompt(original_email, user_profile, context, attempt)
        else:
            prompt_strategy = "standard"
            system_prompt = self._get_standard_system_prompt(user_profile)
            user_prompt = self._build_standard_user_prompt(original_email, user_profile, context)
        
        # Adjust temperature based on attempt and profile confidence
        temperature = self._calculate_temperature(user_profile, attempt)
        
        logger.info(f"Generating response (attempt {attempt + 1})", 
                   strategy=prompt_strategy,
                   temperature=temperature,
                   confidence=user_profile.get('confidence_score', 0))
        
        try:
            response = self.client.messages.create(
                model=self.default_model,
                max_tokens=self.max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            # Safely extract response content - handle different anthropic response formats
            ai_response = ""
            try:
                if hasattr(response, 'content') and response.content:
                    # Handle the content list structure
                    content_block = response.content[0] if response.content else None
                    if content_block:
                        # Convert to string to avoid type checking issues
                        ai_response = str(content_block)
                        # If it looks like a text object, try to get the text attribute
                        if hasattr(content_block, 'text') and callable(getattr(content_block, 'text', None)) == False:
                            try:
                                ai_response = str(getattr(content_block, 'text', content_block))
                            except:
                                ai_response = str(content_block)
                else:
                    logger.error("No content in Claude response")
                    return {'success': False, 'error': 'Empty response from Claude API'}
            except (IndexError, AttributeError) as e:
                logger.error(f"Error extracting response content: {e}")
                return {'success': False, 'error': 'Failed to extract response content'}
            
            ai_response = ai_response.strip() if ai_response else ""
            
            if not ai_response:
                return {'success': False, 'error': 'Empty response from Claude API'}
            
            # Post-process response
            ai_response = self._post_process_response(ai_response, user_profile)
            
            # Validate response quality
            if not self._validate_response(ai_response):
                return {'success': False, 'error': 'Response failed validation'}
            
            # Safely extract token usage
            tokens_used = 0
            try:
                if hasattr(response, 'usage') and response.usage:
                    tokens_used = getattr(response.usage, 'input_tokens', 0) + getattr(response.usage, 'output_tokens', 0)
            except AttributeError:
                tokens_used = 0
            
            return {
                'response': ai_response,
                'prompt_strategy': prompt_strategy,
                'model': self.default_model,
                'tokens_used': tokens_used,
                'temperature': temperature,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_advanced_system_prompt(self, user_profile: Dict[str, Any]) -> str:
        """Generate advanced system prompt for high-quality profiles."""
        
        confidence = user_profile.get('confidence_score', 0)
        sample_count = user_profile.get('sample_count', 0)
        fingerprint = user_profile.get('comprehensive_fingerprint', {})
        
        return f"""You are an expert AI assistant that generates email responses perfectly matching a specific person's writing style. You have analyzed {sample_count} writing samples with {confidence:.1f} confidence.

CRITICAL OBJECTIVES:
1. Write responses that are completely indistinguishable from the person's actual writing
2. Match their exact linguistic fingerprint across all dimensions
3. Maintain their characteristic voice, tone, and communication patterns
4. Be helpful and appropriate while preserving perfect authenticity

LINGUISTIC FINGERPRINT REQUIREMENTS:
- Formality level: {fingerprint.get('formality', 0.5):.2f}/1.0
- Enthusiasm: {fingerprint.get('enthusiasm', 0.5):.2f}/1.0  
- Directness: {fingerprint.get('directness', 0.5):.2f}/1.0
- Syntactic complexity: {fingerprint.get('syntactic_complexity', 0.5):.2f}/1.0
- Vocabulary sophistication: {fingerprint.get('word_sophistication', 0.5):.2f}/1.0
- Politeness level: {fingerprint.get('politeness_complexity', 0.5):.2f}/1.0

Every word choice, sentence structure, greeting, and closing must exactly match their documented patterns. The response must sound like they wrote it themselves."""

    def _get_standard_system_prompt(self, user_profile: Dict[str, Any]) -> str:
        """Generate standard system prompt for basic profiles."""
        
        confidence = user_profile.get('confidence_score', 0)
        formality = user_profile.get('formality_score', 0.5)
        enthusiasm = user_profile.get('enthusiasm_score', 0.5)
        
        if confidence > 0.6:
            confidence_desc = "strong understanding"
        elif confidence > 0.3:
            confidence_desc = "moderate understanding"
        else:
            confidence_desc = "basic understanding"
        
        return f"""You are an AI assistant that writes email responses matching a specific person's communication style. You have a {confidence_desc} of their writing patterns.

STYLE REQUIREMENTS:
- Formality: {self._describe_formality(formality)}
- Energy level: {self._describe_enthusiasm(enthusiasm)}
- Match their typical greeting and closing patterns
- Use similar sentence structure and vocabulary complexity
- Maintain their characteristic tone and approach

Write responses that sound natural and authentic to their voice while being helpful and appropriate."""

    def _build_advanced_user_prompt(self, original_email: str, user_profile: Dict[str, Any], 
                                   context: Optional[str], attempt: int) -> str:
        """Build advanced user prompt with comprehensive style instructions."""
        
        style_instructions = "Match the user's typical writing style as closely as possible."
        
        try:
            # Import profile analyzer safely to avoid circular imports
            import importlib
            module = importlib.import_module('app.services.profile_analyzer')
            analyzer_class = getattr(module, 'WritingProfileAnalyzer', None)
            
            if analyzer_class:
                analyzer = analyzer_class()
                # Generate detailed style instructions if method exists
                if hasattr(analyzer, 'generate_indistinguishable_prompt'):
                    result = analyzer.generate_indistinguishable_prompt(user_profile)
                    if result:
                        style_instructions = str(result)
        except Exception as e:
            logger.warning(f"Could not generate advanced style instructions: {e}")
        
        # Add context-specific guidance
        email_analysis = self._analyze_original_email(original_email)
        response_guidance = self._generate_response_guidance(email_analysis, user_profile)
        
        # Variation instructions for multiple attempts
        variation_note = ""
        if attempt > 0:
            variation_note = f"\nVARIATION NOTE: This is attempt {attempt + 1}. Provide a slightly different but equally authentic response while maintaining all style requirements."
        
        context_str = context or "Standard email response"
        
        return f"""{style_instructions}

ORIGINAL EMAIL TO RESPOND TO:
\"\"\"
{original_email}
\"\"\"

RESPONSE CONTEXT: {context_str}

EMAIL ANALYSIS & RESPONSE GUIDANCE:
{response_guidance}

RESPONSE REQUIREMENTS:
1. Address all main points from the original email appropriately
2. Follow ALL linguistic style requirements above with perfect accuracy
3. Use natural, helpful, and contextually appropriate content
4. Include proper greeting and closing in their exact style
5. Maintain appropriate length (50-300 words typically)
6. Sound exactly like they would write it themselves{variation_note}

Write the response now:"""

    def _build_standard_user_prompt(self, original_email: str, user_profile: Dict[str, Any], 
                                   context: Optional[str]) -> str:
        """Build standard user prompt for basic profiles."""
        
        # Extract key style elements
        greetings = user_profile.get('common_greetings', ['Hi', 'Hello'])
        closings = user_profile.get('common_closings', ['Best regards', 'Thanks'])
        
        greeting = greetings[0] if greetings else "Hi"
        closing = closings[0] if closings else "Best regards"
        
        formality = user_profile.get('formality_score', 0.5)
        enthusiasm = user_profile.get('enthusiasm_score', 0.5)
        
        style_guidance = f"""
STYLE GUIDELINES:
- Greeting: Use "{greeting}" or similar
- Closing: Use "{closing}" or similar
- Formality: {self._describe_formality(formality)}
- Energy: {self._describe_enthusiasm(enthusiasm)}
- Keep responses helpful and concise
"""
        
        return f"""Please write an email response that matches this person's writing style.

{style_guidance}

ORIGINAL EMAIL TO RESPOND TO:
\"\"\"
{original_email}
\"\"\"

CONTEXT: {context or "Standard email response"}

Write a response that:
1. Addresses the main points in the original email
2. Matches the specified writing style
3. Is helpful and appropriate
4. Uses natural language that sounds like the person wrote it

Response:"""

    def _calculate_temperature(self, user_profile: Dict[str, Any], attempt: int) -> float:
        """Calculate appropriate temperature based on profile confidence and attempt."""
        base_temperature = 0.7
        
        # Adjust based on profile confidence
        confidence = user_profile.get('confidence_score', 0)
        if confidence > 0.8:
            base_temperature = 0.5  # More deterministic for high-confidence profiles
        elif confidence > 0.5:
            base_temperature = 0.6
        else:
            base_temperature = 0.8  # More creative for low-confidence profiles
        
        # Slightly increase temperature for retry attempts
        temperature_adjustment = attempt * 0.1
        
        return min(1.0, base_temperature + temperature_adjustment)
    
    def _post_process_response(self, response: str, user_profile: Dict[str, Any]) -> str:
        """Post-process the AI response for quality and style consistency."""
        
        # Remove any unwanted prefixes or suffixes
        response = re.sub(r'^(Response:|Reply:|Here\'s the response:?)\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'\s*(Let me know if you need any changes\.?|Is this helpful\?|Hope this helps\.?)$', '', response, flags=re.IGNORECASE)
        
        # Ensure proper spacing
        response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)  # Max 2 newlines
        response = response.strip()
        
        # Basic quality checks
        if not response:
            return "Thank you for your email. I'll get back to you soon."
        
        # Ensure it ends with appropriate punctuation
        if response and not response[-1] in '.!?':
            response += '.'
        
        return response
    
    def _validate_response(self, response: str) -> bool:
        """Validate response quality and appropriateness."""
        
        if not response or len(response.strip()) < self.min_response_length:
            return False
        
        if len(response) > self.max_response_length:
            return False
        
        # Check for spam-like content
        response_lower = response.lower()
        for pattern in self.spam_patterns:
            if re.search(pattern, response_lower):
                logger.warning(f"Response contains spam-like pattern: {pattern}")
                return False
        
        # Check for basic coherence (has at least some words)
        word_count = len(response.split())
        if word_count < 5:
            return False
        
        return True
    
    def _evaluate_response_quality(self, response: str, user_profile: Dict[str, Any], 
                                  original_email: str) -> float:
        """Evaluate the quality of generated response (0-1 score)."""
        
        score = 0.0
        
        # Length appropriateness (0-0.3)
        word_count = len(response.split())
        if 20 <= word_count <= 200:
            score += 0.3
        elif 10 <= word_count <= 300:
            score += 0.2
        else:
            score += 0.1
        
        # Style consistency checks (0-0.4)
        profile_confidence = user_profile.get('confidence_score', 0)
        if profile_confidence > 0.5:
            # Check for style elements if we have a good profile
            greetings = user_profile.get('common_greetings', [])
            closings = user_profile.get('common_closings', [])
            
            response_lower = response.lower()
            
            # Check greeting match
            if any(greeting.lower() in response_lower for greeting in greetings):
                score += 0.1
            
            # Check closing match
            if any(closing.lower() in response_lower for closing in closings):
                score += 0.1
            
            # Formality consistency
            formality = user_profile.get('formality_score', 0.5)
            formal_indicators = ['sincerely', 'regards', 'thank you very much']
            casual_indicators = ['thanks', 'cheers', 'best']
            
            if formality > 0.7 and any(indicator in response_lower for indicator in formal_indicators):
                score += 0.1
            elif formality < 0.3 and any(indicator in response_lower for indicator in casual_indicators):
                score += 0.1
            
            score += 0.1  # Bonus for having good profile data
        else:
            score += 0.2  # Give points for low-confidence profiles
        
        # Content relevance (0-0.3)
        original_words = set(original_email.lower().split())
        response_words = set(response.lower().split())
        
        # Check if response addresses similar topics
        common_words = original_words.intersection(response_words)
        if len(common_words) > 3:
            score += 0.2
        elif len(common_words) > 1:
            score += 0.1
        
        # Check for appropriate response indicators
        response_indicators = ['thank', 'help', 'question', 'happy', 'glad', 'please']
        if any(indicator in response.lower() for indicator in response_indicators):
            score += 0.1
        
        return min(1.0, score)
    
    def _analyze_original_email(self, original_email: str) -> Dict[str, Any]:
        """Analyze the original email to provide response guidance."""
        
        analysis = {
            'tone': 'neutral',
            'urgency': 'normal',
            'requires_action': False,
            'questions_asked': 0,
            'main_topics': []
        }
        
        email_lower = original_email.lower()
        
        # Detect tone
        positive_words = ['great', 'excellent', 'wonderful', 'amazing', 'fantastic', 'love', 'excited']
        negative_words = ['problem', 'issue', 'concern', 'worried', 'unfortunately', 'sorry', 'difficult']
        urgent_words = ['urgent', 'asap', 'immediately', 'quickly', 'rush', 'deadline']
        
        if any(word in email_lower for word in positive_words):
            analysis['tone'] = 'positive'
        elif any(word in email_lower for word in negative_words):
            analysis['tone'] = 'concerned'
        
        # Detect urgency
        if any(word in email_lower for word in urgent_words):
            analysis['urgency'] = 'high'
        
        # Count questions
        analysis['questions_asked'] = email_lower.count('?')
        
        # Detect action requirements
        action_words = ['need', 'require', 'please', 'could you', 'would you', 'can you']
        if any(word in email_lower for word in action_words):
            analysis['requires_action'] = True
        
        return analysis
    
    def _generate_response_guidance(self, email_analysis: Dict[str, Any], 
                                   user_profile: Dict[str, Any]) -> str:
        """Generate specific guidance for responding to this email."""
        
        guidance_parts = []
        
        # Tone guidance
        tone = email_analysis.get('tone', 'neutral')
        if tone == 'positive':
            guidance_parts.append("Match the positive, enthusiastic tone")
        elif tone == 'concerned':
            guidance_parts.append("Acknowledge concerns with empathy and provide reassurance")
        else:
            guidance_parts.append("Maintain a professional, helpful tone")
        
        # Urgency guidance
        urgency = email_analysis.get('urgency', 'normal')
        if urgency == 'high':
            guidance_parts.append("Acknowledge the urgency and provide timely response")
        
        # Question guidance
        questions = email_analysis.get('questions_asked', 0)
        if questions > 0:
            guidance_parts.append(f"Address the {questions} question(s) directly")
        
        # Action guidance
        if email_analysis.get('requires_action', False):
            guidance_parts.append("Clearly indicate next steps or actions you'll take")
        
        return '. '.join(guidance_parts) + '.' if guidance_parts else "Provide a helpful, contextually appropriate response."
    
    def _describe_formality(self, formality_score: float) -> str:
        """Describe formality level in human terms."""
        if formality_score > 0.7:
            return "formal, professional language"
        elif formality_score > 0.4:
            return "moderately formal, business-casual tone"
        else:
            return "casual, conversational style"
    
    def _describe_enthusiasm(self, enthusiasm_score: float) -> str:
        """Describe enthusiasm level in human terms."""
        if enthusiasm_score > 0.7:
            return "high energy, enthusiastic expression"
        elif enthusiasm_score > 0.4:
            return "moderate energy, positive tone"
        else:
            return "calm, measured expression"
    
    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate standardized error response."""
        return {
            'success': False,
            'error': error_message,
            'response': None,
            'quality_score': 0.0
        }
    
    def get_fallback_response(self, original_email: Optional[str] = None) -> str:
        """Get a generic fallback response when AI generation fails."""
        return """Thank you for your email. I've received your message and will get back to you shortly.

Best regards"""