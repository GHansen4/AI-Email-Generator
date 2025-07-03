import anthropic
from typing import Dict, Any, Optional, cast
from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Initialize Anthropic client with proper type handling
try:
    # Cast to Any to resolve type checking issues with dynamic attributes
    client = cast(Any, anthropic.Anthropic(api_key=settings.claude_api_key))
except Exception as e:
    logger.error("Failed to initialize Anthropic client", error=str(e))
    client = None


class AIResponseGenerator:
    """Generate AI responses using Anthropic Claude API that match user's writing style."""
    
    def __init__(self):
        self.default_model = "claude-3-haiku-20240307"  # Fast and cost-effective
        self.max_tokens = 500  # Reasonable limit for email responses
        self.client = client  # Store client reference
    
    def generate_response(self, original_email: str, user_profile: Dict[str, Any], 
                         context: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate an AI response matching the user's writing style.
        
        Args:
            original_email: The email that needs a response
            user_profile: User's writing style profile
            context: Additional context about the user or situation
            
        Returns:
            Dict containing the generated response and metadata
        """
        try:
            if not self.client:
                raise Exception("Anthropic client not initialized")
                
            # Import here to avoid circular imports
            from app.services.profile_analyzer import WritingProfileAnalyzer
            
            # Use ProfileAnalyzer for sophisticated prompt generation
            profile_analyzer = WritingProfileAnalyzer()
            
            # Check if we have comprehensive fingerprint data
            has_advanced_profile = 'comprehensive_fingerprint' in user_profile
            
            if has_advanced_profile:
                # Use sophisticated prompt generation
                style_prompt = profile_analyzer.generate_indistinguishable_prompt(user_profile)
                system_prompt = self._get_advanced_system_prompt(user_profile)
                user_prompt = self._build_advanced_prompt(original_email, style_prompt, context)
            else:
                # Fallback to basic prompt for limited profiles
                system_prompt = self._get_system_prompt(user_profile)
                user_prompt = self._build_basic_prompt(original_email, user_profile, context)
            
            logger.info("Generating AI response", 
                       model=self.default_model, 
                       profile_confidence=user_profile.get('confidence_score', 0),
                       advanced_profile=has_advanced_profile)
            
            # Make Claude API call with proper error handling
            try:
                # The anthropic client has dynamic attributes that may not be recognized by type checkers
                response = self.client.messages.create(
                    model=self.default_model,
                    max_tokens=self.max_tokens,
                    temperature=0.7,  # Slight randomness for natural responses
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
            except Exception as api_error:
                logger.error("Anthropic API error", error=str(api_error))
                raise api_error
            
            # Extract the response
            ai_response = response.content[0].text.strip()
            
            return {
                'response': ai_response,
                'prompt_used': user_prompt,
                'model': self.default_model,
                'tokens_used': response.usage.input_tokens + response.usage.output_tokens,
                'success': True
            }
            
        except Exception as e:
            logger.error("Error generating AI response", error=str(e))
            return {
                'response': self._get_fallback_response(),
                'prompt_used': user_prompt if 'user_prompt' in locals() else '',
                'model': self.default_model,
                'tokens_used': 0,
                'success': False,
                'error': str(e)
            }
    
    def _build_advanced_prompt(self, original_email: str, style_prompt: str, 
                              context: Optional[str] = None) -> str:
        """Build advanced prompt using ProfileAnalyzer's sophisticated style instructions."""
        
        prompt = f"""{style_prompt}

Original email to respond to:
\"\"\"
{original_email}
\"\"\"

Additional context: {context or "None provided"}

Write a response that:
1. Addresses all main points in the original email
2. Follows ALL the linguistic style requirements above
3. Is appropriate and helpful
4. Maintains the exact writing fingerprint described

Response:"""

        return prompt
    
    def _build_basic_prompt(self, original_email: str, user_profile: Dict[str, Any], 
                           context: Optional[str] = None) -> str:
        """Build basic prompt for profiles without advanced analysis (fallback)."""
        
        # Build basic style description
        style_description = self._describe_writing_style_basic(user_profile)
        
        prompt = f"""Please write a response to the following email. The response should match this specific writing style:

{style_description}

Original email to respond to:
\"\"\"
{original_email}
\"\"\"

Additional context: {context or "None provided"}

Please write a response that:
1. Addresses the main points in the original email
2. Matches the writing style described above
3. Is appropriate in tone and content
4. Includes a natural greeting and closing
5. Is helpful and professional

Response:"""

        return prompt
    
    def _get_advanced_system_prompt(self, user_profile: Dict[str, Any]) -> str:
        """Get system prompt for advanced linguistic profiles."""
        
        confidence = user_profile.get('confidence_score', 0)
        sample_count = user_profile.get('sample_count', 0)
        
        return f"""You are an expert AI writing assistant specialized in replicating individual writing styles with perfect authenticity. 

Profile Quality: {confidence:.1f} confidence from {sample_count} samples.

Your task is to write email responses that are completely indistinguishable from the person's actual writing. You have detailed linguistic analysis of their writing patterns including syntactic preferences, lexical sophistication, pragmatic competence, and cohesion patterns.

CRITICAL: Follow ALL style instructions precisely. Every word choice, sentence structure, and linguistic pattern must match their documented fingerprint. The response must sound exactly like they wrote it themselves.

Always be helpful, accurate, and appropriate while maintaining perfect stylistic authenticity."""
    
    def _get_system_prompt(self, user_profile: Dict[str, Any]) -> str:
        """Get basic system prompt for limited profiles (fallback)."""
        
        confidence = user_profile.get('confidence_score', 0)
        sample_count = user_profile.get('sample_count', 0)
        
        if confidence > 0.7 and sample_count > 5:
            confidence_note = "You have a strong understanding of this user's writing style."
        elif confidence > 0.4 and sample_count > 2:
            confidence_note = "You have a moderate understanding of this user's writing style."
        else:
            confidence_note = "You have limited data about this user's writing style, so aim for a professional but natural tone."
        
        return f"""You are an AI assistant that helps write email responses in a specific person's writing style. {confidence_note}

Your goal is to write responses that sound like they were written by the user themselves. Pay attention to:
- Formality level (formal vs casual)
- Sentence structure and length
- Common greetings and closings
- Vocabulary complexity
- Enthusiasm and tone
- Typical phrases and expressions

Always be helpful, accurate, and appropriate in your responses."""
    
    def _describe_writing_style_basic(self, user_profile: Dict[str, Any]) -> str:
        """Create basic style description for limited profiles (fallback)."""
        
        formality = user_profile.get('formality_score', 0.5)
        enthusiasm = user_profile.get('enthusiasm_score', 0.5)
        avg_sentence_length = user_profile.get('avg_sentence_length', 15)
        greetings = user_profile.get('common_greetings', [])
        closings = user_profile.get('common_closings', [])
        vocabulary_level = user_profile.get('vocabulary_level', 'medium')
        
        # Formality description
        if formality > 0.7:
            formality_desc = "very formal and professional"
        elif formality > 0.4:
            formality_desc = "moderately formal"
        else:
            formality_desc = "casual and friendly"
        
        # Enthusiasm description
        if enthusiasm > 0.7:
            enthusiasm_desc = "enthusiastic and energetic"
        elif enthusiasm > 0.4:
            enthusiasm_desc = "moderately positive"
        else:
            enthusiasm_desc = "neutral and measured"
        
        # Sentence length description
        if avg_sentence_length > 20:
            sentence_desc = "long, detailed sentences"
        elif avg_sentence_length > 10:
            sentence_desc = "medium-length sentences"
        else:
            sentence_desc = "short, concise sentences"
        
        # Build description
        style_parts = [
            f"Tone: {formality_desc} and {enthusiasm_desc}",
            f"Sentence style: Uses {sentence_desc}",
            f"Vocabulary: {vocabulary_level} complexity"
        ]
        
        if greetings:
            style_parts.append(f"Common greetings: {', '.join(greetings[:3])}")
        
        if closings:
            style_parts.append(f"Common closings: {', '.join(closings[:3])}")
        
        return "\n".join(f"- {part}" for part in style_parts)
    
    def _get_fallback_response(self) -> str:
        """Return a fallback response if AI generation fails."""
        return """Thank you for your email. I'll review this and get back to you shortly.

Best regards"""
    
    def generate_profile_response(self, user_email: str) -> str:
        """
        Generate a confirmation response when user sends profile email.
        
        Args:
            user_email: User's email address
            
        Returns:
            Confirmation message
        """
        return f"""Thank you for sharing your writing style with our AI email response service!

I've analyzed your email and added it to your writing profile. This will help me generate more personalized responses that match your communication style.

To use the service, simply forward any email you'd like me to respond to. I'll generate a draft response in your style and send it back to you.

Best regards,
AI Email Assistant
{settings.service_email}""" 