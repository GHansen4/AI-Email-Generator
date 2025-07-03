from typing import Dict, List, Any, Tuple, Optional
import statistics
import re
from collections import Counter
from app.utils.logging import get_logger

# Import NLP libraries with graceful fallbacks
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        SPACY_AVAILABLE = False
        nlp = None
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.tag import pos_tag
    
    # Try to download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
    
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

logger = get_logger(__name__)

class WritingProfileAnalyzer:
    """Analyze writing style patterns from user emails with NLTK-first approach."""

    def __init__(self):
        # Set up NLTK if available
        if NLTK_AVAILABLE:
            try:
                self.stop_words = set(stopwords.words('english'))
            except LookupError:
                self.stop_words = set()
        else:
            self.stop_words = set()

    def analyze_writing_style(self, email_content: str) -> Dict[str, Any]:
        """
        Comprehensive linguistic analysis for indistinguishable writing style matching.
        
        This method performs advanced linguistic profiling including syntactic patterns,
        lexical sophistication, pragmatic competence, cohesion analysis, and generates
        a detailed linguistic fingerprint for precise style matching.
        
        Args:
            email_content: Clean email content to analyze

        Returns:
            Dict containing comprehensive writing style metrics with linguistic fingerprint
        """
        try:
            # Basic preprocessing
            cleaned_text = self._preprocess_text(email_content)
            sentences = self._split_sentences(cleaned_text)
            words = self._tokenize_words(cleaned_text)
            paragraphs = self._split_paragraphs(cleaned_text)

            # Run all analyses
            profile = {
                # Basic metrics
                **self._analyze_basic_metrics(sentences, paragraphs, words),
                **self._analyze_style_patterns(cleaned_text),
                **self._analyze_linguistic_features(cleaned_text, words),
                **self._analyze_readability(cleaned_text),

                # Advanced linguistic features
                'syntactic_patterns': self.analyze_syntactic_patterns(cleaned_text, sentences),
                'dependency_patterns': self.analyze_dependency_patterns(cleaned_text),
                'lexical_sophistication': self.analyze_lexical_sophistication(cleaned_text, words),
                'cohesion_patterns': self.analyze_cohesion_patterns(cleaned_text, sentences),
                'pragmatic_competence': self.analyze_pragmatic_competence(cleaned_text),
                'temporal_deixis': self.analyze_temporal_deixis(cleaned_text)
            }

            # Generate comprehensive style fingerprint
            profile['comprehensive_fingerprint'] = self._generate_comprehensive_fingerprint(profile)

            # Add required fields for single-sample analysis
            profile['sample_count'] = 1
            profile['confidence_score'] = self._calculate_single_sample_confidence(profile)

            logger.info("Comprehensive writing style analysis completed",
                       word_count=profile.get('word_count', 0), 
                       formality=profile.get('formality_score', 0),
                       fingerprint_dimensions=len(profile.get('comprehensive_fingerprint', {})))

            return profile

        except Exception as e:
            logger.error("Error in comprehensive linguistic analysis", error=str(e))
            return self._get_default_profile()

    def merge_profiles(self, existing_profile: Dict[str, Any], new_analysis: Dict[str, Any],
                      existing_sample_count: int) -> Dict[str, Any]:
        """
        Merge new comprehensive analysis with existing writing profile.
        
        Args:
            existing_profile: Current writing profile
            new_analysis: New comprehensive analysis results
            existing_sample_count: Number of samples in existing profile
            
        Returns:
            Updated merged profile with weighted averages
        """
        try:
            new_sample_count = existing_sample_count + 1
            weight_existing = existing_sample_count / new_sample_count
            weight_new = 1 / new_sample_count

            # Merge numerical values using weighted average
            merged_profile = {}
            
            for key in ['formality_score', 'enthusiasm_score', 'politeness_score',
                       'avg_sentence_length', 'avg_paragraph_length', 'lexical_diversity']:
                existing_val = existing_profile.get(key, 0.5)
                new_val = new_analysis.get(key, 0.5)
                merged_profile[key] = existing_val * weight_existing + new_val * weight_new

            # Merge lists by combining and removing duplicates
            for key in ['common_greetings', 'common_closings', 'common_phrases']:
                existing_list = existing_profile.get(key, [])
                new_list = new_analysis.get(key, [])
                merged_profile[key] = list(set(existing_list + new_list))[:10]  # Limit to 10

            # Merge complex structures
            merged_profile['comprehensive_fingerprint'] = self._merge_fingerprints(
                existing_profile.get('comprehensive_fingerprint', {}),
                new_analysis.get('comprehensive_fingerprint', {}),
                weight_existing, weight_new
            )

            # Copy other important fields
            for key in ['word_count', 'sentence_count', 'paragraph_count', 'vocabulary_level']:
                merged_profile[key] = new_analysis.get(key, existing_profile.get(key, 0))

            # Add sample count and confidence
            merged_profile['sample_count'] = new_sample_count
            merged_profile['confidence_score'] = min(1.0, new_sample_count / 10.0)

            return merged_profile

        except Exception as e:
            logger.error("Error merging profiles", error=str(e))
            return existing_profile

    # Advanced linguistic analysis methods
    def analyze_syntactic_patterns(self, text: str, sentences: List[str]) -> Dict[str, Any]:
        """Analyze syntactic complexity and patterns."""
        try:
            patterns = {
                'avg_syntactic_complexity': 1.5,
                'sentence_type_distribution': {'simple': 0.4, 'compound': 0.3, 'complex': 0.2, 'compound_complex': 0.1},
                'coordination_preference': 0.3,
                'subordination_preference': 0.3
            }

            if NLTK_AVAILABLE and sentences:
                # Count conjunctions and complex structures
                text_lower = text.lower()
                coord_conjunctions = ['and', 'but', 'or', 'so', 'yet']
                subord_conjunctions = ['because', 'although', 'while', 'since', 'if', 'when', 'that']
                
                coord_count = sum(text_lower.count(conj) for conj in coord_conjunctions)
                subord_count = sum(text_lower.count(conj) for conj in subord_conjunctions)
                
                total_sentences = len(sentences)
                patterns['coordination_preference'] = min(1.0, coord_count / max(1, total_sentences))
                patterns['subordination_preference'] = min(1.0, subord_count / max(1, total_sentences))
                patterns['avg_syntactic_complexity'] = 1.0 + (coord_count + subord_count) / max(1, total_sentences)

            return patterns
        except Exception as e:
            logger.error("Error in syntactic analysis", error=str(e))
            return {'avg_syntactic_complexity': 1.5, 'coordination_preference': 0.3, 'subordination_preference': 0.3}

    def analyze_dependency_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze dependency patterns and grammatical structures."""
        try:
            patterns = {
                'passive_voice_ratio': 0.1,
                'question_formation_patterns': {'yes_no': 0.3, 'wh_questions': 0.5, 'tag_questions': 0.2},
                'active_voice_preference': 0.9
            }

            # Simple passive voice detection
            passive_indicators = ['was', 'were', 'been', 'being', 'be']
            text_words = text.lower().split()
            passive_count = sum(1 for word in text_words if word in passive_indicators)
            patterns['passive_voice_ratio'] = min(1.0, passive_count / max(1, len(text_words)) * 10)
            patterns['active_voice_preference'] = 1.0 - patterns['passive_voice_ratio']

            return patterns
        except Exception as e:
            logger.error("Error in dependency analysis", error=str(e))
            return {'passive_voice_ratio': 0.1, 'active_voice_preference': 0.9}

    def analyze_lexical_sophistication(self, text: str, words: List[str]) -> Dict[str, Any]:
        """Analyze lexical sophistication and vocabulary patterns."""
        try:
            if not words:
                return {'type_token_ratio': 0.5, 'lexical_density': 0.5, 'vocabulary_sophistication': 0.5}

            # Type-token ratio
            unique_words = set(word.lower() for word in words if word.isalpha())
            total_words = len([word for word in words if word.isalpha()])
            ttr = len(unique_words) / max(1, total_words)

            # Lexical density (content words vs function words)
            content_words = [word for word in words if word.lower() not in self.stop_words and word.isalpha()]
            lexical_density = len(content_words) / max(1, total_words)

            # Vocabulary sophistication based on word length
            avg_word_length = sum(len(word) for word in words if word.isalpha()) / max(1, total_words)
            vocab_sophistication = min(1.0, avg_word_length / 8.0)  # 8 chars = high sophistication

            return {
                'type_token_ratio': ttr,
                'moving_avg_ttr': ttr,  # Simplified
                'hapax_legomena_ratio': 0.3,  # Default
                'avg_word_frequency': 2.0,
                'avg_word_length': avg_word_length,
                'word_length_variability': statistics.stdev([len(w) for w in words if w.isalpha()]) if len(words) > 1 else 1.5,
                'lexical_density': lexical_density,
                'vocabulary_sophistication': vocab_sophistication
            }
        except Exception as e:
            logger.error("Error in lexical analysis", error=str(e))
            return {'type_token_ratio': 0.5, 'lexical_density': 0.5, 'vocabulary_sophistication': 0.5}

    def analyze_cohesion_patterns(self, text: str, sentences: List[str]) -> Dict[str, Any]:
        """Analyze text cohesion and coherence patterns."""
        try:
            text_lower = text.lower()
            
            # Reference patterns
            pronouns = ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that', 'these', 'those']
            pronoun_count = sum(text_lower.count(' ' + pron + ' ') for pron in pronouns)
            
            # Transition patterns
            transitions = {
                'additive': ['and', 'also', 'furthermore', 'moreover', 'additionally'],
                'adversative': ['but', 'however', 'nevertheless', 'although', 'despite'],
                'causal': ['because', 'therefore', 'thus', 'consequently', 'as a result'],
                'temporal': ['then', 'next', 'finally', 'meanwhile', 'afterwards']
            }
            
            transition_counts = {}
            for category, words in transitions.items():
                transition_counts[f'{category}_transitions'] = sum(text_lower.count(word) for word in words)
            
            total_transitions = sum(transition_counts.values())
            transition_density = total_transitions / max(1, len(sentences))

            return {
                'reference_patterns': {
                    'personal_pronouns': pronoun_count,
                    'demonstratives': text_lower.count('this') + text_lower.count('that'),
                    'definite_articles': text_lower.count('the'),
                    'possessives': text_lower.count('my') + text_lower.count('your') + text_lower.count('our')
                },
                'lexical_repetition_ratio': 0.3,  # Simplified
                'transition_usage': transition_counts,
                'transition_density': transition_density,
                'cohesion_score': min(1.0, (pronoun_count + total_transitions) / max(1, len(sentences)))
            }
        except Exception as e:
            logger.error("Error in cohesion analysis", error=str(e))
            return {'cohesion_score': 0.4}

    def analyze_pragmatic_competence(self, text: str) -> Dict[str, Any]:
        """Analyze pragmatic language use and speech acts."""
        try:
            text_lower = text.lower()
            
            # Speech act indicators
            speech_acts = {
                'requests': ['please', 'could you', 'would you', 'can you'],
                'suggestions': ['should', 'might', 'perhaps', 'maybe'],
                'apologies': ['sorry', 'apologize', 'excuse me'],
                'thanks': ['thank', 'thanks', 'grateful', 'appreciate'],
                'commitments': ['will', 'promise', 'commit', 'ensure']
            }
            
            speech_act_counts = {}
            for act, indicators in speech_acts.items():
                speech_act_counts[act] = sum(text_lower.count(indicator) for indicator in indicators)
            
            # Politeness markers
            politeness_markers = ['please', 'thank', 'excuse me', 'sorry', 'kindly']
            politeness_count = sum(text_lower.count(marker) for marker in politeness_markers)
            
            # Hedging and certainty
            hedges = ['maybe', 'perhaps', 'might', 'could', 'possibly']
            certainty_markers = ['definitely', 'certainly', 'absolutely', 'clearly']
            
            hedge_count = sum(text_lower.count(hedge) for hedge in hedges)
            certainty_count = sum(text_lower.count(marker) for marker in certainty_markers)
            
            word_count = len(text.split())
            
            return {
                'speech_act_distribution': speech_act_counts,
                'mitigation_density': hedge_count / max(1, word_count),
                'intensifier_density': certainty_count / max(1, word_count),
                'certainty_ratio': certainty_count / max(1, certainty_count + hedge_count) if (certainty_count + hedge_count) > 0 else 0.5,
                'politeness_complexity': min(1.0, politeness_count / max(1, word_count) * 20)
            }
        except Exception as e:
            logger.error("Error in pragmatic analysis", error=str(e))
            return {'politeness_complexity': 0.3}

    def analyze_temporal_deixis(self, text: str) -> Dict[str, Any]:
        """Analyze temporal reference patterns."""
        try:
            text_lower = text.lower()
            
            temporal_references = {
                'past': ['yesterday', 'ago', 'before', 'previously', 'earlier', 'was', 'were'],
                'present': ['now', 'today', 'currently', 'presently', 'is', 'are'],
                'future': ['tomorrow', 'later', 'soon', 'will', 'going to', 'next']
            }
            
            temporal_counts = {}
            for time_ref, words in temporal_references.items():
                temporal_counts[f'{time_ref}_reference'] = sum(text_lower.count(word) for word in words)
            
            total_temporal = sum(temporal_counts.values())
            
            # Normalize to proportions
            temporal_orientation = {}
            for key, count in temporal_counts.items():
                temporal_orientation[key] = count / max(1, total_temporal)
            
            # Temporal specificity
            specific_temporal = ['monday', 'tuesday', 'january', 'february', 'morning', 'afternoon', 'evening']
            specificity_count = sum(text_lower.count(word) for word in specific_temporal)
            
            return {
                'temporal_orientation': temporal_orientation,
                'temporal_specificity': specificity_count / max(1, len(text.split()))
            }
        except Exception as e:
            logger.error("Error in temporal analysis", error=str(e))
            return {'temporal_orientation': {'past_reference': 0.3, 'present_reference': 0.4, 'future_reference': 0.3}}

    # Helper methods
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis."""
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        # Remove email artifacts
        text = re.sub(r'[<>]', '', text)
        return text

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if NLTK_AVAILABLE:
            try:
                return sent_tokenize(text)
            except:
                pass
        # Fallback simple sentence splitting
        return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    def _tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if NLTK_AVAILABLE:
            try:
                return word_tokenize(text)
            except:
                pass
        # Fallback simple word tokenization
        return re.findall(r'\b\w+\b', text)

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        return [p.strip() for p in text.split('\n\n') if p.strip()]

    def _analyze_basic_metrics(self, sentences: List[str], paragraphs: List[str], words: List[str]) -> Dict[str, Any]:
        """Analyze basic text metrics."""
        word_count = len([w for w in words if w.isalpha()])
        sentence_count = len(sentences)
        paragraph_count = len(paragraphs)
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count,
            'avg_sentence_length': word_count / max(1, sentence_count),
            'avg_paragraph_length': sentence_count / max(1, paragraph_count)
        }

    def _analyze_style_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze writing style patterns."""
        text_lower = text.lower()
        
        # Formality indicators
        formal_words = ['furthermore', 'therefore', 'consequently', 'nevertheless', 'moreover']
        informal_words = ['gonna', 'kinda', 'sorta', 'yeah', 'ok', 'hey']
        
        formal_count = sum(text_lower.count(word) for word in formal_words)
        informal_count = sum(text_lower.count(word) for word in informal_words)
        
        # Enthusiasm indicators
        enthusiasm_indicators = ['!', 'exciting', 'amazing', 'wonderful', 'fantastic', 'great']
        enthusiasm_count = sum(text_lower.count(indicator) for indicator in enthusiasm_indicators)
        
        # Politeness indicators
        politeness_words = ['please', 'thank you', 'appreciate', 'grateful', 'kindly']
        politeness_count = sum(text_lower.count(word) for word in politeness_words)
        
        # Hedging
        hedging_words = ['maybe', 'perhaps', 'might', 'could', 'possibly', 'somewhat']
        hedging_count = sum(text_lower.count(word) for word in hedging_words)
        
        word_count = len(text.split())
        
        return {
            'formality_score': min(1.0, formal_count / max(1, formal_count + informal_count + 1)),
            'enthusiasm_score': min(1.0, enthusiasm_count / max(1, word_count) * 20),
            'politeness_score': min(1.0, politeness_count / max(1, word_count) * 20),
            'hedging_ratio': hedging_count / max(1, word_count),
            'common_greetings': self._extract_greetings(text),
            'common_closings': self._extract_closings(text),
            'common_phrases': []  # Simplified for now
        }

    def _analyze_linguistic_features(self, text: str, words: List[str]) -> Dict[str, Any]:
        """Analyze linguistic features."""
        if not words:
            return {'lexical_diversity': 0.5, 'vocabulary_level': 'medium'}
        
        # Lexical diversity
        unique_words = set(word.lower() for word in words if word.isalpha())
        total_words = len([word for word in words if word.isalpha()])
        lexical_diversity = len(unique_words) / max(1, total_words)
        
        # Vocabulary level based on average word length
        avg_word_length = sum(len(word) for word in words if word.isalpha()) / max(1, total_words)
        
        if avg_word_length < 4.5:
            vocab_level = 'simple'
        elif avg_word_length < 6:
            vocab_level = 'medium'
        else:
            vocab_level = 'advanced'
        
        return {
            'lexical_diversity': lexical_diversity,
            'vocabulary_level': vocab_level
        }

    def _analyze_readability(self, text: str) -> Dict[str, Any]:
        """Analyze text readability with simplified calculations."""
        # Use simplified readability estimation based on text characteristics
        if text:
            words = text.split()
            sentences = text.count('.') + text.count('!') + text.count('?')
            avg_word_length = sum(len(word.strip('.,!?')) for word in words) / max(1, len(words))
            avg_sentence_length = len(words) / max(1, sentences)
            
            # Simplified readability estimates
            estimated_grade = min(12.0, avg_sentence_length / 2.0 + avg_word_length)
            estimated_ease = max(30.0, 100.0 - estimated_grade * 5.0)
            
            return {
                'flesch_kincaid_grade': estimated_grade,
                'flesch_reading_ease': estimated_ease,
                'automated_readability_index': estimated_grade
            }
        
        # Default readability scores
        return {
            'flesch_kincaid_grade': 8.0,
            'flesch_reading_ease': 70.0,
            'automated_readability_index': 8.0
        }

    def _extract_greetings(self, text: str) -> List[str]:
        """Extract greeting patterns from text."""
        greetings = []
        text_lower = text.lower()
        
        common_greetings = ['hi', 'hello', 'dear', 'greetings', 'good morning', 'good afternoon']
        for greeting in common_greetings:
            if greeting in text_lower:
                greetings.append(greeting)
        
        return greetings[:5]  # Limit to 5

    def _extract_closings(self, text: str) -> List[str]:
        """Extract closing patterns from text."""
        closings = []
        text_lower = text.lower()
        
        common_closings = ['thanks', 'thank you', 'best regards', 'sincerely', 'cheers', 'best']
        for closing in common_closings:
            if closing in text_lower:
                closings.append(closing)
        
        return closings[:5]  # Limit to 5

    def _merge_fingerprints(self, existing: Dict[str, Any], new: Dict[str, Any], 
                           weight_existing: float, weight_new: float) -> Dict[str, Any]:
        """Merge two fingerprints using weighted averages."""
        merged = {}
        all_keys = set(existing.keys()) | set(new.keys())
        
        for key in all_keys:
            existing_val = existing.get(key, 0.5)
            new_val = new.get(key, 0.5)
            merged[key] = existing_val * weight_existing + new_val * weight_new
        
        return merged

    def _generate_comprehensive_fingerprint(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive linguistic fingerprint from profile data."""
        fingerprint = {}
        
        # Map profile metrics to fingerprint dimensions
        fingerprint['formality'] = profile.get('formality_score', 0.5)
        fingerprint['enthusiasm'] = profile.get('enthusiasm_score', 0.5)
        fingerprint['politeness_complexity'] = profile.get('politeness_score', 0.5)
        fingerprint['syntactic_complexity'] = profile.get('syntactic_patterns', {}).get('avg_syntactic_complexity', 1.5) / 3.0
        fingerprint['vocabulary_diversity'] = profile.get('lexical_diversity', 0.5)
        fingerprint['word_sophistication'] = profile.get('lexical_sophistication', {}).get('vocabulary_sophistication', 0.5)
        fingerprint['directness'] = 1.0 - min(1.0, profile.get('hedging_ratio', 0.1) * 10.0)
        
        # Additional derived dimensions
        fingerprint['sentence_complexity'] = min(1.0, profile.get('avg_sentence_length', 15) / 30.0)
        fingerprint['lexical_density'] = profile.get('lexical_sophistication', {}).get('lexical_density', 0.5)
        fingerprint['cohesion_strength'] = profile.get('cohesion_patterns', {}).get('cohesion_score', 0.4)
        
        # More dimensions for 16-dimension fingerprint
        fingerprint['temporal_orientation'] = 0.5  # Simplified
        fingerprint['pragmatic_sophistication'] = profile.get('pragmatic_competence', {}).get('politeness_complexity', 0.3)
        fingerprint['coordination_preference'] = profile.get('syntactic_patterns', {}).get('coordination_preference', 0.3)
        fingerprint['subordination_preference'] = profile.get('syntactic_patterns', {}).get('subordination_preference', 0.3)
        fingerprint['lexical_sophistication'] = profile.get('lexical_sophistication', {}).get('vocabulary_sophistication', 0.5)
        fingerprint['discourse_competence'] = profile.get('cohesion_patterns', {}).get('transition_density', 0.1) * 10.0
        
        # Normalize all values to 0-1 range
        for key, value in fingerprint.items():
            fingerprint[key] = max(0.0, min(1.0, float(value)))
        
        return fingerprint

    def _calculate_single_sample_confidence(self, profile: Dict[str, Any]) -> float:
        """Calculate confidence score for single-sample analysis."""
        # Base confidence on content quality indicators
        word_count = profile.get('word_count', 0)
        sentence_count = profile.get('sentence_count', 0)
        
        # Length-based confidence
        length_confidence = min(0.7, word_count / 100.0)  # Max 0.7 for 100+ words
        
        # Structure-based confidence  
        structure_confidence = min(0.3, sentence_count / 10.0)  # Max 0.3 for 10+ sentences
        
        # Combined confidence (max 0.8 for single sample)
        total_confidence = min(0.8, length_confidence + structure_confidence)
        
        return total_confidence

    def _get_default_profile(self) -> Dict[str, Any]:
        """Return default comprehensive writing profile for error cases."""
        # Create default advanced features
        default_profile = {
            # Basic metrics
            'avg_sentence_length': 15.0,
            'avg_paragraph_length': 3.0,
            'formality_score': 0.5,
            'enthusiasm_score': 0.5,
            'politeness_score': 0.5,
            'hedging_ratio': 0.1,
            'common_greetings': ['hi', 'hello'],
            'common_closings': ['thanks', 'best'],
            'common_phrases': [],
            'vocabulary_level': 'medium',
            'lexical_diversity': 0.5,
            'word_count': 0,
            'sentence_count': 0,
            'paragraph_count': 0,

            # Advanced linguistic features with default values 
            'syntactic_patterns': {
                'avg_syntactic_complexity': 1.5,
                'sentence_type_distribution': {'simple': 0.4, 'compound': 0.3, 'complex': 0.2, 'compound_complex': 0.1},
                'coordination_preference': 0.3,
                'subordination_preference': 0.3
            },
            'dependency_patterns': {
                'passive_voice_ratio': 0.1,
                'question_formation_patterns': {'yes_no': 0.3, 'wh_questions': 0.5, 'tag_questions': 0.2},
                'active_voice_preference': 0.9
            },
            'lexical_sophistication': {
                'type_token_ratio': 0.5,
                'moving_avg_ttr': 0.5,
                'hapax_legomena_ratio': 0.3,
                'avg_word_frequency': 2.0,
                'avg_word_length': 5.0,
                'word_length_variability': 1.5,
                'lexical_density': 0.5,
                'vocabulary_sophistication': 0.5
            },
            'cohesion_patterns': {
                'reference_patterns': {'personal_pronouns': 5, 'demonstratives': 2, 'definite_articles': 8, 'possessives': 3},
                'lexical_repetition_ratio': 0.3,
                'transition_usage': {'additive_transitions': 1, 'adversative_transitions': 1, 'causal_transitions': 1, 'temporal_transitions': 1, 'comparative_transitions': 0},
                'transition_density': 0.1,
                'cohesion_score': 0.4
            },
            'pragmatic_competence': {
                'speech_act_distribution': {'requests': 2, 'suggestions': 1, 'apologies': 1, 'thanks': 2, 'commitments': 1},
                'mitigation_density': 0.02,
                'intensifier_density': 0.01,
                'certainty_ratio': 0.5,
                'politeness_complexity': 0.3
            },
            'temporal_deixis': {
                'temporal_orientation': {'past_reference': 0.3, 'present_reference': 0.4, 'future_reference': 0.3},
                'temporal_specificity': 0.05
            },

            # Sample count and confidence
            'sample_count': 1,
            'confidence_score': 0.1
        }

        # Generate comprehensive fingerprint
        default_profile['comprehensive_fingerprint'] = self._generate_comprehensive_fingerprint(default_profile)

        return default_profile

    def generate_indistinguishable_prompt(self, writing_profile: Dict[str, Any]) -> str:
        """Generate a prompt for indistinguishable writing style matching."""
        try:
            fingerprint = writing_profile.get('comprehensive_fingerprint', {})
            
            # Build style description from fingerprint
            formality = fingerprint.get('formality', 0.5)
            enthusiasm = fingerprint.get('enthusiasm', 0.5)
            directness = fingerprint.get('directness', 0.5)
            
            style_desc = []
            
            if formality > 0.7:
                style_desc.append("formal and professional")
            elif formality < 0.3:
                style_desc.append("casual and conversational")
            else:
                style_desc.append("moderately formal")
            
            if enthusiasm > 0.7:
                style_desc.append("enthusiastic and energetic")
            elif enthusiasm < 0.3:
                style_desc.append("measured and reserved")
            
            if directness > 0.7:
                style_desc.append("direct and straightforward")
            elif directness < 0.3:
                style_desc.append("diplomatic and hedged")
            
            # Add structural preferences
            avg_sentence_length = writing_profile.get('avg_sentence_length', 15)
            if avg_sentence_length > 20:
                style_desc.append("using longer, complex sentences")
            elif avg_sentence_length < 10:
                style_desc.append("using short, concise sentences")
            
            # Add common patterns
            greetings = writing_profile.get('common_greetings', [])
            closings = writing_profile.get('common_closings', [])
            
            prompt = f"Write in a {', '.join(style_desc)} style"
            
            if greetings:
                prompt += f", using greetings like '{greetings[0]}'"
            if closings:
                prompt += f", and closing with phrases like '{closings[0]}'"
            
            prompt += ". Match the tone and structure exactly."
            
            return prompt
            
        except Exception as e:
            logger.error("Error generating prompt", error=str(e))
            return "Write in a professional and courteous tone."

class ProfileQualityValidator:
    """Validate and ensure quality of writing profiles for optimal AI generation."""
    
    def __init__(self):
        self.min_sample_count = 3
        self.min_confidence_threshold = 0.4
        self.quality_thresholds = {
            'excellent': 0.8,
            'good': 0.6,
            'fair': 0.4,
            'poor': 0.2
        }
    
    def validate_profile_quality(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive profile quality validation."""
        
        validation_result = {
            'overall_quality': 'poor',
            'confidence_score': profile.get('confidence_score', 0),
            'sample_count': profile.get('sample_count', 0),
            'quality_issues': [],
            'recommendations': [],
            'usability_score': 0.0,
            'dimension_scores': {},
            'is_production_ready': False
        }
        
        try:
            # Validate basic requirements
            basic_score = self._validate_basic_requirements(profile, validation_result)
            
            # Validate linguistic dimensions
            linguistic_score = self._validate_linguistic_dimensions(profile, validation_result)
            
            # Validate consistency
            consistency_score = self._validate_consistency(profile, validation_result)
            
            # Validate comprehensiveness
            comprehensiveness_score = self._validate_comprehensiveness(profile, validation_result)
            
            # Calculate overall usability score
            validation_result['usability_score'] = (
                basic_score * 0.3 +
                linguistic_score * 0.3 +
                consistency_score * 0.2 +
                comprehensiveness_score * 0.2
            )
            
            # Determine overall quality
            validation_result['overall_quality'] = self._determine_quality_level(
                validation_result['usability_score']
            )
            
            # Determine production readiness
            validation_result['is_production_ready'] = (
                validation_result['usability_score'] >= self.min_confidence_threshold and
                validation_result['sample_count'] >= self.min_sample_count and
                len(validation_result['quality_issues']) <= 2
            )
            
            # Generate recommendations
            self._generate_recommendations(validation_result)
            
            logger.info("Profile quality validation completed",
                       quality=validation_result['overall_quality'],
                       usability_score=validation_result['usability_score'],
                       production_ready=validation_result['is_production_ready'])
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Profile validation error: {e}")
            validation_result['quality_issues'].append(f"Validation error: {str(e)}")
            return validation_result
    
    def _validate_basic_requirements(self, profile: Dict[str, Any], 
                                   validation_result: Dict[str, Any]) -> float:
        """Validate basic profile requirements."""
        
        score = 0.0
        issues = validation_result['quality_issues']
        
        # Sample count validation
        sample_count = profile.get('sample_count', 0)
        if sample_count < 1:
            issues.append("No writing samples available")
        elif sample_count < 3:
            issues.append("Insufficient samples for reliable analysis (minimum 3 recommended)")
            score += 0.3
        elif sample_count < 5:
            score += 0.6
        else:
            score += 1.0
        
        # Confidence score validation
        confidence = profile.get('confidence_score', 0)
        if confidence < 0.2:
            issues.append("Very low confidence score indicates unreliable profile")
        elif confidence < 0.4:
            issues.append("Low confidence score may affect response quality")
            score += 0.3
        elif confidence < 0.7:
            score += 0.6
        else:
            score += 1.0
        
        # Essential fields validation
        essential_fields = ['formality_score', 'enthusiasm_score', 'avg_sentence_length']
        missing_fields = [field for field in essential_fields if field not in profile]
        if missing_fields:
            issues.append(f"Missing essential fields: {', '.join(missing_fields)}")
        else:
            score += 0.5
        
        validation_result['dimension_scores']['basic_requirements'] = score / 2.5
        return score / 2.5
    
    def _validate_linguistic_dimensions(self, profile: Dict[str, Any], 
                                      validation_result: Dict[str, Any]) -> float:
        """Validate linguistic analysis dimensions."""
        
        score = 0.0
        issues = validation_result['quality_issues']
        
        # Check for comprehensive fingerprint
        fingerprint = profile.get('comprehensive_fingerprint', {})
        if not fingerprint:
            issues.append("Missing comprehensive linguistic fingerprint")
            return 0.1
        
        # Required fingerprint dimensions
        required_dimensions = [
            'syntactic_complexity', 'formality', 'enthusiasm', 'directness',
            'vocabulary_diversity', 'word_sophistication', 'politeness_complexity'
        ]
        
        missing_dimensions = [dim for dim in required_dimensions if dim not in fingerprint]
        if missing_dimensions:
            issues.append(f"Missing fingerprint dimensions: {', '.join(missing_dimensions)}")
            score += 0.3
        else:
            score += 1.0
        
        # Validate dimension value ranges (should be 0-1)
        invalid_ranges = []
        for dim, value in fingerprint.items():
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                invalid_ranges.append(dim)
        
        if invalid_ranges:
            issues.append(f"Invalid dimension ranges: {', '.join(invalid_ranges)}")
        else:
            score += 0.5
        
        # Check for advanced linguistic features
        advanced_features = ['syntactic_patterns', 'dependency_patterns', 'lexical_sophistication']
        present_features = [feat for feat in advanced_features if feat in profile]
        
        if len(present_features) == len(advanced_features):
            score += 1.0
        elif len(present_features) >= 2:
            score += 0.7
        elif len(present_features) >= 1:
            score += 0.4
        else:
            issues.append("Missing advanced linguistic analysis features")
        
        validation_result['dimension_scores']['linguistic_dimensions'] = score / 2.5
        return score / 2.5
    
    def _validate_consistency(self, profile: Dict[str, Any], 
                            validation_result: Dict[str, Any]) -> float:
        """Validate internal consistency of profile data."""
        
        score = 1.0
        issues = validation_result['quality_issues']
        
        # Check formality consistency
        formality_main = profile.get('formality_score', 0.5)
        formality_fingerprint = profile.get('comprehensive_fingerprint', {}).get('formality', 0.5)
        
        if abs(formality_main - formality_fingerprint) > 0.3:
            issues.append("Inconsistent formality scores between main profile and fingerprint")
            score -= 0.2
        
        # Check enthusiasm consistency
        enthusiasm_main = profile.get('enthusiasm_score', 0.5)
        enthusiasm_fingerprint = profile.get('comprehensive_fingerprint', {}).get('enthusiasm', 0.5)
        
        if abs(enthusiasm_main - enthusiasm_fingerprint) > 0.3:
            issues.append("Inconsistent enthusiasm scores")
            score -= 0.2
        
        # Check sentence length reasonableness
        avg_length = profile.get('avg_sentence_length', 15)
        if avg_length < 3 or avg_length > 50:
            issues.append(f"Unrealistic average sentence length: {avg_length}")
            score -= 0.3
        
        # Check vocabulary level consistency with word sophistication
        vocab_level = profile.get('vocabulary_level', 'medium')
        word_sophistication = profile.get('comprehensive_fingerprint', {}).get('word_sophistication', 0.5)
        
        expected_sophistication = {'simple': 0.3, 'medium': 0.5, 'advanced': 0.8}
        expected = expected_sophistication.get(vocab_level, 0.5)
        
        if abs(word_sophistication - expected) > 0.4:
            issues.append("Vocabulary level inconsistent with word sophistication score")
            score -= 0.2
        
        # Check for reasonable confidence vs sample count relationship
        confidence = profile.get('confidence_score', 0)
        sample_count = profile.get('sample_count', 0)
        
        # Confidence should generally increase with sample count
        expected_confidence = min(1.0, sample_count / 10.0)
        if confidence > expected_confidence + 0.3:
            issues.append("Confidence score suspiciously high for sample count")
            score -= 0.1
        
        validation_result['dimension_scores']['consistency'] = max(0.0, score)
        return max(0.0, score)
    
    def _validate_comprehensiveness(self, profile: Dict[str, Any], 
                                  validation_result: Dict[str, Any]) -> float:
        """Validate comprehensiveness of profile data."""
        
        score = 0.0
        issues = validation_result['quality_issues']
        
        # Check for greeting/closing patterns
        greetings = profile.get('common_greetings', [])
        closings = profile.get('common_closings', [])
        
        if not greetings:
            issues.append("No greeting patterns identified")
        else:
            score += 0.2
        
        if not closings:
            issues.append("No closing patterns identified")
        else:
            score += 0.2
        
        # Check for phrase patterns
        phrases = profile.get('common_phrases', [])
        if len(phrases) < 3:
            issues.append("Limited characteristic phrases identified")
        else:
            score += 0.2
        
        # Check for advanced features
        cohesion_patterns = profile.get('cohesion_patterns', {})
        pragmatic_competence = profile.get('pragmatic_competence', {})
        temporal_deixis = profile.get('temporal_deixis', {})
        
        advanced_feature_count = sum([
            bool(cohesion_patterns),
            bool(pragmatic_competence),
            bool(temporal_deixis)
        ])
        
        score += (advanced_feature_count / 3) * 0.4
        
        validation_result['dimension_scores']['comprehensiveness'] = score
        return score
    
    def _determine_quality_level(self, usability_score: float) -> str:
        """Determine overall quality level."""
        
        if usability_score >= self.quality_thresholds['excellent']:
            return 'excellent'
        elif usability_score >= self.quality_thresholds['good']:
            return 'good'
        elif usability_score >= self.quality_thresholds['fair']:
            return 'fair'
        else:
            return 'poor'
    
    def _generate_recommendations(self, validation_result: Dict[str, Any]) -> None:
        """Generate specific recommendations for profile improvement."""
        
        recommendations = validation_result['recommendations']
        sample_count = validation_result['sample_count']
        quality = validation_result['overall_quality']
        
        # Sample count recommendations
        if sample_count < 3:
            recommendations.append("Add more writing samples (minimum 3, recommended 5-10)")
        elif sample_count < 5:
            recommendations.append("Add 2-3 more samples to improve reliability")
        elif sample_count < 10 and quality != 'excellent':
            recommendations.append("Consider adding more samples for better style matching")
        
        # Quality-specific recommendations
        if quality == 'poor':
            recommendations.extend([
                "Profile needs significant improvement before production use",
                "Focus on adding high-quality writing samples",
                "Ensure samples contain substantial original writing (50+ words each)"
            ])
        elif quality == 'fair':
            recommendations.extend([
                "Profile is usable but could be improved",
                "Add more diverse writing samples",
                "Consider samples from different contexts (formal/informal)"
            ])
        elif quality == 'good':
            recommendations.append("Profile is good - minor improvements possible with more samples")
        else:  # excellent
            recommendations.append("Profile is excellent and ready for production use")
        
        # Specific issue recommendations
        issues = validation_result['quality_issues']
        if any('fingerprint' in issue.lower() for issue in issues):
            recommendations.append("Re-analyze existing samples to generate missing linguistic features")
        
        if any('consistency' in issue.lower() for issue in issues):
            recommendations.append("Review and re-validate profile data for consistency")
        
        if any('patterns' in issue.lower() for issue in issues):
            recommendations.append("Add samples with clear greeting/closing patterns")

class ProfileOptimizer:
    """Optimize profiles for better AI response generation."""
    
    def __init__(self):
        self.validator = ProfileQualityValidator()
    
    def optimize_profile_for_generation(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize profile specifically for AI response generation."""
        
        try:
            # Validate current quality
            validation = self.validator.validate_profile_quality(profile)
            
            # Create optimized profile
            optimized_profile = profile.copy()
            
            # Enhance fingerprint for generation
            self._optimize_fingerprint(optimized_profile)
            
            # Ensure essential patterns are present
            self._ensure_essential_patterns(optimized_profile)
            
            # Optimize confidence scoring
            self._optimize_confidence_scoring(optimized_profile, validation)
            
            # Add generation-specific metadata
            optimized_profile['generation_optimized'] = True
            optimized_profile['optimization_timestamp'] = validation
            
            logger.info("Profile optimized for generation",
                       original_quality=validation['overall_quality'],
                       original_usability=validation['usability_score'])
            
            return optimized_profile
            
        except Exception as e:
            logger.error(f"Profile optimization error: {e}")
            return profile  # Return the original profile instead of None

    def _optimize_fingerprint(self, profile: Dict[str, Any]) -> None:
        """Optimize linguistic fingerprint for better AI generation."""
        fingerprint = profile.get('comprehensive_fingerprint', {})
        
        # Normalize fingerprint values to ensure they're in 0-1 range
        for key, value in fingerprint.items():
            if isinstance(value, (int, float)):
                fingerprint[key] = max(0.0, min(1.0, float(value)))
        
        profile['comprehensive_fingerprint'] = fingerprint

    def _ensure_essential_patterns(self, profile: Dict[str, Any]) -> None:
        """Ensure essential patterns are present for generation."""
        # Ensure minimum greeting/closing patterns
        if not profile.get('common_greetings'):
            profile['common_greetings'] = ['hello', 'hi', 'dear']
        
        if not profile.get('common_closings'):
            profile['common_closings'] = ['best regards', 'thank you', 'sincerely']

    def _optimize_confidence_scoring(self, profile: Dict[str, Any], validation: Dict[str, Any]) -> None:
        """Optimize confidence scoring based on validation results."""
        # Use validation results to adjust confidence
        usability_score = validation.get('usability_score', 0.5)
        sample_count = profile.get('sample_count', 1)
        
        # Adjust confidence based on both usability and sample count
        adjusted_confidence = min(1.0, (usability_score + sample_count / 10.0) / 2.0)
        profile['confidence_score'] = adjusted_confidence