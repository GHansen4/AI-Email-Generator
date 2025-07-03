import re
import statistics
from typing import Dict, List, Any, Optional
from collections import Counter
from app.utils.logging import get_logger

# Import NLP libraries with graceful fallbacks
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        nlp = None
        SPACY_AVAILABLE = False
except ImportError:
    spacy = None
    nlp = None
    SPACY_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
    
    # Ensure required NLTK data is available
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        pass  # Data will be downloaded when needed
        
except ImportError:
    nltk = None
    NLTK_AVAILABLE = False

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    textstat = None
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
        
        # Common patterns for analysis
        self.common_greetings = [
            'hi', 'hello', 'hey', 'dear', 'greetings', 'good morning', 
            'good afternoon', 'good evening', 'hope you', 'i hope'
        ]
        
        self.common_closings = [
            'best', 'regards', 'sincerely', 'thanks', 'thank you', 
            'cheers', 'talk soon', 'best wishes', 'warmly', 'yours'
        ]
        
        # Formality indicators
        self.formal_indicators = [
            'please', 'kindly', 'would you', 'could you', 'i would like',
            'i am writing', 'dear sir', 'dear madam', 'yours truly',
            'yours sincerely', 'respectfully'
        ]
        
        self.casual_indicators = [
            'hey', 'hi there', 'what\'s up', 'how\'s it going', 'thanks!',
            'cheers', 'catch you', 'talk soon', 'let me know', 'sounds good'
        ]
        
        # Enhanced patterns for advanced analysis
        self.politeness_markers = {
            'high': ['would you mind', 'if you could', 'would it be possible', 'i would appreciate'],
            'medium': ['please', 'could you', 'would you', 'if possible'],
            'low': ['can you', 'do this', 'send me', 'i need']
        }
        
        self.hedging_patterns = [
            'i think', 'i believe', 'perhaps', 'maybe', 'might', 'could be',
            'seems like', 'appears to', 'sort of', 'kind of', 'somewhat'
        ]
        
        self.discourse_markers = {
            'additive': ['also', 'furthermore', 'moreover', 'additionally', 'besides'],
            'adversative': ['however', 'but', 'nevertheless', 'on the other hand'],
            'causal': ['therefore', 'thus', 'consequently', 'as a result'],
            'temporal': ['first', 'then', 'next', 'finally', 'meanwhile']
        }
    
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
            
            # Run all analyses
            profile = {
                # Basic metrics (from existing code)
                **self._analyze_basic_metrics(sentences, [], words),
                **self._analyze_style_patterns(cleaned_text),
                **self._analyze_linguistic_features(cleaned_text, words),
                
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
        
        Handles both basic and advanced linguistic features from comprehensive analysis.
        
        Args:
            existing_profile: Current writing profile
            new_analysis: New comprehensive analysis results
            existing_sample_count: Number of samples in existing profile
            
        Returns:
            Updated writing profile with merged comprehensive features
        """
        try:
            new_sample_count = existing_sample_count + 1
            weight_existing = existing_sample_count / new_sample_count
            weight_new = 1 / new_sample_count
            
            # Merge numerical values using weighted average
            merged_profile = {}
            
            # Basic metrics
            for key in ['avg_sentence_length', 'avg_paragraph_length', 'formality_score', 
                       'enthusiasm_score', 'politeness_score', 'hedging_ratio']:
                existing_val = existing_profile.get(key, 0.5)
                new_val = new_analysis.get(key, 0.5)
                merged_profile[key] = existing_val * weight_existing + new_val * weight_new
            
            # Merge advanced linguistic features
            # Syntactic patterns
            if 'syntactic_patterns' in new_analysis:
                merged_profile['syntactic_patterns'] = new_analysis['syntactic_patterns']
            
            # Dependency patterns
            if 'dependency_patterns' in new_analysis:
                merged_profile['dependency_patterns'] = new_analysis['dependency_patterns']
            
            # Lexical sophistication
            if 'lexical_sophistication' in new_analysis:
                merged_profile['lexical_sophistication'] = new_analysis['lexical_sophistication']
            
            # Cohesion patterns
            if 'cohesion_patterns' in new_analysis:
                merged_profile['cohesion_patterns'] = new_analysis['cohesion_patterns']
            
            # Pragmatic competence
            if 'pragmatic_competence' in new_analysis:
                merged_profile['pragmatic_competence'] = new_analysis['pragmatic_competence']
            
            # Temporal deixis
            if 'temporal_deixis' in new_analysis:
                merged_profile['temporal_deixis'] = new_analysis['temporal_deixis']
            
            # Merge lists (keep most frequent items)
            for key in ['common_greetings', 'common_closings', 'common_phrases']:
                merged_profile[key] = self._merge_common_items(
                    existing_profile.get(key, []),
                    new_analysis.get(key, [])
                )
            
            # Update vocabulary level
            merged_profile['vocabulary_level'] = self._merge_vocabulary_level(
                existing_profile.get('vocabulary_level', 'medium'),
                new_analysis.get('vocabulary_level', 'medium')
            )
            
            # Generate updated comprehensive fingerprint
            merged_profile['comprehensive_fingerprint'] = self._generate_comprehensive_fingerprint(merged_profile)
            
            # Add sample count and confidence
            merged_profile['sample_count'] = new_sample_count
            merged_profile['confidence_score'] = min(1.0, new_sample_count / 10.0)
            
            return merged_profile
            
        except Exception as e:
            logger.error("Error merging comprehensive profiles", error=str(e))
            return existing_profile
    
    def analyze_syntactic_patterns(self, text: str, sentences: List[str]) -> Dict[str, Any]:
        """Analyze deep syntactic patterns that create linguistic fingerprints."""
        
        # Sentence complexity analysis
        complexity_scores = []
        sentence_types = {'simple': 0, 'compound': 0, 'complex': 0, 'compound_complex': 0}
        
        for sentence in sentences:
            # Count clauses by conjunction patterns
            clause_count = 1 + sentence.count(' and ') + sentence.count(' but ') + sentence.count(' or ')
            complexity_scores.append(clause_count)
            
            # Classify sentence types based on conjunctions and subordinators
            has_coord = any(conj in sentence.lower() for conj in [' and ', ' but ', ' or ', ' yet '])
            has_subord = any(sub in sentence.lower() for sub in [' because ', ' although ', ' if ', ' when ', ' while ', ' since '])
            
            if has_coord and has_subord:
                sentence_types['compound_complex'] += 1
            elif has_coord:
                sentence_types['compound'] += 1
            elif has_subord:
                sentence_types['complex'] += 1
            else:
                sentence_types['simple'] += 1
        
        # Calculate syntactic complexity ratio
        avg_clause_complexity = statistics.mean(complexity_scores) if complexity_scores else 1
        
        # Analyze coordination vs subordination preference
        total_sentences = len(sentences)
        sentence_type_ratios = {}
        if total_sentences > 0:
            for key, count in sentence_types.items():
                sentence_type_ratios[key] = count / total_sentences
        else:
            sentence_type_ratios = {key: 0.0 for key in sentence_types}
        
        return {
            'avg_syntactic_complexity': avg_clause_complexity,
            'sentence_type_distribution': sentence_type_ratios,
            'coordination_preference': sentence_type_ratios['compound'] + sentence_type_ratios['compound_complex'],
            'subordination_preference': sentence_type_ratios['complex'] + sentence_type_ratios['compound_complex']
        }

    def analyze_dependency_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze grammatical dependency patterns (even without spaCy)."""
        
        # Passive voice detection using heuristics
        passive_indicators = [
            r'\b(was|were|is|are|am|been|being)\s+\w+ed\b',
            r'\b(was|were|is|are|am|been|being)\s+\w+en\b',
            r'\bby\s+\w+\s*$'  # "by someone" at end of sentence
        ]
        
        sentences = self._split_sentences(text)
        passive_count = 0
        
        for sentence in sentences:
            for pattern in passive_indicators:
                if re.search(pattern, sentence, re.IGNORECASE):
                    passive_count += 1
                    break
        
        passive_ratio = passive_count / len(sentences) if sentences else 0
        
        # Question formation patterns
        question_counts = {
            'yes_no': len(re.findall(r'\b(do|does|did|will|would|can|could|should|is|are|was|were)\s+\w+.*\?', text, re.IGNORECASE)),
            'wh_questions': len(re.findall(r'\b(what|when|where|who|why|how)\s+.*\?', text, re.IGNORECASE)),
            'tag_questions': len(re.findall(r',\s*(right|correct|ok|okay|yes|no)\?', text, re.IGNORECASE))
        }
        
        total_questions = text.count('?')
        question_ratios = {}
        if total_questions > 0:
            for key, count in question_counts.items():
                question_ratios[key] = count / total_questions
        else:
            question_ratios = {key: 0.0 for key in question_counts}
        
        return {
            'passive_voice_ratio': passive_ratio,
            'question_formation_patterns': question_ratios,
            'active_voice_preference': 1 - passive_ratio
        }

    def analyze_lexical_sophistication(self, text: str, words: List[str]) -> Dict[str, Any]:
        """Deep lexical analysis for vocabulary fingerprinting."""
        
        # Advanced vocabulary metrics
        content_words = [w for w in words if w.isalpha() and len(w) > 2 and w not in self.stop_words]
        
        if not content_words:
            return {'lexical_sophistication': 0, 'vocabulary_diversity': 0, 'rare_word_usage': 0}
        
        # Type-Token Ratio (vocabulary diversity)
        unique_words = set(content_words)
        ttr = len(unique_words) / len(content_words)
        
        # Moving Average Type-Token Ratio (more stable)
        mattr = self._calculate_moving_average_ttr(content_words)
        
        # Word frequency analysis
        word_freq = Counter(content_words)
        
        # Hapax legomena (words used only once) - indicator of vocabulary range
        hapax_ratio = sum(1 for count in word_freq.values() if count == 1) / len(word_freq)
        
        # Average word frequency (lower = more diverse vocabulary)
        avg_word_freq = statistics.mean(word_freq.values())
        
        # Word length distribution
        word_lengths = [len(word) for word in content_words]
        avg_word_length = statistics.mean(word_lengths)
        word_length_std = statistics.stdev(word_lengths) if len(word_lengths) > 1 else 0
        
        # Lexical density (content words vs total words)
        lexical_density = len(content_words) / len(words)
        
        return {
            'type_token_ratio': ttr,
            'moving_avg_ttr': mattr,
            'hapax_legomena_ratio': hapax_ratio,
            'avg_word_frequency': avg_word_freq,
            'avg_word_length': avg_word_length,
            'word_length_variability': word_length_std,
            'lexical_density': lexical_density,
            'vocabulary_sophistication': (avg_word_length + ttr + (1 - hapax_ratio)) / 3
        }

    def _calculate_moving_average_ttr(self, words: List[str], window_size: int = 50) -> float:
        """Calculate Moving Average Type-Token Ratio for vocabulary diversity."""
        if len(words) < window_size:
            return len(set(words)) / len(words) if words else 0
        
        ttrs = []
        for i in range(len(words) - window_size + 1):
            window = words[i:i + window_size]
            ttr = len(set(window)) / len(window)
            ttrs.append(ttr)
        
        return statistics.mean(ttrs)

    def analyze_cohesion_patterns(self, text: str, sentences: List[str]) -> Dict[str, Any]:
        """Analyze how user connects ideas and maintains text cohesion."""
        
        # Reference patterns (pronouns, demonstratives)
        reference_patterns = {
            'personal_pronouns': len(re.findall(r'\b(i|you|he|she|it|we|they|me|him|her|us|them)\b', text, re.IGNORECASE)),
            'demonstratives': len(re.findall(r'\b(this|that|these|those)\b', text, re.IGNORECASE)),
            'definite_articles': len(re.findall(r'\bthe\b', text, re.IGNORECASE)),
            'possessives': len(re.findall(r'\b(my|your|his|her|its|our|their)\b', text, re.IGNORECASE))
        }
        
        # Lexical cohesion (word repetition patterns)
        words = text.lower().split()
        content_words = [w for w in words if w.isalpha() and len(w) > 3 and w not in self.stop_words]
        word_freq = Counter(content_words)
        
        # Calculate lexical repetition ratio
        repeated_words = sum(count for count in word_freq.values() if count > 1)
        total_content_words = len(content_words)
        lexical_repetition_ratio = repeated_words / total_content_words if total_content_words > 0 else 0
        
        # Sentence transitions and connectives
        transition_patterns = {
            'additive': ['furthermore', 'moreover', 'additionally', 'also', 'besides'],
            'adversative': ['however', 'nevertheless', 'nonetheless', 'conversely', 'on the contrary'],
            'causal': ['therefore', 'consequently', 'thus', 'hence', 'as a result'],
            'temporal': ['meanwhile', 'subsequently', 'previously', 'finally', 'next'],
            'comparative': ['similarly', 'likewise', 'in contrast', 'on the other hand']
        }
        
        transition_usage = {}
        for category, transitions in transition_patterns.items():
            count = sum(1 for transition in transitions if transition in text.lower())
            transition_usage[f'{category}_transitions'] = count
        
        total_transitions = sum(transition_usage.values())
        transition_density = total_transitions / len(sentences) if sentences else 0
        
        return {
            'reference_patterns': reference_patterns,
            'lexical_repetition_ratio': lexical_repetition_ratio,
            'transition_usage': transition_usage,
            'transition_density': transition_density,
            'cohesion_score': (lexical_repetition_ratio + transition_density) / 2
        }

    def analyze_pragmatic_competence(self, text: str) -> Dict[str, Any]:
        """Analyze pragmatic language use - how user accomplishes social functions."""
        
        # Speech act patterns
        speech_acts = {
            'requests': len(re.findall(r'\b(please|could you|would you|can you|may i|might i)\b', text, re.IGNORECASE)),
            'suggestions': len(re.findall(r'\b(should|might want to|perhaps|maybe|how about|what about)\b', text, re.IGNORECASE)),
            'apologies': len(re.findall(r'\b(sorry|apologize|forgive me|my mistake|my bad)\b', text, re.IGNORECASE)),
            'thanks': len(re.findall(r'\b(thank|thanks|grateful|appreciate|much obliged)\b', text, re.IGNORECASE)),
            'commitments': len(re.findall(r'\b(will|shall|promise|guarantee|ensure|commit)\b', text, re.IGNORECASE))
        }
        
        # Mitigation strategies (softening language)
        mitigation_patterns = [
            'sort of', 'kind of', 'somewhat', 'rather', 'quite', 'pretty',
            'i think', 'i believe', 'it seems', 'appears to', 'tends to',
            'might', 'could', 'may', 'perhaps', 'possibly', 'probably'
        ]
        
        mitigation_count = sum(1 for pattern in mitigation_patterns if pattern in text.lower())
        mitigation_density = mitigation_count / len(text.split())
        
        # Intensification patterns (strengthening language)
        intensifiers = [
            'very', 'extremely', 'really', 'absolutely', 'definitely', 'certainly',
            'completely', 'totally', 'quite', 'rather', 'pretty', 'fairly'
        ]
        
        intensifier_count = sum(1 for intensifier in intensifiers if intensifier in text.lower())
        intensifier_density = intensifier_count / len(text.split())
        
        # Modality patterns (expressing certainty/uncertainty)
        certainty_markers = ['definitely', 'certainly', 'obviously', 'clearly', 'undoubtedly']
        uncertainty_markers = ['maybe', 'perhaps', 'possibly', 'might', 'could', 'seems']
        
        certainty_count = sum(1 for marker in certainty_markers if marker in text.lower())
        uncertainty_count = sum(1 for marker in uncertainty_markers if marker in text.lower())
        
        certainty_ratio = certainty_count / (certainty_count + uncertainty_count) if (certainty_count + uncertainty_count) > 0 else 0.5
        
        return {
            'speech_act_distribution': speech_acts,
            'mitigation_density': mitigation_density,
            'intensifier_density': intensifier_density,
            'certainty_ratio': certainty_ratio,
            'politeness_complexity': (mitigation_density + speech_acts['requests'] / len(text.split())) / 2
        }

    def analyze_temporal_deixis(self, text: str) -> Dict[str, Any]:
        """Analyze temporal reference patterns."""
        
        temporal_counts = {
            'past_reference': len(re.findall(r'\b(yesterday|last|ago|previously|earlier|before)\b', text, re.IGNORECASE)),
            'present_reference': len(re.findall(r'\b(now|today|currently|presently|at the moment)\b', text, re.IGNORECASE)),
            'future_reference': len(re.findall(r'\b(tomorrow|later|soon|next|upcoming|will|going to)\b', text, re.IGNORECASE))
        }
        
        total_temporal = sum(temporal_counts.values())
        temporal_ratios = {}
        if total_temporal > 0:
            for key, count in temporal_counts.items():
                temporal_ratios[key] = count / total_temporal
        else:
            temporal_ratios = {key: 0.0 for key in temporal_counts}
        
        return {
            'temporal_orientation': temporal_ratios,
            'temporal_specificity': total_temporal / len(text.split()) if text.split() else 0
        }

    def _generate_comprehensive_fingerprint(self, profile: Dict[str, Any]) -> Dict[str, float]:
        """Generate detailed linguistic fingerprint for precise matching."""
        
        try:
            # Extract key patterns
            syntactic = profile.get('syntactic_patterns', {})
            dependency = profile.get('dependency_patterns', {})
            lexical = profile.get('lexical_sophistication', {})
            cohesion = profile.get('cohesion_patterns', {})
            pragmatic = profile.get('pragmatic_competence', {})
            
            fingerprint = {
                # Syntactic complexity
                'syntactic_complexity': syntactic.get('avg_syntactic_complexity', 1.0) / 3.0,
                'coordination_preference': syntactic.get('coordination_preference', 0.3),
                'subordination_preference': syntactic.get('subordination_preference', 0.3),
                
                # Voice and question patterns
                'active_voice_preference': dependency.get('active_voice_preference', 0.8),
                'question_directness': dependency.get('question_formation_patterns', {}).get('yes_no', 0.5),
                
                # Lexical sophistication
                'vocabulary_diversity': lexical.get('type_token_ratio', 0.5),
                'word_sophistication': min(1.0, lexical.get('avg_word_length', 5) / 8),
                'lexical_density': lexical.get('lexical_density', 0.5),
                
                # Cohesion and flow
                'cohesion_strength': cohesion.get('cohesion_score', 0.5),
                'transition_usage': cohesion.get('transition_density', 0.1),
                
                # Pragmatic style
                'mitigation_tendency': pragmatic.get('mitigation_density', 0.1) * 10,
                'certainty_level': pragmatic.get('certainty_ratio', 0.5),
                'politeness_complexity': pragmatic.get('politeness_complexity', 0.3),
                
                # Overall style markers
                'formality': profile.get('formality_score', 0.5),
                'enthusiasm': profile.get('enthusiasm_score', 0.5),
                'directness': 1 - profile.get('hedging_ratio', 0.1) * 5
            }
            
            # Normalize all values to 0-1 range
            for key, value in fingerprint.items():
                fingerprint[key] = max(0.0, min(1.0, value))
            
            return fingerprint
            
        except Exception as e:
            logger.error("Error generating comprehensive fingerprint", error=str(e))
            return {key: 0.5 for key in [
                'syntactic_complexity', 'coordination_preference', 'subordination_preference',
                'active_voice_preference', 'question_directness', 'vocabulary_diversity',
                'word_sophistication', 'lexical_density', 'cohesion_strength',
                'transition_usage', 'mitigation_tendency', 'certainty_level',
                'politeness_complexity', 'formality', 'enthusiasm', 'directness'
            ]}
    
    def generate_indistinguishable_prompt(self, profile: Dict[str, Any]) -> str:
        """
        Generate extremely detailed prompt for indistinguishable writing style matching.
        This is the key to making responses sound exactly like the user.
        """
        
        fingerprint = profile.get('comprehensive_fingerprint', {})
        
        # Syntactic style instructions
        syntax_instructions = self._generate_syntax_instructions(profile, fingerprint)
        
        # Lexical style instructions
        lexical_instructions = self._generate_lexical_instructions(profile, fingerprint)
        
        # Pragmatic style instructions
        pragmatic_instructions = self._generate_pragmatic_instructions(profile, fingerprint)
        
        # Cohesion and flow instructions
        cohesion_instructions = self._generate_cohesion_instructions(profile, fingerprint)
        
        # Specific patterns to replicate
        pattern_instructions = self._generate_pattern_instructions(profile)
        
        prompt = f"""You must write in this person's exact linguistic style. This is critical for authenticity.

SYNTACTIC PATTERNS (CRITICAL):
{syntax_instructions}

LEXICAL SOPHISTICATION:
{lexical_instructions}

PRAGMATIC STYLE (HOW THEY ACCOMPLISH SOCIAL FUNCTIONS):
{pragmatic_instructions}

TEXT COHESION AND FLOW:
{cohesion_instructions}

SPECIFIC PATTERNS TO REPLICATE:
{pattern_instructions}

ABSOLUTE REQUIREMENTS:
- Match ALL linguistic fingerprint dimensions exactly
- Use their exact greeting and closing patterns
- Replicate their sentence complexity and structure preferences
- Mirror their vocabulary sophistication level precisely
- Match their politeness and mitigation strategies
- Use their specific discourse markers and transitions
- Maintain their characteristic certainty/uncertainty patterns

The response must be indistinguishable from their actual writing. Every linguistic choice should reflect their documented patterns."""

        return prompt

    def _generate_syntax_instructions(self, profile: Dict[str, Any], fingerprint: Dict[str, float]) -> str:
        """Generate syntactic style instructions."""
        
        syntactic = profile.get('syntactic_patterns', {})
        dependency = profile.get('dependency_patterns', {})
        
        # Sentence complexity guidance
        complexity = fingerprint.get('syntactic_complexity', 0.5)
        if complexity > 0.7:
            complexity_desc = "Use complex sentences with multiple clauses. Prefer sophisticated sentence structures with subordination."
        elif complexity > 0.4:
            complexity_desc = "Use moderately complex sentences. Mix simple and compound structures."
        else:
            complexity_desc = "Prefer simple, direct sentences. Avoid overly complex constructions."
        
        # Coordination vs subordination
        coord_pref = fingerprint.get('coordination_preference', 0.3)
        subord_pref = fingerprint.get('subordination_preference', 0.3)
        
        if coord_pref > subord_pref:
            structure_pref = "Prefer coordinated structures (and, but, or) over subordinated ones."
        elif subord_pref > coord_pref:
            structure_pref = "Prefer subordinated structures (because, although, when) over coordination."
        else:
            structure_pref = "Balance coordinated and subordinated sentence structures."
        
        # Voice preference
        active_pref = fingerprint.get('active_voice_preference', 0.8)
        if active_pref > 0.7:
            voice_desc = "Strongly prefer active voice. Avoid passive constructions."
        elif active_pref > 0.4:
            voice_desc = "Generally use active voice, occasional passive is acceptable."
        else:
            voice_desc = "Use both active and passive voice. More formal, academic style."
        
        # Average sentence length
        avg_length = profile.get('avg_sentence_length', 15)
        length_desc = f"Target average sentence length: {avg_length:.1f} words."
        
        return f"""
- {complexity_desc}
- {structure_pref}
- {voice_desc}
- {length_desc}
- Sentence type distribution: {syntactic.get('sentence_type_distribution', {})}
"""

    def _generate_lexical_instructions(self, profile: Dict[str, Any], fingerprint: Dict[str, float]) -> str:
        """Generate lexical sophistication instructions."""
        
        lexical = profile.get('lexical_sophistication', {})
        vocab_level = profile.get('vocabulary_level', 'medium')
        
        # Vocabulary diversity
        diversity = fingerprint.get('vocabulary_diversity', 0.5)
        if diversity > 0.7:
            diversity_desc = "Use highly diverse vocabulary. Avoid repetition, employ synonyms."
        elif diversity > 0.4:
            diversity_desc = "Use moderately diverse vocabulary with some repetition for emphasis."
        else:
            diversity_desc = "Use focused vocabulary with strategic repetition of key terms."
        
        # Word sophistication
        sophistication = fingerprint.get('word_sophistication', 0.5)
        if sophistication > 0.7:
            word_desc = "Use sophisticated, longer words when appropriate. Academic/professional vocabulary."
        elif sophistication > 0.4:
            word_desc = "Use moderate vocabulary complexity. Balance simple and sophisticated terms."
        else:
            word_desc = "Prefer simple, clear words. Avoid unnecessarily complex vocabulary."
        
        # Lexical density
        density = fingerprint.get('lexical_density', 0.5)
        density_desc = f"Content word ratio: {density:.2f} (use {'more' if density > 0.6 else 'fewer'} content words relative to function words)"
        
        return f"""
- Vocabulary level: {vocab_level}
- {diversity_desc}
- {word_desc}
- {density_desc}
- Average word length: {lexical.get('avg_word_length', 5):.1f} characters
- Word repetition patterns: {lexical.get('hapax_legomena_ratio', 0.3):.2f} unique usage ratio
"""

    def _generate_pragmatic_instructions(self, profile: Dict[str, Any], fingerprint: Dict[str, float]) -> str:
        """Generate pragmatic competence instructions."""
        
        pragmatic = profile.get('pragmatic_competence', {})
        
        # Mitigation tendency
        mitigation = fingerprint.get('mitigation_tendency', 0.3)
        if mitigation > 0.6:
            mitigation_desc = "Use extensive hedging and softening language (I think, perhaps, might, seems like)."
        elif mitigation > 0.3:
            mitigation_desc = "Use moderate hedging when appropriate. Balance directness with politeness."
        else:
            mitigation_desc = "Be direct and confident. Minimal hedging or softening language."
        
        # Certainty patterns
        certainty = fingerprint.get('certainty_level', 0.5)
        if certainty > 0.7:
            certainty_desc = "Express high certainty. Use definitive language (definitely, clearly, obviously)."
        elif certainty > 0.3:
            certainty_desc = "Balance certainty and uncertainty based on context."
        else:
            certainty_desc = "Express uncertainty frequently (maybe, possibly, might be)."
        
        # Speech act patterns
        speech_acts = pragmatic.get('speech_act_distribution', {})
        
        # Politeness complexity
        politeness_complexity = fingerprint.get('politeness_complexity', 0.3)
        if politeness_complexity > 0.6:
            politeness_desc = "Use complex politeness strategies with extensive courtesy markers."
        elif politeness_complexity > 0.3:
            politeness_desc = "Use standard politeness markers appropriately."
        else:
            politeness_desc = "Use minimal politeness markers. More direct communication style."
        
        return f"""
- {mitigation_desc}
- {certainty_desc}
- {politeness_desc}
- Speech act preferences: Requests ({speech_acts.get('requests', 0)}), Thanks ({speech_acts.get('thanks', 0)}), Apologies ({speech_acts.get('apologies', 0)})
- Intensifier usage: {pragmatic.get('intensifier_density', 0.1):.3f} per word
"""

    def _generate_cohesion_instructions(self, profile: Dict[str, Any], fingerprint: Dict[str, float]) -> str:
        """Generate text cohesion and flow instructions."""
        
        cohesion = profile.get('cohesion_patterns', {})
        
        # Transition usage
        transition_density = fingerprint.get('transition_usage', 0.1)
        if transition_density > 0.3:
            transition_desc = "Use frequent transition words and phrases to connect ideas."
        elif transition_density > 0.1:
            transition_desc = "Use moderate transitions between ideas."
        else:
            transition_desc = "Use minimal transitions. Let ideas flow naturally."
        
        # Reference patterns
        ref_patterns = cohesion.get('reference_patterns', {})
        
        # Lexical repetition
        repetition = cohesion.get('lexical_repetition_ratio', 0.3)
        if repetition > 0.5:
            repetition_desc = "Use strategic word repetition for emphasis and coherence."
        elif repetition > 0.3:
            repetition_desc = "Use moderate lexical repetition."
        else:
            repetition_desc = "Minimize word repetition. Use varied vocabulary."
        
        return f"""
- {transition_desc}
- {repetition_desc}
- Pronoun usage: {ref_patterns.get('personal_pronouns', 0)} personal pronouns typically
- Demonstrative usage: {ref_patterns.get('demonstratives', 0)} this/that/these/those typically
- Transition patterns: {cohesion.get('transition_usage', {})}
"""

    def _generate_pattern_instructions(self, profile: Dict[str, Any]) -> str:
        """Generate specific pattern replication instructions."""
        
        # Greeting and closing patterns
        greetings = profile.get('common_greetings', ['hi'])
        closings = profile.get('common_closings', ['thanks'])
        
        # Common phrases
        phrases = profile.get('common_phrases', [])
        
        # Temporal patterns
        temporal = profile.get('temporal_deixis', {})
        temporal_orientation = temporal.get('temporal_orientation', {})
        
        return f"""
- Always use these greeting styles: {', '.join(greetings[:3])}
- Always use these closing styles: {', '.join(closings[:3])}
- Incorporate these characteristic phrases naturally: {', '.join(phrases[:5])}
- Temporal reference preference: Past ({temporal_orientation.get('past_reference', 0.3):.2f}), Present ({temporal_orientation.get('present_reference', 0.3):.2f}), Future ({temporal_orientation.get('future_reference', 0.3):.2f})
- Formality level: {profile.get('formality_score', 0.5):.2f}/1.0
- Enthusiasm level: {profile.get('enthusiasm_score', 0.5):.2f}/1.0
- Hedging frequency: {profile.get('hedging_ratio', 0.1):.3f} per word
"""
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove email headers and signatures
        text = re.sub(r'^(From:|To:|Subject:|Date:).*$', '', text, flags=re.MULTILINE)
        # Remove quoted text (lines starting with >)
        text = re.sub(r'^>.*$', '', text, flags=re.MULTILINE)
        return text.strip()
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK if available, otherwise regex."""
        if NLTK_AVAILABLE:
            try:
                return sent_tokenize(text)
            except:
                pass
        
        # Fallback to regex
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _tokenize_words(self, text: str) -> List[str]:
        """Tokenize words using NLTK if available, otherwise simple split."""
        if NLTK_AVAILABLE:
            try:
                return word_tokenize(text.lower())
            except:
                pass
        
        # Fallback to simple tokenization
        return re.findall(r'\b\w+\b', text.lower())
    
    def _analyze_basic_metrics(self, sentences: List[str], paragraphs: List[str], words: List[str]) -> Dict[str, Any]:
        """Calculate basic text metrics."""
        word_count = len(words)
        sentence_count = len(sentences)
        paragraph_count = len(paragraphs)
        
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        avg_paragraph_length = sentence_count / paragraph_count if paragraph_count > 0 else 0
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count,
            'avg_sentence_length': avg_sentence_length,
            'avg_paragraph_length': avg_paragraph_length
        }
    
    def _analyze_style_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze formality, enthusiasm, and other style patterns."""
        text_lower = text.lower()
        
        # Formality analysis
        formality_score = self._calculate_formality_score(text_lower)
        
        # Enthusiasm analysis
        enthusiasm_score = self._calculate_enthusiasm_score(text)
        
        # Politeness analysis
        politeness_score = self._calculate_politeness_score(text_lower)
        
        # Hedging analysis
        hedging_ratio = self._calculate_hedging_ratio(text_lower)
        
        # Extract patterns
        greetings = self._extract_greetings(text_lower)
        closings = self._extract_closings(text_lower)
        common_phrases = self._extract_common_phrases(text_lower)
        
        return {
            'formality_score': formality_score,
            'enthusiasm_score': enthusiasm_score,
            'politeness_score': politeness_score,
            'hedging_ratio': hedging_ratio,
            'common_greetings': greetings,
            'common_closings': closings,
            'common_phrases': common_phrases
        }
    
    def _analyze_linguistic_features(self, text: str, words: List[str]) -> Dict[str, Any]:
        """Analyze linguistic features and vocabulary."""
        # Vocabulary analysis
        content_words = [w for w in words if w.isalpha() and w not in self.stop_words]
        vocabulary_level = self._assess_vocabulary_level(content_words)
        
        # Discourse marker analysis
        discourse_metrics = self._analyze_discourse_markers(text.lower())
        
        return {
            'vocabulary_level': vocabulary_level,
            'lexical_diversity': len(set(content_words)) / len(content_words) if content_words else 0,
            'content_word_ratio': len(content_words) / len(words) if words else 0,
            **discourse_metrics
        }
    
    def _analyze_readability(self, text: str) -> Dict[str, Any]:
        """Analyze readability metrics using textstat."""
        try:
            if TEXTSTAT_AVAILABLE and textstat:
                readability_metrics = {}
                
                # Use getattr to safely access methods
                flesch_ease_func = getattr(textstat, 'flesch_reading_ease', None)
                if flesch_ease_func:
                    readability_metrics['flesch_reading_ease'] = flesch_ease_func(text)
                
                flesch_grade_func = getattr(textstat, 'flesch_kincaid_grade', None)
                if flesch_grade_func:
                    readability_metrics['flesch_kincaid_grade'] = flesch_grade_func(text)
                
                ari_func = getattr(textstat, 'automated_readability_index', None)
                if ari_func:
                    readability_metrics['automated_readability_index'] = ari_func(text)
                
                reading_time_func = getattr(textstat, 'reading_time', None)
                if reading_time_func:
                    readability_metrics['reading_time'] = reading_time_func(text)
                
                return readability_metrics
            else:
                return {}
        except Exception:
            return {}
    
    def _calculate_formality_score(self, text_lower: str) -> float:
        """Calculate formality score (0 = casual, 1 = formal)."""
        formal_count = sum(1 for indicator in self.formal_indicators 
                          if indicator in text_lower)
        casual_count = sum(1 for indicator in self.casual_indicators 
                          if indicator in text_lower)
        
        total_indicators = formal_count + casual_count
        if total_indicators == 0:
            return 0.5  # Neutral
        
        return formal_count / total_indicators
    
    def _calculate_enthusiasm_score(self, text: str) -> float:
        """Calculate enthusiasm score based on punctuation and word choice."""
        # Count exclamation marks
        exclamation_count = text.count('!')
        
        # Count enthusiastic words/phrases
        enthusiastic_words = [
            'amazing', 'awesome', 'fantastic', 'great', 'excellent',
            'wonderful', 'brilliant', 'perfect', 'love', 'excited'
        ]
        
        text_lower = text.lower()
        enthusiastic_count = sum(1 for word in enthusiastic_words 
                                if word in text_lower)
        
        # Normalize based on text length
        word_count = len(text.split())
        if word_count == 0:
            return 0.5
        
        # Simple scoring: exclamations + enthusiastic words / total words
        enthusiasm_ratio = (exclamation_count + enthusiastic_count) / word_count
        
        # Cap at 1.0 and scale
        return min(1.0, enthusiasm_ratio * 10)
    
    def _calculate_politeness_score(self, text_lower: str) -> float:
        """Calculate politeness based on courtesy markers."""
        total_score = 0
        total_markers = 0
        
        for level, markers in self.politeness_markers.items():
            count = sum(1 for marker in markers if marker in text_lower)
            if level == 'high':
                total_score += count * 1.0
            elif level == 'medium':
                total_score += count * 0.6
            else:  # low
                total_score += count * 0.2
            total_markers += count
        
        return total_score / max(1, total_markers)
    
    def _calculate_hedging_ratio(self, text_lower: str) -> float:
        """Calculate hedging ratio based on hedging patterns."""
        hedging_count = sum(1 for hedge in self.hedging_patterns 
                           if hedge in text_lower)
        word_count = len(text_lower.split())
        return hedging_count / max(1, word_count)
    
    def _extract_greetings(self, text_lower: str) -> List[str]:
        """Extract greeting patterns from email."""
        found_greetings = []
        
        # Look for greetings in the first few sentences
        first_part = ' '.join(text_lower.split()[:50])  # First 50 words
        
        for greeting in self.common_greetings:
            if greeting in first_part:
                found_greetings.append(greeting)
        
        return found_greetings[:3]  # Keep top 3
    
    def _extract_closings(self, text_lower: str) -> List[str]:
        """Extract closing patterns from email."""
        found_closings = []
        
        # Look for closings in the last few sentences
        last_part = ' '.join(text_lower.split()[-50:])  # Last 50 words
        
        for closing in self.common_closings:
            if closing in last_part:
                found_closings.append(closing)
        
        return found_closings[:3]  # Keep top 3
    
    def _extract_common_phrases(self, text_lower: str) -> List[str]:
        """Extract common phrases or expressions."""
        words = text_lower.split()
        
        # Look for common bigrams
        phrases = []
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i + 1]}"
            if len(bigram) > 6:  # Avoid very short phrases
                phrases.append(bigram)
        
        # Count frequency and return most common
        phrase_counts = Counter(phrases)
        return [phrase for phrase, count in phrase_counts.most_common(5)]
    
    def _assess_vocabulary_level(self, content_words: List[str]) -> str:
        """Assess vocabulary complexity level."""
        if not content_words:
            return 'medium'
        
        # Simple heuristic based on word length
        avg_word_length = statistics.mean(len(word) for word in content_words)
        
        if avg_word_length < 4:
            return 'simple'
        elif avg_word_length > 6:
            return 'advanced'
        else:
            return 'medium'
    
    def _analyze_discourse_markers(self, text_lower: str) -> Dict[str, Any]:
        """Analyze discourse markers and text organization."""
        discourse_usage = {}
        for category, markers in self.discourse_markers.items():
            usage_count = sum(1 for marker in markers if marker in text_lower)
            discourse_usage[f'{category}_markers'] = usage_count
        
        total_markers = sum(discourse_usage.values())
        word_count = len(text_lower.split())
        
        return {
            **discourse_usage,
            'total_discourse_markers': total_markers,
            'discourse_density': total_markers / max(1, word_count)
        }
    
    def _merge_common_items(self, existing_items: List[str], new_items: List[str]) -> List[str]:
        """Merge two lists keeping most frequent items."""
        all_items = existing_items + new_items
        item_counts = Counter(all_items)
        return [item for item, count in item_counts.most_common(5)]
    
    def _merge_vocabulary_level(self, existing_level: str, new_level: str) -> str:
        """Determine vocabulary level from existing and new analysis."""
        if existing_level == new_level:
            return existing_level
        
        # Priority order: advanced > medium > simple
        level_priority = {'advanced': 3, 'medium': 2, 'simple': 1}
        
        if level_priority.get(new_level, 2) > level_priority.get(existing_level, 2):
            return new_level
        
        return existing_level
    
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