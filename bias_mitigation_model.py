import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import defaultdict
import re
from typing import Dict, List, Tuple
import logging

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiasNormalizationEngine:
    """
    Advanced bias mitigation engine that normalizes ideological biases
    across five constituencies in Indian news media.
    
    Based on the research paper: "Analysis of Media Bias in Policy Discourse in India"
    """
    
    def __init__(self):
        """Initialize the bias normalization engine with constituency definitions"""
        
        # Initialize sentiment analyzer
        self.sia = SentimentIntensityAnalyzer()
        
        # Define the five constituencies with their characteristic keywords
        # Based on paper's methodology (Table 2 and Section 3.3)
        self.constituencies = {
            'poor': {
                'keywords': [
                    'poverty', 'slum', 'hunger', 'unemployment', 'daily wage',
                    'migrant worker', 'laborer', 'labourer', 'homeless', 'BPL',
                    'ration', 'MGNREGA', 'below poverty line', 'informal labour',
                    'marginalized', 'deprived', 'underprivileged', 'poor families'
                ],
                'weight': 1.0,
                'ideal_coverage': 0.20  # Target 20% coverage
            },
            'middle_class': {
                'keywords': [
                    'salary', 'EMI', 'tax', 'savings', 'education', 'urban',
                    'professional', 'IT', 'bank account', 'credit card', 'salaried',
                    'income tax', 'housing loan', 'middle income', 'white collar'
                ],
                'weight': 1.0,
                'ideal_coverage': 0.20  # Target 20% coverage
            },
            'corporate': {
                'keywords': [
                    'business', 'industry', 'company', 'profit', 'investment',
                    'market', 'stock', 'CEO', 'enterprise', 'revenue', 'sensex',
                    'nifty', 'corporate tax', 'quarterly results', 'BSE', 'NSE'
                ],
                'weight': 1.0,
                'ideal_coverage': 0.20  # Target 20% coverage
            },
            'informal_sector': {
                'keywords': [
                    'vendor', 'small business', 'trader', 'unorganized',
                    'self-employed', 'shopkeeper', 'hawker', 'street vendor',
                    'MSME', 'small trader', 'informal economy', 'unregistered'
                ],
                'weight': 1.0,
                'ideal_coverage': 0.20  # Target 20% coverage
            },
            'government': {
                'keywords': [
                    'policy', 'minister', 'parliament', 'legislation', 'scheme',
                    'government', 'official', 'bureaucrat', 'administration',
                    'Modi', 'BJP', 'Congress', 'opposition', 'PM', 'ministry'
                ],
                'weight': 1.0,
                'ideal_coverage': 0.20  # Target 20% coverage
            }
        }
        
        # Known source biases from paper (Table 2 and Section 5)
        self.source_bias_profiles = {
            'Times of India': {
                'political': 'pro-BJP',
                'ideological': ['pro-corporate', 'pro-government'],
                'bias_correction_factor': 1.2
            },
            'The Hindu': {
                'political': 'pro-INC',
                'ideological': ['anti-corporate', 'anti-government', 'pro-poor'],
                'bias_correction_factor': 0.9
            },
            'Indian Express': {
                'political': 'pro-INC',
                'ideological': ['neutral'],
                'bias_correction_factor': 0.8
            },
            'Hindustan Times': {
                'political': 'anti-INC',
                'ideological': ['neutral'],
                'bias_correction_factor': 0.85
            },
            'Deccan Herald': {
                'political': 'anti-both',
                'ideological': ['pro-poor', 'pro-middle_class', 'pro-informal_sector'],
                'bias_correction_factor': 0.8
            }
        }
    
    def calculate_constituency_alignment(self, text: str) -> Dict[str, float]:
        """
        Calculate constituency alignment scores for a given text.
        
        Based on equation in Section 3.3 of the paper:
        alignment_score = (keyword_count * weight) / text_length
        
        Args:
            text: Article text to analyze
            
        Returns:
            Dictionary mapping constituencies to alignment scores
        """
        if not text or pd.isna(text):
            return {const: 0.0 for const in self.constituencies}
        
        text_lower = str(text).lower()
        text_words = text_lower.split()
        text_length = len(text_words)
        
        alignment_scores = {}
        
        for constituency, config in self.constituencies.items():
            # Count occurrences of constituency keywords
            keyword_count = sum(
                text_lower.count(keyword.lower())
                for keyword in config['keywords']
            )
            
            # Normalize by text length (as per paper's methodology)
            if text_length > 0:
                alignment_scores[constituency] = (
                    keyword_count * config['weight'] / text_length
                )
            else:
                alignment_scores[constituency] = 0.0
        
        return alignment_scores
    
    def calculate_sentiment(self, text: str) -> Dict[str, float]:
        """
        Calculate sentiment scores using VADER and TextBlob.
        
        Based on Section 3.2 of the paper.
        
        Args:
            text: Article text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not text or pd.isna(text):
            return {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'textblob_polarity': 0.0
            }
        
        # VADER sentiment
        vader_scores = self.sia.polarity_scores(str(text))
        
        # TextBlob sentiment
        try:
            blob = TextBlob(str(text))
            vader_scores['textblob_polarity'] = blob.sentiment.polarity
            vader_scores['textblob_subjectivity'] = blob.sentiment.subjectivity
        except:
            vader_scores['textblob_polarity'] = 0.0
            vader_scores['textblob_subjectivity'] = 0.0
        
        return vader_scores
    
    def calculate_frame_bias(self, articles: List[Dict]) -> Dict[str, Dict[str, float]]:
        """
        Calculate frame bias scores for each source.
        
        Based on equations (1), (2), and (3) from Section 3.3:
        - Calculate sentiment offset from mean
        - Weight by constituency alignment
        - Aggregate across articles
        
        Args:
            articles: List of article dictionaries with 'source', 'content', etc.
            
        Returns:
            Dictionary mapping sources to their frame bias scores
        """
        source_frame_scores = defaultdict(lambda: defaultdict(list))
        
        # First pass: calculate mean sentiment per constituency
        constituency_sentiments = defaultdict(list)
        
        for article in articles:
            if not article.get('content'):
                continue
            
            sentiment = self.calculate_sentiment(article['content'])
            alignment = self.calculate_constituency_alignment(article['content'])
            
            for const, align_score in alignment.items():
                if align_score > 0:
                    constituency_sentiments[const].append(sentiment['compound'])
        
        # Calculate mean sentiments
        mean_sentiments = {
            const: np.mean(sents) if sents else 0.0
            for const, sents in constituency_sentiments.items()
        }
        
        # Second pass: calculate frame bias with sentiment offset
        for article in articles:
            source = article.get('source', 'Unknown')
            content = article.get('content', '')
            
            if not content:
                continue
            
            sentiment = self.calculate_sentiment(content)
            alignment = self.calculate_constituency_alignment(content)
            
            for const, align_score in alignment.items():
                if align_score > 0:
                    # Calculate sentiment offset from mean (Equation 2)
                    sentiment_offset = sentiment['compound'] - mean_sentiments.get(const, 0.0)
                    
                    # Calculate frame score (combining alignment and sentiment)
                    frame_score = align_score * sentiment_offset
                    
                    # Determine pro/anti stance
                    if sentiment['compound'] > 0.1:
                        source_frame_scores[source][f'pro_{const}'].append(
                            align_score * abs(sentiment['compound'])
                        )
                    elif sentiment['compound'] < -0.1:
                        source_frame_scores[source][f'anti_{const}'].append(
                            align_score * abs(sentiment['compound'])
                        )
                    else:
                        source_frame_scores[source][f'neutral_{const}'].append(align_score)
        
        # Average the scores
        result = {}
        for source, frames in source_frame_scores.items():
            result[source] = {
                frame: np.mean(scores) if scores else 0.0
                for frame, scores in frames.items()
            }
        
        return result
    
    def detect_coverage_bias(self, articles: List[Dict]) -> Dict[str, Dict[str, float]]:
        """
        Detect coverage bias: how much each source covers each constituency.
        
        Based on Section 4.1 and equation (4) from the paper.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Dictionary mapping sources to constituency coverage percentages
        """
        source_coverage = defaultdict(lambda: defaultdict(float))
        source_word_counts = defaultdict(float)
        
        for article in articles:
            source = article.get('source', 'Unknown')
            content = article.get('content', '')
            
            if not content:
                continue
            
            word_count = len(str(content).split())
            source_word_counts[source] += word_count
            
            alignment = self.calculate_constituency_alignment(content)
            
            for const, score in alignment.items():
                source_coverage[source][const] += score * word_count
        
        # Normalize by total word count
        result = {}
        for source, constituencies in source_coverage.items():
            total_words = source_word_counts[source]
            if total_words > 0:
                result[source] = {
                    const: (score / total_words)
                    for const, score in constituencies.items()
                }
            else:
                result[source] = {const: 0.0 for const in self.constituencies}
        
        return result
    
    def calculate_bias_metrics(self, articles: List[Dict]) -> Dict:
        """
        Calculate comprehensive bias metrics for the article set.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Dictionary containing all bias metrics
        """
        logger.info(f"Calculating bias metrics for {len(articles)} articles...")
        
        # Calculate frame bias
        frame_bias = self.calculate_frame_bias(articles)
        
        # Calculate coverage bias
        coverage_bias = self.detect_coverage_bias(articles)
        
        # Calculate overall constituency representation
        overall_coverage = defaultdict(float)
        total_weight = 0
        
        for source, coverages in coverage_bias.items():
            for const, coverage in coverages.items():
                overall_coverage[const] += coverage
                total_weight += 1
        
        if total_weight > 0:
            overall_coverage = {
                const: cov / total_weight
                for const, cov in overall_coverage.items()
            }
        
        # Calculate bias severity (deviation from ideal coverage)
        bias_severity = {}
        for const, actual_coverage in overall_coverage.items():
            ideal = self.constituencies[const]['ideal_coverage']
            bias_severity[const] = abs(actual_coverage - ideal)
        
        return {
            'frame_bias': frame_bias,
            'coverage_bias': coverage_bias,
            'overall_coverage': overall_coverage,
            'bias_severity': bias_severity,
            'total_articles': len(articles)
        }
    
    def normalize_and_mitigate_bias(self, articles: List[Dict]) -> Dict:
        """
        Core bias mitigation algorithm that normalizes ideological biases.
        
        Strategy (based on paper's findings in Sections 4.2 and 7):
        1. Identify over-represented and under-represented constituencies
        2. Apply weighted averaging favoring under-represented groups
        3. Adjust for known source biases
        4. Generate balanced summary
        
        Args:
            articles: List of article dictionaries from multiple sources
            
        Returns:
            Dictionary containing mitigated content and metrics
        """
        logger.info("Starting bias normalization process...")
        
        if not articles:
            return {
                'mitigated_summary': "No articles available for analysis.",
                'bias_metrics': {},
                'normalization_applied': False
            }
        
        # Step 1: Calculate bias metrics
        bias_metrics = self.calculate_bias_metrics(articles)
        
        # Step 2: Identify under-represented constituencies
        overall_coverage = bias_metrics['overall_coverage']
        under_represented = []
        over_represented = []
        
        for const, coverage in overall_coverage.items():
            ideal = self.constituencies[const]['ideal_coverage']
            if coverage < ideal * 0.5:  # Less than 50% of ideal coverage
                under_represented.append(const)
            elif coverage > ideal * 1.5:  # More than 150% of ideal coverage
                over_represented.append(const)
        
        logger.info(f"Under-represented: {under_represented}")
        logger.info(f"Over-represented: {over_represented}")
        
        # Step 3: Calculate article weights based on constituency representation
        article_weights = []
        
        for article in articles:
            content = article.get('content', '')
            if not content:
                article_weights.append(0.0)
                continue
            
            alignment = self.calculate_constituency_alignment(content)
            
            # Base weight
            weight = 1.0
            
            # Boost articles covering under-represented constituencies
            for const in under_represented:
                if alignment.get(const, 0) > 0:
                    weight *= (2.0 + alignment[const] * 10)  # Significant boost
            
            # Reduce weight for over-represented constituencies
            for const in over_represented:
                if alignment.get(const, 0) > 0:
                    weight *= (0.5 - alignment[const] * 0.3)  # Moderate reduction
            
            # Adjust for source bias (if known)
            source = article.get('source', '')
            if source in self.source_bias_profiles:
                correction = self.source_bias_profiles[source]['bias_correction_factor']
                weight *= correction
            
            # Ensure positive weight
            weight = max(0.1, weight)
            article_weights.append(weight)
        
        # Step 4: Generate balanced summary using weighted extraction
        summary_sentences = []
        constituency_sentences = defaultdict(list)
        
        for article, weight in zip(articles, article_weights):
            content = article.get('content', '')
            source = article.get('source', 'Unknown')
            
            if not content:
                continue
            
            # Split into sentences
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            if not sentences:
                continue
            
            # Analyze each sentence for constituency alignment
            for sentence in sentences[:5]:  # Limit to first 5 sentences per article
                alignment = self.calculate_constituency_alignment(sentence)
                max_const = max(alignment, key=alignment.get)
                
                if alignment[max_const] > 0:
                    constituency_sentences[max_const].append({
                        'sentence': sentence,
                        'weight': weight,
                        'source': source,
                        'alignment': alignment[max_const]
                    })
        
        # Step 5: Select balanced sentences from each constituency
        target_sentences_per_const = 3
        
        for const in self.constituencies:
            sentences = constituency_sentences.get(const, [])
            
            if not sentences:
                # If no coverage, add a note about lack of coverage
                if const in under_represented:
                    summary_sentences.append({
                        'text': f"[NOTE: Limited coverage of {const.replace('_', ' ')} perspectives in source articles]",
                        'constituency': const,
                        'synthetic': True
                    })
                continue
            
            # Sort by weight and select top sentences
            sentences.sort(key=lambda x: x['weight'] * x['alignment'], reverse=True)
            
            # Take more sentences for under-represented constituencies
            num_sentences = target_sentences_per_const
            if const in under_represented:
                num_sentences = min(len(sentences), target_sentences_per_const + 2)
            elif const in over_represented:
                num_sentences = min(len(sentences), target_sentences_per_const - 1)
            
            for sent_data in sentences[:num_sentences]:
                summary_sentences.append({
                    'text': sent_data['sentence'],
                    'constituency': const,
                    'source': sent_data['source'],
                    'weight': sent_data['weight'],
                    'synthetic': False
                })
        
        # Step 6: Order sentences for coherent summary
        # Group by constituency and create structured summary
        structured_summary = []
        
        structured_summary.append("=== BIAS-MITIGATED NEWS SUMMARY ===\n")
        
        for const in ['poor', 'informal_sector', 'middle_class', 'corporate', 'government']:
            const_sents = [s for s in summary_sentences if s['constituency'] == const]
            
            if const_sents:
                structured_summary.append(f"\n[{const.replace('_', ' ').title()} Perspective]")
                
                for sent in const_sents:
                    if sent['synthetic']:
                        structured_summary.append(f"• {sent['text']}")
                    else:
                        structured_summary.append(f"• {sent['text']} (Source: {sent['source']})")
        
        # Step 7: Add bias analysis summary
        structured_summary.append("\n\n=== BIAS ANALYSIS ===")
        structured_summary.append(f"\nTotal articles analyzed: {len(articles)}")
        structured_summary.append(f"Sources: {len(set(a.get('source', 'Unknown') for a in articles))}")
        
        structured_summary.append("\n[Constituency Coverage]")
        for const, coverage in overall_coverage.items():
            ideal = self.constituencies[const]['ideal_coverage']
            status = "✓" if abs(coverage - ideal) < 0.05 else "⚠"
            structured_summary.append(
                f"{status} {const.replace('_', ' ').title()}: "
                f"{coverage*100:.1f}% (ideal: {ideal*100:.0f}%)"
            )
        
        if under_represented:
            structured_summary.append(f"\n[Under-represented]: {', '.join(c.replace('_', ' ') for c in under_represented)}")
        
        if over_represented:
            structured_summary.append(f"[Over-represented]: {', '.join(c.replace('_', ' ') for c in over_represented)}")
        
        mitigated_summary = '\n'.join(structured_summary)
        
        return {
            'mitigated_summary': mitigated_summary,
            'bias_metrics': bias_metrics,
            'normalization_applied': True,
            'under_represented': under_represented,
            'over_represented': over_represented,
            'article_count': len(articles),
            'constituency_balance': overall_coverage
        }
    
    def generate_humanized_summary(self, articles: List[Dict], max_words: int = 300) -> str:
        """
        Generate a human-friendly, bias-mitigated summary.
        
        Args:
            articles: List of article dictionaries
            max_words: Maximum words in summary
            
        Returns:
            Human-readable summary string
        """
        result = self.normalize_and_mitigate_bias(articles)
        
        if not result['normalization_applied']:
            return result['mitigated_summary']
        
        # Extract key sentences from each constituency
        summary_parts = []
        
        # Parse the structured summary
        full_summary = result['mitigated_summary']
        
        # Extract sentences by constituency
        lines = full_summary.split('\n')
        current_const = None
        const_content = defaultdict(list)
        
        for line in lines:
            if line.startswith('[') and 'Perspective]' in line:
                current_const = line.strip('[]').replace(' Perspective', '')
            elif line.startswith('•') and current_const:
                # Clean the sentence
                sentence = line.strip('• ').split('(Source:')[0].strip()
                if '[NOTE:' not in sentence and sentence:
                    const_content[current_const].append(sentence)
        
        # Build humanized summary
        summary_text = []
        
        # Opening statement
        summary_text.append(
            f"Based on analysis of {result['article_count']} articles from multiple sources, "
            "here is a balanced perspective on this issue:\n"
        )
        
        # Add content from each constituency
        for const, sentences in const_content.items():
            if sentences:
                # Take first sentence from each
                summary_text.append(sentences[0])
        
        # Add bias note if significant
        if result['under_represented']:
            summary_text.append(
                f"\nNote: Coverage of {', '.join(result['under_represented'])} "
                "perspectives was limited in the original sources."
            )
        
        # Join and limit length
        full_text = ' '.join(summary_text)
        words = full_text.split()
        
        if len(words) > max_words:
            full_text = ' '.join(words[:max_words]) + '...'
        
        return full_text


# Example usage and testing
if __name__ == "__main__":
    # Initialize the engine
    engine = BiasNormalizationEngine()
    
    # Sample test articles (simulating different sources and biases)
    test_articles = [
        {
            'source': 'Times of India',
            'content': '''The stock market surged today as corporate profits exceeded expectations. 
            Major companies reported strong quarterly results. Sensex gained 500 points on positive sentiment.
            Investors are optimistic about economic growth driven by industrial expansion.'''
        },
        {
            'source': 'The Hindu',
            'content': '''Daily wage workers continue to struggle with unemployment and poverty. 
            The situation in slum areas has worsened significantly. Many migrant workers lack access to basic ration.
            Below poverty line families are facing severe hunger due to economic policies.'''
        },
        {
            'source': 'Indian Express',
            'content': '''The government announced new policy measures in parliament today. 
            Ministers discussed implementation of welfare schemes. Opposition raised concerns about farmer distress.
            Small traders and vendors in the informal sector are worried about new regulations.'''
        },
        {
            'source': 'Deccan Herald',
            'content': '''Middle class families are struggling with EMI payments and rising living costs.
            Income tax burden has increased for salaried professionals. Housing loan interest rates have gone up.
            Urban professionals in IT sector are facing job insecurity and salary cuts.'''
        }
    ]
    
    # Test bias metrics calculation
    print("=" * 80)
    print("TESTING BIAS NORMALIZATION ENGINE")
    print("=" * 80)
    
    metrics = engine.calculate_bias_metrics(test_articles)
    print("\n[BIAS METRICS]")
    print(f"Overall Coverage: {metrics['overall_coverage']}")
    print(f"Bias Severity: {metrics['bias_severity']}")
    
    # Test normalization
    print("\n" + "=" * 80)
    print("[BIAS-MITIGATED OUTPUT]")
    print("=" * 80)
    result = engine.normalize_and_mitigate_bias(test_articles)
    print(result['mitigated_summary'])
    
    # Test humanized summary
    print("\n" + "=" * 80)
    print("[HUMANIZED SUMMARY]")
    print("=" * 80)
    humanized = engine.generate_humanized_summary(test_articles)
    print(humanized)
