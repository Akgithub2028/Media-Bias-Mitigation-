import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial.distance import euclidean
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class BiasAnalysisEDA:
    """
    Comprehensive EDA and preprocessing for bias analysis
    Based on research paper methodology
    """
    
    def __init__(self, articles_path=None):
        """
        Initialize with scraped articles
        """
        self.articles_df = None
        if articles_path:
            self.load_articles(articles_path)
        
        # Initialize sentiment analyzer
        self.sia = SentimentIntensityAnalyzer()
        
        # Define constituencies and their characteristics (from paper)
        self.constituencies = {
            'poor': {
                'keywords': ['poverty', 'slum', 'hunger', 'unemployment', 'daily wage',
                           'migrant worker', 'laborer', 'homeless', 'BPL', 'ration',
                           'MGNREGA', 'below poverty line', 'informal labour'],
                'color': '#e74c3c',
                'weight': 1.0
            },
            'middle_class': {
                'keywords': ['salary', 'EMI', 'tax', 'savings', 'education', 'urban',
                           'professional', 'IT', 'bank account', 'credit card',
                           'income tax', 'housing loan', 'salaried'],
                'color': '#3498db',
                'weight': 1.0
            },
            'corporate': {
                'keywords': ['business', 'industry', 'company', 'profit', 'investment',
                           'market', 'stock', 'CEO', 'enterprise', 'revenue',
                           'sensex', 'nifty', 'corporate tax'],
                'color': '#f39c12',
                'weight': 1.0
            },
            'informal_sector': {
                'keywords': ['vendor', 'small business', 'trader', 'unorganized',
                           'self-employed', 'shopkeeper', 'hawker', 'street vendor',
                           'MSME', 'small trader', 'informal economy'],
                'color': '#9b59b6',
                'weight': 1.0
            },
            'government': {
                'keywords': ['policy', 'minister', 'parliament', 'legislation', 'scheme',
                           'government', 'official', 'bureaucrat', 'administration',
                           'Modi', 'BJP', 'Congress', 'opposition'],
                'color': '#27ae60',
                'weight': 1.0
            }
        }
        
        # News source characteristics (from paper findings)
        self.source_profiles = {
            'Times of India': {'political': 'pro-BJP', 'ideological': 'pro-corporate'},
            'The Hindu': {'political': 'pro-INC', 'ideological': 'anti-corporate'},
            'Indian Express': {'political': 'pro-INC', 'ideological': 'neutral'},
            'Hindustan Times': {'political': 'anti-INC', 'ideological': 'neutral'},
            'Deccan Herald': {'political': 'anti-both', 'ideological': 'pro-poor'}
        }
    
    def load_articles(self, path):
        """Load scraped articles from JSON or CSV"""
        if path.endswith('.json'):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.articles_df = pd.DataFrame(data)
        else:
            self.articles_df = pd.read_csv(path)
        
        print(f"Loaded {len(self.articles_df)} articles")
        return self.articles_df
    
    def preprocess_text(self, text):
        """Preprocess text for analysis"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters but keep spaces
        import re
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_aspects_lda(self, articles, n_topics=15):
        """
        Extract aspects using LDA (paper's methodology)
        """
        # Preprocess articles
        processed = [self.preprocess_text(art) for art in articles]
        
        # Vectorize
        vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=2,
            stop_words='english',
            max_features=1000  # Limited for Colab
        )
        
        doc_term_matrix = vectorizer.fit_transform(processed)
        
        # LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=50  # Limited iterations for Colab
        )
        
        doc_topic_dist = lda.fit_transform(doc_term_matrix)
        
        # Get topic words
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'weight': topic[top_words_idx].tolist()
            })
        
        return doc_topic_dist, topics, lda, vectorizer
    
    def calculate_sentiment(self, text):
        """
        Calculate sentiment scores (VADER + TextBlob)
        """
        if pd.isna(text) or text == "":
            return {'compound': 0, 'positive': 0, 'negative': 0, 'neutral': 1}
        
        # VADER sentiment
        scores = self.sia.polarity_scores(text)
        
        # TextBlob sentiment
        try:
            blob = TextBlob(text)
            scores['textblob_polarity'] = blob.sentiment.polarity
            scores['textblob_subjectivity'] = blob.sentiment.subjectivity
        except:
            scores['textblob_polarity'] = 0
            scores['textblob_subjectivity'] = 0
        
        return scores
    
    def detect_constituency_alignment(self, text):
        """
        Detect constituency alignment in article (paper's methodology)
        """
        if pd.isna(text):
            return {const: 0 for const in self.constituencies}
        
        text_lower = str(text).lower()
        alignment_scores = {}
        
        for constituency, config in self.constituencies.items():
            # Count keyword occurrences
            keyword_count = sum(
                text_lower.count(keyword.lower())
                for keyword in config['keywords']
            )
            
            # Normalize by text length
            text_length = len(text_lower.split())
            if text_length > 0:
                alignment_scores[constituency] = (
                    keyword_count * config['weight'] / text_length
                )
            else:
                alignment_scores[constituency] = 0
        
        return alignment_scores
    
    def analyze_frame_bias(self, df):
        """
        Analyze framing bias for each news source
        """
        frame_analysis = []
        
        for source in df['source'].unique():
            source_articles = df[df['source'] == source]
            
            # Initialize frame scores
            frame_scores = {
                f'{stance}_{const}': []
                for const in self.constituencies
                for stance in ['pro', 'anti']
            }
            
            for _, article in source_articles.iterrows():
                if pd.isna(article['content']):
                    continue
                
                # Get sentiment
                sentiment = self.calculate_sentiment(article['content'])
                
                # Get constituency alignment
                alignment = self.detect_constituency_alignment(article['content'])
                
                # Calculate frame scores
                for constituency, alignment_score in alignment.items():
                    if sentiment['compound'] > 0.1:
                        frame_scores[f'pro_{constituency}'].append(
                            alignment_score * sentiment['compound']
                        )
                        frame_scores[f'anti_{constituency}'].append(0)
                    elif sentiment['compound'] < -0.1:
                        frame_scores[f'pro_{constituency}'].append(0)
                        frame_scores[f'anti_{constituency}'].append(
                            alignment_score * abs(sentiment['compound'])
                        )
                    else:
                        frame_scores[f'pro_{constituency}'].append(alignment_score * 0.5)
                        frame_scores[f'anti_{constituency}'].append(alignment_score * 0.5)
            
            # Average scores
            avg_scores = {
                frame: np.mean(scores) if scores else 0
                for frame, scores in frame_scores.items()
            }
            avg_scores['source'] = source
            frame_analysis.append(avg_scores)
        
        return pd.DataFrame(frame_analysis)
    
    def create_bias_visualizations(self, df):
        """
        Create comprehensive bias visualizations (following paper's style)
        """
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Constituency Coverage by Source (Bar Chart)
        ax1 = plt.subplot(2, 3, 1)
        constituency_coverage = []
        
        for source in df['source'].unique():
            source_articles = df[df['source'] == source]
            coverages = []
            
            for const in self.constituencies:
                const_scores = []
                for _, article in source_articles.iterrows():
                    if pd.notna(article['content']):
                        alignment = self.detect_constituency_alignment(article['content'])
                        const_scores.append(alignment.get(const, 0))
                coverages.append(np.mean(const_scores) if const_scores else 0)
            
            constituency_coverage.append(coverages)
        
        # Create grouped bar chart
        x = np.arange(len(self.constituencies))
        width = 0.15
        
        for i, source in enumerate(df['source'].unique()):
            ax1.bar(x + i*width, constituency_coverage[i], width, label=source)
        
        ax1.set_xlabel('Constituency')
        ax1.set_ylabel('Relative Coverage')
        ax1.set_title('Constituency Coverage by News Source')
        ax1.set_xticks(x + width * 2)
        ax1.set_xticklabels(list(self.constituencies.keys()))
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Sentiment Distribution by Source
        ax2 = plt.subplot(2, 3, 2)
        sentiment_data = []
        
        for source in df['source'].unique():
            source_articles = df[df['source'] == source]
            sentiments = []
            for _, article in source_articles.iterrows():
                if pd.notna(article['content']):
                    sent = self.calculate_sentiment(article['content'])
                    sentiments.append(sent['compound'])
            
            if sentiments:
                sentiment_data.append(sentiments)
        
        ax2.violinplot(sentiment_data, positions=range(len(df['source'].unique())),
                      widths=0.7, showmeans=True, showmedians=True)
        ax2.set_xticks(range(len(df['source'].unique())))
        ax2.set_xticklabels(df['source'].unique(), rotation=45)
        ax2.set_ylabel('Sentiment Score')
        ax2.set_title('Sentiment Distribution by News Source')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # 3. Frame Bias Heatmap
        ax3 = plt.subplot(2, 3, 3)
        frame_df = self.analyze_frame_bias(df)
        
        # Prepare data for heatmap
        frame_matrix = []
        for source in df['source'].unique():
            source_frames = frame_df[frame_df['source'] == source]
            if not source_frames.empty:
                row = []
                for const in self.constituencies:
                    pro_score = source_frames[f'pro_{const}'].values[0] if not source_frames.empty else 0
                    anti_score = source_frames[f'anti_{const}'].values[0] if not source_frames.empty else 0
                    row.append(pro_score - anti_score)  # Net bias
                frame_matrix.append(row)
        
        if frame_matrix:
            im = ax3.imshow(frame_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.05, vmax=0.05)
            ax3.set_xticks(range(len(self.constituencies)))
            ax3.set_xticklabels(list(self.constituencies.keys()), rotation=45)
            ax3.set_yticks(range(len(df['source'].unique())))
            ax3.set_yticklabels(df['source'].unique())
            ax3.set_title('Frame Bias Heatmap (Pro-Anti Score)')
            plt.colorbar(im, ax=ax3)
        
        # 4. Word Cloud for Poor Constituency
        ax4 = plt.subplot(2, 3, 4)
        poor_keywords = ' '.join(self.constituencies['poor']['keywords'] * 10)
        
        if poor_keywords:
            wordcloud = WordCloud(width=400, height=300, 
                                 background_color='white',
                                 colormap='Reds').generate(poor_keywords)
            ax4.imshow(wordcloud, interpolation='bilinear')
            ax4.set_title('Keywords: Poor Constituency')
            ax4.axis('off')
        
        # 5. Aspect Coverage Distribution (from paper's methodology)
        ax5 = plt.subplot(2, 3, 5)
        
        if 'event' in df.columns:
            event_counts = df['event'].value_counts()
            colors = ['#e74c3c', '#3498db', '#f39c12', '#27ae60', '#9b59b6']
            ax5.pie(event_counts.values, labels=event_counts.index, 
                   autopct='%1.1f%%', colors=colors[:len(event_counts)])
            ax5.set_title('Article Distribution by Event')
        
        # 6. Bias Score Summary
        ax6 = plt.subplot(2, 3, 6)
        
        # Calculate aggregate bias scores
        bias_scores = []
        for source in df['source'].unique():
            source_articles = df[df['source'] == source]
            
            # Calculate various bias metrics
            poor_coverage = 0
            middle_coverage = 0
            corp_coverage = 0
            
            for _, article in source_articles.iterrows():
                if pd.notna(article['content']):
                    alignment = self.detect_constituency_alignment(article['content'])
                    poor_coverage += alignment.get('poor', 0)
                    middle_coverage += alignment.get('middle_class', 0)
                    corp_coverage += alignment.get('corporate', 0)
            
            n = len(source_articles)
            if n > 0:
                bias_scores.append({
                    'source': source,
                    'poor_bias': poor_coverage / n,
                    'middle_bias': middle_coverage / n,
                    'corp_bias': corp_coverage / n
                })
        
        if bias_scores:
            bias_df = pd.DataFrame(bias_scores)
            x_pos = np.arange(len(bias_df))
            
            ax6.bar(x_pos - 0.25, bias_df['poor_bias'], 0.25, 
                   label='Poor', color='#e74c3c')
            ax6.bar(x_pos, bias_df['middle_bias'], 0.25, 
                   label='Middle Class', color='#3498db')
            ax6.bar(x_pos + 0.25, bias_df['corp_bias'], 0.25, 
                   label='Corporate', color='#f39c12')
            
            ax6.set_xticks(x_pos)
            ax6.set_xticklabels(bias_df['source'], rotation=45)
            ax6.set_ylabel('Bias Score')
            ax6.set_title('Constituency Bias Comparison')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('bias_analysis_eda.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def generate_bias_report(self, df):
        """
        Generate comprehensive bias analysis report
        """
        print("\n" + "="*80)
        print(" MEDIA BIAS ANALYSIS REPORT ".center(80))
        print("="*80)
        
        # 1. Overall Statistics
        print("\n1. DATASET OVERVIEW")
        print("-" * 40)
        print(f"Total Articles: {len(df)}")
        print(f"News Sources: {df['source'].nunique()}")
        if 'event' in df.columns:
            print(f"Events Covered: {df['event'].nunique()}")
        print(f"Date Range: {df['scraped_at'].min()} to {df['scraped_at'].max()}")
        
        # 2. Source-wise Analysis
        print("\n2. SOURCE-WISE ANALYSIS")
        print("-" * 40)
        
        for source in df['source'].unique():
            source_articles = df[df['source'] == source]
            print(f"\n{source}:")
            print(f"  Articles: {len(source_articles)}")
            
            # Average sentiment
            sentiments = []
            for _, article in source_articles.iterrows():
                if pd.notna(article['content']):
                    sent = self.calculate_sentiment(article['content'])
                    sentiments.append(sent['compound'])
            
            if sentiments:
                print(f"  Avg Sentiment: {np.mean(sentiments):.3f}")
                print(f"  Sentiment Std: {np.std(sentiments):.3f}")
            
            # Political profile (from paper)
            if source in self.source_profiles:
                profile = self.source_profiles[source]
                print(f"  Political Lean: {profile['political']}")
                print(f"  Ideological: {profile['ideological']}")
        
        # 3. Constituency Coverage Analysis
        print("\n3. CONSTITUENCY COVERAGE ANALYSIS")
        print("-" * 40)
        
        overall_coverage = {const: [] for const in self.constituencies}
        
        for _, article in df.iterrows():
            if pd.notna(article['content']):
                alignment = self.detect_constituency_alignment(article['content'])
                for const, score in alignment.items():
                    overall_coverage[const].append(score)
        
        print("\nAverage Coverage Scores:")
        for const, scores in overall_coverage.items():
            if scores:
                avg_score = np.mean(scores)
                print(f"  {const.replace('_', ' ').title()}: {avg_score:.4f}")
        
        # 4. Bias Indicators
        print("\n4. KEY BIAS INDICATORS")
        print("-" * 40)
        
        # Calculate bias metrics
        poor_coverage_avg = np.mean(overall_coverage['poor']) if overall_coverage['poor'] else 0
        middle_coverage_avg = np.mean(overall_coverage['middle_class']) if overall_coverage['middle_class'] else 0
        corp_coverage_avg = np.mean(overall_coverage['corporate']) if overall_coverage['corporate'] else 0
        
        bias_ratio = middle_coverage_avg / poor_coverage_avg if poor_coverage_avg > 0 else float('inf')
        
        print(f"\nMiddle Class to Poor Coverage Ratio: {bias_ratio:.2f}")
        print(f"Corporate to Poor Coverage Ratio: {corp_coverage_avg/poor_coverage_avg if poor_coverage_avg > 0 else 'N/A':.2f}")
        
        # Identify most biased source
        max_bias_source = None
        max_bias_score = 0
        
        for source in df['source'].unique():
            source_articles = df[df['source'] == source]
            poor_scores = []
            middle_scores = []
            
            for _, article in source_articles.iterrows():
                if pd.notna(article['content']):
                    alignment = self.detect_constituency_alignment(article['content'])
                    poor_scores.append(alignment.get('poor', 0))
                    middle_scores.append(alignment.get('middle_class', 0))
            
            if poor_scores and middle_scores:
                source_bias = np.mean(middle_scores) - np.mean(poor_scores)
                if abs(source_bias) > max_bias_score:
                    max_bias_score = abs(source_bias)
                    max_bias_source = source
        
        if max_bias_source:
            print(f"\nMost Biased Source: {max_bias_source} (bias score: {max_bias_score:.4f})")
        
        # 5. Recommendations
        print("\n5. BIAS MITIGATION RECOMMENDATIONS")
        print("-" * 40)
        
        if bias_ratio > 2:
            print("⚠ HIGH BIAS DETECTED: Middle class coverage significantly exceeds poor coverage")
            print("Recommendations:")
            print("  • Increase coverage of poverty-related issues")
            print("  • Include more perspectives from marginalized communities")
            print("  • Balance political rhetoric with ground-level reporting")
        elif bias_ratio > 1.5:
            print("⚠ MODERATE BIAS: Some imbalance in constituency coverage")
            print("Recommendations:")
            print("  • Diversify sources and perspectives")
            print("  • Include more informal sector voices")
        else:
            print("✓ Relatively balanced coverage across constituencies")
        
        print("\n" + "="*80)
        
        return overall_coverage

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = BiasAnalysisEDA()
    
    # Create sample data for demonstration
    sample_articles = [
        {
            'source': 'Times of India',
            'content': 'The stock market surged today as corporate profits exceeded expectations. Major companies reported strong quarterly results.',
            'event': 'demonetization',
            'scraped_at': '2024-01-01'
        },
        {
            'source': 'The Hindu',
            'content': 'Daily wage workers continue to struggle with unemployment. The poverty situation has worsened in slum areas.',
            'event': 'migrant_crisis',
            'scraped_at': '2024-01-02'
        },
        {
            'source': 'Indian Express',
            'content': 'The government announced new policy measures in parliament today. Ministers discussed the implementation of schemes.',
            'event': 'gst',
            'scraped_at': '2024-01-03'
        }
    ]
    
    # Convert to DataFrame
    df = pd.DataFrame(sample_articles)
    analyzer.articles_df = df
    
    # Generate visualizations
    print("Generating bias analysis visualizations...")
    analyzer.create_bias_visualizations(df)
    
    # Generate report
    print("\nGenerating bias report...")
    analyzer.generate_bias_report(df)
