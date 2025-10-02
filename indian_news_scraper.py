import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
import json
from urllib.parse import urljoin, urlparse
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import re

class IndianNewsScraperPipeline:
    """
    Targeted news scraping for Indian sources with constituency-aware keywords
    Based on the research paper's methodology
    """
    
    def __init__(self):
        # Five major news sources from the paper
        self.sources = {
            'toi': {
                'name': 'Times of India',
                'base_url': 'https://timesofindia.indiatimes.com',
                'search_url': 'https://timesofindia.indiatimes.com/topic/',
                'selectors': {
                    'title': ['h1.heading', 'h1._1Y-96', 'h1'],
                    'content': ['div.article-content', 'div._3WlLe', 'div.Normal'],
                    'date': ['span.date', 'div._3Mkg-'],
                    'author': ['span.author', 'span.auth_detail']
                }
            },
            'hindu': {
                'name': 'The Hindu',
                'base_url': 'https://www.thehindu.com',
                'search_url': 'https://www.thehindu.com/search/?q=',
                'selectors': {
                    'title': ['h1.title', 'h1', 'h1.article-headline'],
                    'content': ['div.article-text', 'div[itemprop="articleBody"]', 'div.paywall'],
                    'date': ['div.publish-time', 'div.article-publish-date'],
                    'author': ['span.author-name', 'a.auth-nm']
                }
            },
            'indianexpress': {
                'name': 'Indian Express',
                'base_url': 'https://indianexpress.com',
                'search_url': 'https://indianexpress.com/?s=',
                'selectors': {
                    'title': ['h1', 'h1.native-story-title'],
                    'content': ['div.full-details', 'div.story_details', 'div.ie2013-contentstory'],
                    'date': ['span.date', 'span.posted-date'],
                    'author': ['span.contributor', 'a.story-date-author-link']
                }
            },
            'ht': {
                'name': 'Hindustan Times',
                'base_url': 'https://www.hindustantimes.com',
                'search_url': 'https://www.hindustantimes.com/search?q=',
                'selectors': {
                    'title': ['h1.hdg1', 'h1'],
                    'content': ['div.storyDetails', 'div.detail', 'div.story-details'],
                    'date': ['span.timestampUpdated', 'span.date'],
                    'author': ['span.author', 'div.dateTime']
                }
            },
            'deccanherald': {
                'name': 'Deccan Herald',
                'base_url': 'https://www.deccanherald.com',
                'search_url': 'https://www.deccanherald.com/search?searchText=',
                'selectors': {
                    'title': ['h1.title', 'h1'],
                    'content': ['div.article-content', 'div.content', 'div.body'],
                    'date': ['div.article-date', 'ul.meta-info'],
                    'author': ['span.author', 'div.author-name']
                }
            }
        }
        
        # Keywords for each event based on paper's methodology
        self.event_keywords = {
            'demonetization': {
                'keywords': ['demonetization', 'demonetisation', 'note ban', 'cash withdrawal',
                            'black money', 'cashless economy', 'digital payment', 'currency switch',
                            'long queue', 'ATM', 'cash crunch', 'Rs 500', 'Rs 1000'],
                'timeline': ('2016-11-08', '2017-03-31')
            },
            'gst': {
                'keywords': ['GST', 'goods service tax', 'goods and services tax', 'tax reform',
                           'indirect tax', 'GST council', 'GST rate', 'tax slab', 'GST registration'],
                'timeline': ('2017-07-01', '2018-06-30')
            },
            'farmers_protest': {
                'keywords': ['farmer protest', 'farmers protest', 'farm loan', 'loan waiver',
                           'farmer suicide', 'agrarian crisis', 'MSP', 'minimum support price',
                           'kisan', 'farm bills', 'farmer agitation'],
                'timeline': ('2020-09-01', '2021-12-31')
            },
            'migrant_crisis': {
                'keywords': ['migrant worker', 'migrant crisis', 'lockdown', 'exodus',
                           'daily wage', 'labour', 'shramik train', 'stranded workers',
                           'reverse migration'],
                'timeline': ('2020-03-24', '2020-08-31')
            },
            'ladakh_protests': {
                'keywords': ['ladakh protest', 'article 370', 'union territory', 'ladakh UT',
                           'sixth schedule', 'statehood demand', 'sonam wangchuk', 'climate fast'],
                'timeline': ('2023-01-01', '2024-12-31')
            }
        }
        
        # Constituency keywords based on paper's methodology
        self.constituency_keywords = {
            'poor': ['poverty', 'slum', 'hunger', 'unemployment', 'daily wage', 'BPL',
                    'migrant worker', 'laborer', 'homeless', 'ration', 'MGNREGA',
                    'below poverty line', 'informal labour'],
            'middle_class': ['salary', 'EMI', 'tax', 'savings', 'education', 'urban',
                           'professional', 'IT', 'bank account', 'credit card', 'salaried',
                           'income tax', 'housing loan'],
            'corporate': ['business', 'industry', 'company', 'profit', 'investment',
                        'market', 'stock', 'CEO', 'enterprise', 'revenue', 'sensex',
                        'nifty', 'corporate tax', 'quarterly results'],
            'informal_sector': ['vendor', 'small business', 'trader', 'unorganized',
                              'self-employed', 'shopkeeper', 'hawker', 'street vendor',
                              'MSME', 'small trader', 'informal economy'],
            'government': ['policy', 'minister', 'parliament', 'legislation', 'scheme',
                         'government', 'official', 'bureaucrat', 'administration',
                         'Modi', 'BJP', 'Congress', 'opposition']
        }
        
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging for debugging and monitoring"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('indian_news_scraper.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def search_articles(self, source, keywords, start_date, end_date):
        """
        Search for articles using keywords within date range
        Limited scraping for Colab constraints
        """
        articles = []
        base_url = self.sources[source]['base_url']
        search_url = self.sources[source]['search_url']
        
        # Limit articles per keyword to avoid overloading
        max_articles_per_keyword = 5  # Conservative for Colab
        
        for keyword in keywords[:3]:  # Only use top 3 keywords
            try:
                # Construct search URL
                url = f"{search_url}{keyword.replace(' ', '%20')}"
                
                response = requests.get(url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Find article links (generic approach)
                    links = soup.find_all('a', href=True)[:max_articles_per_keyword]
                    
                    for link in links:
                        article_url = urljoin(base_url, link['href'])
                        if self.is_valid_article_url(article_url, source):
                            articles.append({
                                'url': article_url,
                                'keyword': keyword,
                                'source': source,
                                'event_period': f"{start_date} to {end_date}"
                            })
                
                # Rate limiting
                time.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Error searching {source} for {keyword}: {e}")
        
        return articles
    
    def is_valid_article_url(self, url, source):
        """Validate if URL is an actual article"""
        invalid_patterns = ['video', 'gallery', 'photogallery', 'slideshow', 
                          'live', 'podcast', 'author', 'topic']
        
        # Check if URL contains article indicators
        article_patterns = {
            'toi': '/articleshow/',
            'hindu': '/article',
            'indianexpress': '/article/',
            'ht': '/news/',
            'deccanherald': '/news/'
        }
        
        # Avoid non-article pages
        if any(pattern in url.lower() for pattern in invalid_patterns):
            return False
        
        # Check for article pattern if defined
        if source in article_patterns:
            return article_patterns[source] in url
        
        return True
    
    def scrape_article(self, article_info):
        """
        Scrape individual article with constituency analysis
        """
        try:
            response = requests.get(article_info['url'], timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            source = article_info['source']
            selectors = self.sources[source]['selectors']
            
            # Extract content with fallback selectors
            article_data = {
                'url': article_info['url'],
                'source': self.sources[source]['name'],
                'source_key': source,
                'keyword': article_info['keyword'],
                'event_period': article_info['event_period'],
                'scraped_at': datetime.now().isoformat()
            }
            
            # Try multiple selectors for each field
            for field, selector_list in selectors.items():
                content = None
                for selector in selector_list:
                    element = soup.select_one(selector)
                    if element:
                        content = element.text.strip()
                        break
                article_data[field] = content
            
            # Clean and analyze content
            if article_data.get('content'):
                article_data['content'] = self.clean_text(article_data['content'])
                article_data['word_count'] = len(article_data['content'].split())
                
                # Analyze constituency mentions
                article_data['constituency_mentions'] = self.analyze_constituency_mentions(
                    article_data['content']
                )
            
            return article_data
            
        except Exception as e:
            self.logger.error(f"Error scraping {article_info['url']}: {e}")
            return None
    
    def analyze_constituency_mentions(self, text):
        """
        Count constituency keyword mentions in article
        Based on paper's methodology
        """
        text_lower = text.lower()
        mentions = {}
        
        for constituency, keywords in self.constituency_keywords.items():
            count = sum(text_lower.count(keyword.lower()) for keyword in keywords)
            mentions[constituency] = count
        
        # Normalize by text length
        text_length = len(text.split())
        if text_length > 0:
            mentions = {k: v/text_length for k, v in mentions.items()}
        
        return mentions
    
    def clean_text(self, text):
        """Clean extracted text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters but keep punctuation
        text = ''.join(char for char in text if char.isprintable())
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        return text
    
    def run_targeted_scraping(self, events_to_scrape=None, max_articles_per_event=25):
        """
        Run scraping for specific events with limited articles
        Optimized for Colab's constraints
        """
        if events_to_scrape is None:
            events_to_scrape = list(self.event_keywords.keys())
        
        all_articles = []
        
        for event in events_to_scrape:
            self.logger.info(f"Scraping articles for: {event}")
            event_data = self.event_keywords[event]
            
            articles_per_source = max_articles_per_event // len(self.sources)
            
            for source in self.sources:
                self.logger.info(f"  Scraping {source}...")
                
                # Search for articles
                article_urls = self.search_articles(
                    source,
                    event_data['keywords'],
                    event_data['timeline'][0],
                    event_data['timeline'][1]
                )[:articles_per_source]
                
                # Scrape articles
                for article_info in article_urls:
                    article_data = self.scrape_article(article_info)
                    if article_data:
                        article_data['event'] = event
                        all_articles.append(article_data)
                        self.logger.info(f"    Scraped: {article_data['title'][:50] if article_data.get('title') else 'No title'}")
                    
                    # Rate limiting
                    time.sleep(1)
        
        # Save results
        self.save_data(all_articles)
        
        return all_articles
    
    def save_data(self, articles):
        """Save scraped data to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as JSON
        with open(f'indian_news_articles_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        
        # Save as CSV
        df = pd.DataFrame(articles)
        df.to_csv(f'indian_news_articles_{timestamp}.csv', index=False, encoding='utf-8')
        
        self.logger.info(f"Data saved: {len(articles)} articles")
        
        # Print summary statistics
        if articles:
            print("\n=== Scraping Summary ===")
            print(f"Total articles: {len(articles)}")
            
            # Articles per source
            source_counts = df['source'].value_counts()
            print("\nArticles per source:")
            for source, count in source_counts.items():
                print(f"  {source}: {count}")
            
            # Articles per event
            if 'event' in df.columns:
                event_counts = df['event'].value_counts()
                print("\nArticles per event:")
                for event, count in event_counts.items():
                    print(f"  {event}: {count}")

# Usage
if __name__ == "__main__":
    scraper = IndianNewsScraperPipeline()
    
    # Scrape limited articles for each event
    articles = scraper.run_targeted_scraping(
        events_to_scrape=['demonetization', 'gst', 'farmers_protest', 'migrant_crisis'],
        max_articles_per_event=25  # Conservative for Colab
    )
    
    print(f"\nTotal articles scraped: {len(articles)}")
