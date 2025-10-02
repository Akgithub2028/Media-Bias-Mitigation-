# flask_chatbot_app.py
"""
Fully corrected Flask application with richer/detailed sample articles
and interactive endpoints.

Place this file in same directory as:
- chatbot_html_interface.html
- bias_mitigation_model.py
- indian_news_scraper.py

Run:
    python3 flask_chatbot_app.py
Open:
    http://localhost:5000
"""

from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
from datetime import datetime, timedelta
import os
import logging
import traceback
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import user modules (assumes these exist in the same folder)
from bias_mitigation_model import BiasNormalizationEngine
from indian_news_scraper import IndianNewsScraperPipeline

app = Flask(__name__, static_folder=None)
app.secret_key = os.environ.get("FLASK_SECRET", "change-this-secret-in-prod")
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bias_engine = BiasNormalizationEngine()
scraper = IndianNewsScraperPipeline()
article_cache = {}

AVAILABLE_ASPECTS = {
    'demonetization': [
        'Queues at Banks and ATMs',
        'Impact on Small Businesses',
        'Black Money and Corruption',
        'Digital Payment Adoption',
        'Impact on Poor and Daily Wage Workers',
        'Political Statements and Debates',
        'Economic Growth Impact',
        'Currency Exchange Problems'
    ],
    'gst': [
        'Tax Rate Structure',
        'Small Business Compliance',
        'Market and Economic Impact',
        'Implementation Challenges',
        'Political Discussions',
        'Impact on Informal Sector',
        'Revenue Collection',
        'Consumer Prices'
    ],
    'farmers_protest': [
        'Loan Waivers and Debt',
        'MSP and Crop Prices',
        'Farmer Suicides',
        'Political Support and Opposition',
        'Farm Bills and Legislation',
        'Irrigation and Water Issues',
        'Agricultural Crisis',
        'Government Response'
    ],
    'migrant_crisis': [
        'Lockdown Impact on Workers',
        'Migration to Home States',
        'Food and Shelter Issues',
        'Shramik Trains',
        'Daily Wage Loss',
        'Government Relief Measures',
        'Health Concerns',
        'Employment Crisis'
    ],
    'ladakh_protests': [
        'Article 370 Impact',
        'Statehood Demands',
        'Sixth Schedule Request',
        'Environmental Concerns',
        'Climate Activism',
        'Government Response',
        'Local Community Impact'
    ]
}

def _sample_numeric_range(base, pct=0.15):
    """helper to produce plausible numeric variations"""
    delta = base * pct
    return round(base + random.uniform(-delta, delta), 2)

def generate_sample_articles(event_key: str, aspect: str, n: int = 5, verbosity: str = "medium"):
    """
    Generates highly detailed, structured sample articles for demo / interactive UI.
    verbosity: "low" | "medium" | "high"
    Each article contains:
      - title, source, url, published_at
      - content (full text)
      - summary (short)
      - key_points (list)
      - quotes (list)
      - metrics: word_count, estimated_impact_score (0-1), sentiment (-1..1)
      - entities: list of named things (people/orgs)
    """
    sources = ['Times of India', 'The Hindu', 'Indian Express', 'Hindustan Times', 'Deccan Herald']
    now_iso = datetime.now().isoformat()
    articles = []

    # Base facts to incorporate into articles (simulated numbers)
    base_impact_pct = 3.5 if event_key == 'gst' else 7.2 if event_key == 'demonetization' else 5.0
    avg_income_loss = 1200 if event_key == 'migrant_crisis' else 400 if event_key == 'farmers_protest' else 0

    for i in range(min(n, len(sources))):
        src = sources[i]
        title = f"{aspect} — {src} analysis and implications"
        # Compose sections
        lead = (
            f"{aspect} has drawn diverse responses across stakeholders. "
            f"This report synthesizes reactions from local communities, market analysts and policymakers."
        )

        background = (
            "Background: Over the last 12 months policy developments and ground-level events have created a complex picture. "
            "Multiple sources highlight logistical issues, pockets of economic stress, and political debate surrounding the issue."
        )

        # Simulate data paragraph
        impact_val = _sample_numeric_range(base_impact_pct, pct=0.25)
        data_paragraph = (
            f"Data snapshot: our aggregated review of media reporting and available surveys suggests an estimated {impact_val}% "
            f"measurable impact on short-term economic indicators (employment rates, daily earnings, local business revenues)."
        )

        # Human stories paragraph
        human_paragraph = (
            f"Voices from the field: 'We had to wait in line for hours, and the daily earnings dropped significantly,' said a local worker. "
            f"Small shop owners reported inventory turnover slowing by an estimated {_sample_numeric_range(12, pct=0.3)}% in affected districts."
        )

        # Policy and commentary
        policy_paragraph = (
            "Policy context: Officials say measures are aimed at longer-term stability, but immediate relief and clearer communication are being demanded by opposition parties and civil society."
        )

        # Add extra depth at high verbosity
        expert_quotes = []
        if verbosity in ("medium", "high"):
            expert_quotes.append({
                'text': "This change will likely shift consumption patterns in the short term, but structural adjustments are needed for small traders.",
                'speaker': "Dr. A. Sharma, economist"
            })
        if verbosity == "high":
            expert_quotes.append({
                'text': "Micro-surveys indicate highly localized distress; state-level interventions targeted at informal workers could reduce adverse effects.",
                'speaker': "Prof. N. Iyer, social policy researcher"
            })

        # Compose full content and summary
        sections = [lead, background, data_paragraph, human_paragraph, policy_paragraph]
        if verbosity == "high":
            sections.append(
                "Deeper analysis: We include a short timeline of events, breakdown of affected sectors, and a mini-case study documenting supply chain disruptions in one district."
            )
        content = "\n\n".join(sections)

        # Key points
        key_points = [
            f"Estimated short-term impact: {impact_val}%",
            "Immediate strain on daily-wage earners and small retailers",
            "Calls for improved communication and targeted relief measures"
        ]
        if verbosity == "high":
            key_points.append("Suggested policy measures: targeted cash transfers, temporary tax relief, expedited digital adoption training for SMEs")

        # Sentiment and metrics
        sentiment = round(random.uniform(-0.3, 0.4) + (-0.1 if 'government' in aspect.lower() else 0), 3)
        estimated_impact_score = min(1.0, max(0.0, impact_val / 20.0))  # normalized

        article_obj = {
            'title': title,
            'source': src,
            'source_key': src.lower().replace(' ', ''),
            'url': f"https://example.com/{event_key}/{i}",
            'published_at': now_iso,
            'summary': (content[:360] + '...') if len(content) > 360 else content,
            'content': content,
            'key_points': key_points,
            'quotes': expert_quotes,
            'metrics': {
                'word_count': len(content.split()),
                'estimated_impact_score': round(estimated_impact_score, 3),
                'sentiment': sentiment
            },
            'entities': ['Small businesses', 'Daily wage workers', 'State government'],
            'event': event_key,
            'aspect': aspect
        }

        # If verbosity=='low' strip some fields
        if verbosity == "low":
            article_obj.pop('quotes', None)
            article_obj['summary'] = article_obj['summary'][:180] + '...'
            article_obj['key_points'] = article_obj['key_points'][:2]

        # If verbosity == 'high' add a simulated data table snippet
        if verbosity == "high":
            article_obj['data_snippet'] = {
                'timeline': [
                    {'date': (datetime.now() - timedelta(days=90)).date().isoformat(), 'event': 'Initial policy announcement'},
                    {'date': (datetime.now() - timedelta(days=30)).date().isoformat(), 'event': 'Local disruptions reported'},
                ],
                'sample_stats': {
                    'avg_daily_loss_local_rs': round(_sample_numeric_range(avg_income_loss if avg_income_loss > 0 else 100, pct=0.2)),
                    'affected_villages_sample': random.randint(3, 25)
                }
            }

        articles.append(article_obj)

    return articles

@app.route('/')
def index():
    html_name = 'chatbot_html_interface.html'
    if not os.path.exists(html_name):
        return "<h3>Frontend file not found: chatbot_chatbot_interface.html</h3>", 404
    return send_from_directory('.', html_name)

@app.route('/api/get_events', methods=['GET'])
def get_events():
    return jsonify({'success': True, 'events': list(AVAILABLE_ASPECTS.keys())})

@app.route('/api/get_aspects', methods=['POST'])
def get_aspects():
    try:
        data = request.get_json(force=True)
        event = data.get('event', '')
        if event not in AVAILABLE_ASPECTS:
            return jsonify({'success': False, 'error': 'Invalid event selected'}), 400
        return jsonify({'success': True, 'event': event, 'aspects': AVAILABLE_ASPECTS[event]})
    except Exception:
        logger.exception("get_aspects")
        return jsonify({'success': False, 'error': 'Server error'}), 500

@app.route('/api/fetch_articles', methods=['POST'])
def fetch_articles():
    """
    Request body: { event, aspect, fetch_live (optional), verbosity (low|medium|high) }
    """
    try:
        data = request.get_json(force=True)
        event = data.get('event', '')
        aspect = data.get('aspect', '')
        fetch_live = bool(data.get('fetch_live', False))
        verbosity = data.get('verbosity', 'medium')
        verbosity = verbosity if verbosity in ('low', 'medium', 'high') else 'medium'

        if not event or not aspect:
            return jsonify({'success': False, 'error': 'event and aspect required'}), 400

        cache_key = f"{event}___{aspect}___{verbosity}"
        if cache_key in article_cache and not fetch_live:
            return jsonify({'success': True, 'articles': article_cache[cache_key], 'cached': True})

        # Attempt to use scraper (best-effort, but do not block indefinitely)
        found_articles = []
        try:
            keywords = [w for w in aspect.split() if len(w) > 2]
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=365)
            for source_key in list(scraper.sources.keys()):
                try:
                    src_results = scraper.search_articles(source_key, keywords[:3], start_date.isoformat(), end_date.isoformat())
                    if src_results:
                        with ThreadPoolExecutor(max_workers=4) as ex:
                            futures = []
                            for art in src_results[:2]:
                                futures.append(ex.submit(scraper.scrape_article, art))
                            for fut in as_completed(futures, timeout=20):
                                try:
                                    a = fut.result()
                                    if a:
                                        # Normalize to article schema (best-effort)
                                        normalized = {
                                            'title': a.get('title') or a.get('headline', 'Article'),
                                            'source': scraper.sources.get(source_key, {}).get('name', source_key),
                                            'url': a.get('url'),
                                            'published_at': a.get('published_at', datetime.now().isoformat()),
                                            'content': a.get('content', a.get('summary', '')),
                                            'summary': (a.get('summary') or '')[:360],
                                            'key_points': [],
                                            'quotes': [],
                                            'metrics': {'word_count': len((a.get('content') or '').split())},
                                            'entities': [],
                                            'event': event,
                                            'aspect': aspect
                                        }
                                        found_articles.append(normalized)
                                except Exception:
                                    logger.debug("article fetch failed: %s", traceback.format_exc())
                except Exception:
                    logger.debug("search failed for source %s: %s", source_key, traceback.format_exc())

            # If scraper produced insufficient articles, generate samples
            if len(found_articles) < 2:
                found_articles = generate_sample_articles(event, aspect, n=5, verbosity=verbosity)
        except Exception:
            logger.exception("Scraper pipeline failed; returning generated samples")
            found_articles = generate_sample_articles(event, aspect, n=5, verbosity=verbosity)

        article_cache[cache_key] = found_articles
        return jsonify({'success': True, 'articles': found_articles, 'cached': False, 'message': f'Fetched {len(found_articles)} articles', 'verbosity': verbosity})

    except Exception:
        logger.exception("fetch_articles error")
        return jsonify({'success': False, 'error': 'Server error'}), 500

@app.route('/api/mitigate_bias', methods=['POST'])
def mitigate_bias():
    try:
        data = request.get_json(force=True)
        articles = data.get('articles', [])
        summary_type = data.get('summary_type', 'humanized')

        if not articles:
            return jsonify({'success': False, 'error': 'No articles provided'}), 400

        try:
            if summary_type == 'humanized':
                mitigated_summary = bias_engine.generate_humanized_summary(articles)
                bias_metrics = bias_engine.calculate_bias_metrics(articles)
            else:
                res = bias_engine.normalize_and_mitigate_bias(articles)
                mitigated_summary = res.get('mitigated_summary', '')
                bias_metrics = res.get('bias_metrics', {})
        except Exception:
            logger.exception("bias engine failed")
            mitigated_summary = "Error: could not generate bias-mitigated summary."
            bias_metrics = {}

        return jsonify({
            'success': True,
            'mitigated_summary': mitigated_summary,
            'bias_metrics': {
                'overall_coverage': bias_metrics.get('overall_coverage', {}) if isinstance(bias_metrics, dict) else {},
                'bias_severity': bias_metrics.get('bias_severity', {}) if isinstance(bias_metrics, dict) else {},
                'total_articles': bias_metrics.get('total_articles', len(articles)) if isinstance(bias_metrics, dict) else len(articles)
            }
        })
    except Exception:
        logger.exception("mitigate_bias error")
        return jsonify({'success': False, 'error': 'Server error'}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True)
        user_message = (data.get('message') or '').strip()
        event = data.get('event', '')
        aspect = data.get('aspect', '')

        if 'conversation_history' not in session:
            session['conversation_history'] = []
        session['conversation_history'].append({'user': user_message, 'ts': datetime.now().isoformat()})
        if len(session['conversation_history']) > 30:
            session['conversation_history'] = session['conversation_history'][-30:]

        res = {'success': True, 'timestamp': datetime.now().isoformat()}

        msg_lower = user_message.lower()
        if any(w in msg_lower for w in ['hello', 'hi', 'hey', 'start']):
            res['message'] = "Hello! Select an event or type which event you'd like to explore (e.g., 'gst', 'demonetization')."
            res['type'] = 'greeting'
            return jsonify(res)

        if 'help' in msg_lower:
            res['message'] = "To explore: select an event, choose an aspect, then pick 'Request detail' to fetch interactive articles. You can click 'More detail' per article for expanded content."
            return jsonify(res)

        if event and aspect:
            res['type'] = 'processing'
            res['event'] = event
            res['aspect'] = aspect
            res['trigger_fetch'] = True
            res['message'] = f"Triggering fetch+analysis for {aspect}."
            return jsonify(res)

        # fallback
        res['message'] = "I couldn't match your input. Try 'demonetization' or pick an event."
        return jsonify(res)
    except Exception:
        logger.exception("chat error")
        return jsonify({'success': False, 'error': 'Server error'}), 500

@app.route('/api/get_bias_explanation', methods=['GET'])
def get_bias_explanation():
    explanation = {
        'success': True,
        'bias_types': {
            'coverage_bias': {'name': 'Coverage Bias', 'description': 'Unequal attention given to different constituencies.'},
            'framing_bias': {'name': 'Framing Bias', 'description': 'Issues presented from particular perspectives.'},
            'sentiment_bias': {'name': 'Sentiment Bias', 'description': 'Positive/negative slant toward actors.'}
        }
    }
    return jsonify(explanation)

if __name__ == '__main__':
    logger.info("Starting backend with richer article support...")
    app.run(host='0.0.0.0', port=5000, debug=True)
