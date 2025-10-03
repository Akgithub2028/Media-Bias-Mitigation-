#!/usr/bin/env python3
# preprocessing_eda.py
"""
Enhanced preprocessing + intense EDA for Indian news scraper outputs.

Usage:
    python preprocessing_eda.py --input /path/to/scraped_dir_or_file --outdir ./eda_outputs --topics 12

Outputs:
 - CSVs and PNGs saved to outdir
 - Returns intermediate artifacts in a results dict when used as module
Requirements:
 pip install pandas numpy matplotlib seaborn scikit-learn nltk textblob wordcloud scipy tqdm
"""

import os
import re
import json
import glob
import math
import argparse
import warnings
warnings.filterwarnings("ignore")

from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import euclidean
from scipy.special import rel_entr

# Ensure NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# -----------------------
# Utilities
# -----------------------
def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def read_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, 'r', encoding='utf8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except Exception:
                # try trimming trailing comma
                try:
                    out.append(json.loads(ln.rstrip(',')))
                except:
                    continue
    return out

def jensen_shannon(p: np.ndarray, q: np.ndarray, eps=1e-12) -> float:
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (rel_entr(p, m).sum() + rel_entr(q, m).sum())

# -----------------------
# Main class
# -----------------------
class PreprocessEDA:
    def __init__(self, input_path: str, out_dir: str = "./eda_outputs"):
        self.input_path = input_path
        self.out_dir = out_dir
        safe_mkdir(self.out_dir)

        # NLP helpers
        self.stopset = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.sia = SentimentIntensityAnalyzer()

        # Expanded constituency keywords (India-focused)
        self.constituencies = {
            'poor': {
                'keywords': [
                    'poverty','slum','hunger','unemployment','daily wage','migrant worker','laborer','homeless','bpl','ration',
                    'mgnrega','below poverty line','informal labour','ration card','jan dhan','food security','aadhar','pds',
                    'anganwadi','low income','manual scavenger','jobless','villager','daily wage earner','welfare','poverty line'
                ],
                'color': '#e74c3c'
            },
            'middle_class': {
                'keywords': [
                    'salary','emi','tax','savings','education','urban','professional','it','bank account','credit card','income tax',
                    'housing loan','salaried','gst returns','middle income','tuition fees','job market','metro','aspirational',
                    'upper middle class','middle class','white collar','dual income'
                ],
                'color': '#3498db'
            },
            'corporate': {
                'keywords': [
                    'business','industry','company','profit','investment','market','stock','ceo','enterprise','revenue','sensex','nifty',
                    'corporate tax','ipo','merger','acquisition','startup','venture capital','fdi','multinational','exports','imports',
                    'private equity','shareholder','board','corporation','msme formal'
                ],
                'color': '#f39c12'
            },
            'informal_sector': {
                'keywords': [
                    'vendor','small business','trader','unorganized','self-employed','shopkeeper','hawker','street vendor','msme',
                    'informal economy','handicraft','artisan','daily trader','mandi','flea market','self help group','cottage industry',
                    'microenterprise','local vendor','kiosk','tiffin', 'dabbawala'
                ],
                'color': '#9b59b6'
            },
            'government': {
                'keywords': [
                    'policy','minister','parliament','legislation','scheme','government','official','bureaucrat','administration','modi',
                    'bjp','congress','opposition','state govt','cabinet','ordinance','election','supreme court','rajya sabha','lok sabha',
                    'policy reform','rbi','niti aayog','planning commission','scheme launch','govt scheme','gazette'
                ],
                'color': '#27ae60'
            }
        }

        # loaded data
        self.df: Optional[pd.DataFrame] = None

    # -----------------------
    # Data loading
    # -----------------------
    def load(self):
        if os.path.isdir(self.input_path):
            records = []
            # look for jsonl/json/csv
            for fp in glob.glob(os.path.join(self.input_path, "*.jsonl")) + glob.glob(os.path.join(self.input_path, "*.json")) + glob.glob(os.path.join(self.input_path, "*.csv")):
                try:
                    if fp.endswith(".jsonl"):
                        records.extend(read_jsonl(fp))
                    elif fp.endswith(".json"):
                        with open(fp, 'r', encoding='utf8') as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            records.extend(data)
                        elif isinstance(data, dict):
                            records.append(data)
                    elif fp.endswith(".csv"):
                        df = pd.read_csv(fp)
                        records.extend(df.to_dict('records'))
                except Exception as e:
                    print("Error reading", fp, e)
            self.df = pd.DataFrame(records)
        else:
            if self.input_path.endswith(".jsonl"):
                self.df = pd.DataFrame(read_jsonl(self.input_path))
            elif self.input_path.endswith(".json"):
                with open(self.input_path, 'r', encoding='utf8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self.df = pd.DataFrame(data)
                else:
                    self.df = pd.DataFrame([data])
            elif self.input_path.endswith(".csv"):
                self.df = pd.read_csv(self.input_path)
            else:
                raise ValueError("Unsupported input path type")

        if self.df is None or self.df.empty:
            raise ValueError("No articles loaded")

        # Normalize columns: prefer content/text/body/article_text
        content_cols = ['content','text','body','article_text','article','articleBody']
        for c in content_cols:
            if c in self.df.columns:
                self.df['content'] = self.df.get('content', pd.Series(dtype=str)).fillna(self.df[c].astype(str))
        if 'content' not in self.df.columns:
            # try combining paragraphs
            self.df['content'] = self.df.astype(str).sum(axis=1)

        # Source
        for c in ['source','handle','site','publisher','source_name']:
            if c in self.df.columns:
                self.df['source'] = self.df.get('source', pd.Series(dtype=str)).fillna(self.df[c].astype(str))
        if 'source' not in self.df.columns:
            self.df['source'] = 'Unknown'

        # Title
        for c in ['title','headline','heading']:
            if c in self.df.columns:
                self.df['title'] = self.df.get('title', pd.Series(dtype=str)).fillna(self.df[c].astype(str))

        # created_at normalization (attempt parse)
        if 'created_at' in self.df.columns:
            try:
                self.df['created_at'] = pd.to_datetime(self.df['created_at'], errors='coerce')
            except:
                self.df['created_at'] = pd.NaT
        else:
            self.df['created_at'] = pd.NaT

        # dedupe by url or title or content hash
        if 'url' in self.df.columns:
            self.df = self.df.drop_duplicates(subset=['url'])
        elif 'title' in self.df.columns:
            self.df = self.df.drop_duplicates(subset=['title'])
        else:
            # hash content
            self.df['__chash'] = self.df['content'].apply(lambda x: hash(str(x)[:2000]))
            self.df = self.df.drop_duplicates(subset=['__chash'])
            self.df.drop(columns=['__chash'], inplace=True)

        # fill missing fields
        self.df['source'] = self.df['source'].fillna('Unknown').astype(str)
        self.df['title'] = self.df.get('title', pd.Series(['']*len(self.df))).fillna('').astype(str)
        self.df['content'] = self.df['content'].fillna('').astype(str)
        # basic length stats
        self.df['char_count_raw'] = self.df['content'].apply(lambda x: len(x))
        self.df['word_count_raw'] = self.df['content'].apply(lambda x: len(str(x).split()))
        print(f"Loaded {len(self.df)} articles from {self.input_path}")
        return self.df

    # -----------------------
    # Preprocessing
    # -----------------------
    def clean_text(self, text: str, lower: bool = True, remove_digits: bool = True, remove_punct: bool = True) -> str:
        s = str(text) if text is not None else ""
        if lower:
            s = s.lower()
        # remove urls
        s = re.sub(r'http\S+|www\.\S+', ' ', s)
        # remove emails
        s = re.sub(r'\S+@\S+', ' ', s)
        if remove_digits:
            s = re.sub(r'\d+', ' ', s)
        if remove_punct:
            s = re.sub(r'[^\w\s]', ' ', s)
        # normalize whitespace
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    def preprocess(self, text_col: str = 'content', out_col: str = 'content_clean'):
        if self.df is None:
            raise ValueError("Load data first")
        # apply cleaning
        self.df[out_col] = self.df[text_col].apply(lambda x: self.clean_text(x))
        # tokenize, remove stopwords, lemmatize
        def tok_and_filter(s):
            toks = [t for t in word_tokenize(s) if len(t) > 2]
            toks = [t for t in toks if t not in self.stopset]
            toks = self.lemmatize_tokens(toks)
            return toks
        tqdm.pandas(desc="Tokenizing")
        self.df['tokens'] = self.df[out_col].progress_apply(tok_and_filter)
        self.df['content_clean'] = self.df['tokens'].apply(lambda toks: " ".join(toks))
        self.df['unique_tokens'] = self.df['tokens'].apply(lambda t: len(set(t)))
        # sentences and sentiment
        self.df['sentences'] = self.df['content'].apply(lambda x: sent_tokenize(x))
        self.df['vader_compound'] = self.df['content'].apply(lambda x: self.sia.polarity_scores(x)['compound'] if x else 0)
        self.df['textblob_polarity'] = self.df['content'].apply(lambda x: TextBlob(x).sentiment.polarity if x else 0)
        # re-compute word counts on cleaned
        self.df['word_count'] = self.df['content_clean'].apply(lambda x: len(str(x).split()))
        self.df['char_count'] = self.df['content_clean'].apply(lambda x: len(x))
        return self.df

    # -----------------------
    # Topic / aspect extraction
    # -----------------------
    def run_lda(self, docs: List[str], n_topics: int = 12, max_features: int = 3000, min_df: int = 5) -> Tuple[np.ndarray, List[dict], LatentDirichletAllocation, CountVectorizer]:
        vectorizer = CountVectorizer(max_df=0.95, min_df=min_df, max_features=max_features)
        dtm = vectorizer.fit_transform(docs)
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=30)
        doc_topic = lda.fit_transform(dtm)
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for i, comp in enumerate(lda.components_):
            top_idx = comp.argsort()[-15:][::-1]
            top_words = [feature_names[j] for j in top_idx]
            topics.append({'topic_id': i, 'words': top_words, 'weights': comp[top_idx].tolist()})
        return doc_topic, topics, lda, vectorizer

    def assign_topics(self, doc_topic: np.ndarray, threshold: float = 0.25) -> List[List[int]]:
        assignments = []
        for row in doc_topic:
            assigned = [i for i, v in enumerate(row) if v >= threshold]
            if not assigned:
                assigned = [int(np.argmax(row))]
            assignments.append(assigned)
        return assignments

    # -----------------------
    # Constituency alignment
    # -----------------------
    def constituency_scores(self, text: str, tfidf_vec: Optional[TfidfVectorizer] = None, topic_words: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute normalized constituency scores:
          - simple keyword overlap normalized by doc length
          - optionally combine TF-IDF affinity by checking overlap with topic_words
        """
        scores = {k: 0.0 for k in self.constituencies.keys()}
        if not text or len(text.strip()) == 0:
            return scores
        text_l = text.lower()
        words = text_l.split()
        L = max(1, len(words))
        # keyword counts
        for cname, cinfo in self.constituencies.items():
            hits = sum(text_l.count(kw) for kw in cinfo['keywords'])
            scores[cname] = hits / L
        # extra: if topic_words provided, boost constituency mapping where topic words match keywords
        if topic_words:
            # topic_words is list of strings (words) from a topic/aspect
            for cname, cinfo in self.constituencies.items():
                common = sum(1 for w in topic_words if any(kw in w or w in kw for kw in cinfo['keywords']))
                # small boost normalized
                scores[cname] += 0.2 * (common / max(1, len(topic_words)))
        # normalize to sum <=1 (not strictly necessary)
        total = sum(abs(v) for v in scores.values()) or 1.0
        for k in scores:
            scores[k] = scores[k] / total
        return scores

    # -----------------------
    # Coverage & frame computation (approximate)
    # -----------------------
    def coverage_by_source(self, assignments: List[List[int]], topics: List[dict]) -> pd.DataFrame:
        df = self.df.copy()
        df['topic_assignments'] = assignments
        n_topics = len(topics)
        sources = df['source'].unique()
        rows = []
        for src in sources:
            src_df = df[df['source'] == src]
            total_words = src_df['word_count'].sum() or 1
            # per topic, compute word share
            shares = []
            for t in range(n_topics):
                mask = src_df['topic_assignments'].apply(lambda a: t in a)
                words_t = src_df[mask]['word_count'].sum()
                shares.append(words_t / total_words)
            row = {'source': src}
            row.update({f'aspect_{i}': shares[i] for i in range(n_topics)})
            rows.append(row)
        cov_df = pd.DataFrame(rows).fillna(0)
        return cov_df

    def compute_frame_matrix(self, doc_topic: np.ndarray, topics: List[dict]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Approximate frame matrix M(n,c).
        Steps:
         - primary topic per doc = argmax(doc_topic)
         - compute sentiment offset per topic (Sian - Savg(a))
         - approximate topic->constituency mapping U[a,c] via overlap between topic words & constituency keywords (and small TF-IDF boost)
         - compute M[n,c] = sum_a U[a,c] * San (weighted by word counts)
        """
        df = self.df.reset_index(drop=True).copy()
        primary = np.argmax(doc_topic, axis=1)
        df['primary_topic'] = primary
        n_topics = doc_topic.shape[1]
        consts = list(self.constituencies.keys())
        # avg sentiment per topic
        aspect_sent = {}
        for a in range(n_topics):
            mask = df['primary_topic'] == a
            aspect_sent[a] = df.loc[mask, 'vader_compound'].mean() if mask.sum() > 0 else 0.0

        # build U matrix
        U = {}
        for a, tinfo in enumerate(topics):
            twords = tinfo['words']
            affin = {}
            for c in consts:
                kwset = set(self.constituencies[c]['keywords'])
                overlap = sum(1 for w in twords if any(kw in w or w in kw for kw in kwset))
                # also check substring matches in constituency keywords vs topic words
                substr = sum(1 for w in twords for kw in kwset if kw in w or w in kw)
                affin[c] = overlap + 0.3 * substr
            # normalize affinities to probabilities
            total = sum(affin.values()) or 1.0
            U[a] = {c: affin[c] / total for c in consts}

        # compute M per source
        sources = df['source'].unique()
        M_list = []
        frame_rows = []
        for src in sources:
            src_df = df[df['source'] == src]
            # group by topic
            M_row = {c: 0.0 for c in consts}
            for a in range(n_topics):
                mask = src_df['primary_topic'] == a
                if mask.sum() == 0:
                    continue
                words_a = src_df.loc[mask, 'word_count']
                total_words_a = words_a.sum() or 1.0
                # compute San = sum_i Cian * (Sian - Savg(a)); Cian = words_i / total_words_a
                Sian_minus = src_df.loc[mask, 'vader_compound'] - aspect_sent[a]
                Cian = words_a / total_words_a
                San = (Cian.values * Sian_minus.values).sum()
                for c in consts:
                    M_row[c] += U[a][c] * San
            M_list.append([M_row[c] for c in consts])
            # store pro/anti
            row = {'source': src}
            for c in consts:
                val = M_row[c]
                row[f'pro_{c}'] = max(0, val)
                row[f'anti_{c}'] = max(0, -val)
            frame_rows.append(row)
        frame_df = pd.DataFrame(frame_rows).fillna(0)
        M = np.array(M_list)
        return frame_df, M, consts

    # -----------------------
    # EDA plots & saves
    # -----------------------
    def plot_basic(self):
        df = self.df
        safe_mkdir(self.out_dir)
        # articles per source
        plt.figure(figsize=(10,5))
        vc = df['source'].value_counts().nlargest(30)
        sns.barplot(x=vc.index, y=vc.values)
        plt.xticks(rotation=45, ha='right')
        plt.title("Articles per source")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "articles_per_source.png"), dpi=200)
        plt.close()

        # length distribution
        plt.figure(figsize=(8,4))
        sns.histplot(df['word_count'], bins=60, kde=True)
        plt.title("Word count distribution (cleaned)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "wordcount_distribution.png"), dpi=200)
        plt.close()

        # sentiment by source (violin)
        plt.figure(figsize=(12,6))
        top_src = df['source'].value_counts().nlargest(12).index
        sns.violinplot(x='source', y='vader_compound', data=df[df['source'].isin(top_src)], inner='quartile')
        plt.xticks(rotation=45, ha='right')
        plt.title("VADER sentiment by top sources")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "sentiment_by_source_violin.png"), dpi=200)
        plt.close()

    def plot_wordclouds(self, topics: List[dict], top_n: int = 6):
        # topic wordclouds
        for t in topics[:top_n]:
            words = " ".join(t['words'])
            wc = WordCloud(width=800, height=400, background_color='white').generate(words)
            plt.figure(figsize=(10,4))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title(f"Topic {t['topic_id']} wordcloud")
            plt.tight_layout()
            plt.savefig(os.path.join(self.out_dir, f"topic_{t['topic_id']}_wordcloud.png"), dpi=200)
            plt.close()

    def plot_frame_heatmap(self, M: np.ndarray, sources: List[str], consts: List[str]):
        if M.size == 0:
            return
        plt.figure(figsize=(10, max(4, len(sources) * 0.35)))
        sns.heatmap(M, cmap='RdBu_r', center=0, xticklabels=consts, yticklabels=sources, annot=False)
        plt.title("Frame alignment matrix M(n,c) (signed)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "frame_alignment_heatmap.png"), dpi=200)
        plt.close()

    def plot_pca(self, M: np.ndarray, sources: List[str]):
        if M.shape[0] < 2:
            return
        pca = PCA(n_components=2)
        X = pca.fit_transform(M)
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=X[:,0], y=X[:,1])
        for i, s in enumerate(sources):
            plt.text(X[i,0]+0.01, X[i,1]+0.01, s, fontsize=9)
        plt.title("PCA of sources from frame matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "frame_pca.png"), dpi=200)
        plt.close()

    def plot_coverage_distances(self, cov_df: pd.DataFrame):
        aspect_cols = [c for c in cov_df.columns if c.startswith('aspect_')]
        if not aspect_cols:
            return
        mean_vec = cov_df[aspect_cols].mean().values
        cov_df['euclidean_to_mean'] = cov_df[aspect_cols].apply(lambda r: euclidean(r.values, mean_vec), axis=1)
        cov_df['jsd_to_mean'] = cov_df[aspect_cols].apply(lambda r: jensen_shannon(r.values, mean_vec), axis=1)
        cov_df.to_csv(os.path.join(self.out_dir, "coverage_distances.csv"), index=False)

        plt.figure(figsize=(10,5))
        sns.barplot(x='source', y='euclidean_to_mean', data=cov_df.sort_values('euclidean_to_mean', ascending=False).head(30))
        plt.xticks(rotation=45, ha='right')
        plt.title("Top sources by Euclidean distance to mean coverage")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "euclidean_distance_to_mean.png"), dpi=200)
        plt.close()

    def plot_constituency_histograms(self):
        # compute constituency scores per article and aggregate by source
        cols = list(self.constituencies.keys())
        cscores = self.df['content_clean'].apply(lambda t: pd.Series(self.constituency_scores(t)))
        cscores.columns = [f'const_{c}' for c in cols]
        out = pd.concat([self.df[['source']], cscores], axis=1)
        agg = out.groupby('source').mean().reset_index()
        # melt and plot top sources
        top_src = self.df['source'].value_counts().nlargest(12).index
        agg_top = agg[agg['source'].isin(top_src)]
        melted = agg_top.melt(id_vars='source', var_name='const', value_name='score')
        plt.figure(figsize=(12,6))
        sns.barplot(x='source', y='score', hue='const', data=melted)
        plt.xticks(rotation=45, ha='right')
        plt.title("Average constituency alignment scores for top sources")
        plt.legend(bbox_to_anchor=(1.02,1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "constituency_alignment_by_source.png"), dpi=200)
        plt.close()

    def timeline_trends(self):
        if 'created_at' not in self.df.columns or self.df['created_at'].isna().all():
            return
        df = self.df.copy()
        df = df.dropna(subset=['created_at'])
        df['date'] = df['created_at'].dt.date
        # daily counts per event or source
        plt.figure(figsize=(12,5))
        daily = df.groupby('date').size().reset_index(name='count')
        sns.lineplot(x='date', y='count', data=daily)
        plt.xticks(rotation=45)
        plt.title("Daily article counts")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "daily_article_counts.png"), dpi=200)
        plt.close()

        # daily avg sentiment
        plt.figure(figsize=(12,5))
        daily_sent = df.groupby('date')['vader_compound'].mean().reset_index()
        sns.lineplot(x='date', y='vader_compound', data=daily_sent)
        plt.xticks(rotation=45)
        plt.title("Daily average VADER sentiment")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "daily_sentiment_trend.png"), dpi=200)
        plt.close()

    # -----------------------
    # Run pipeline
    # -----------------------
    def run(self, n_topics: int = 12, min_df: int = 5, max_features: int = 3000, assign_thresh: float = 0.25):
        # load and preprocess
        self.load()
        self.preprocess()

        # LDA
        docs = self.df['content_clean'].fillna("").tolist()
        print("Running LDA...")
        doc_topic, topics, lda_model, vectorizer = self.run_lda(docs, n_topics=n_topics, max_features=max_features, min_df=min_df)
        # assignments
        assignments = self.assign_topics(doc_topic, threshold=assign_thresh)
        self.df['topic_assignments'] = assignments
        # save topics
        topics_df = pd.DataFrame([{'topic_id': t['topic_id'], 'top_words': ",".join(t['words'])} for t in topics])
        topics_df.to_csv(os.path.join(self.out_dir, "lda_topics.csv"), index=False)

        # coverage
        cov_df = self.coverage_by_source(assignments, topics)
        cov_df.to_csv(os.path.join(self.out_dir, "aspect_coverage_by_source.csv"), index=False)

        # frame matrix
        print("Computing frame matrix...")
        frame_df, M, consts = self.compute_frame_matrix(doc_topic, topics)
        frame_df.to_csv(os.path.join(self.out_dir, "frame_df.csv"), index=False)
        np.save(os.path.join(self.out_dir, "frame_matrix.npy"), M)

        # plots
        print("Producing plots...")
        self.plot_basic()
        self.plot_wordclouds(topics, top_n=8)
        sources = list(self.df['source'].unique())
        self.plot_frame_heatmap(M, sources, consts)
        self.plot_pca(M, sources)
        self.plot_coverage_distances(cov_df)
        self.plot_constituency_histograms()
        self.timeline_trends()

        print("All outputs saved to:", self.out_dir)
        return {
            'df': self.df,
            'doc_topic': doc_topic,
            'topics': topics,
            'cov_df': cov_df,
            'frame_df': frame_df,
            'frame_matrix': M
        }

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing + intense EDA for scraped Indian news")
    parser.add_argument("--input", required=True, help="input directory or file (JSONL/JSON/CSV)")
    parser.add_argument("--outdir", default="./eda_outputs", help="output directory for CSVs/PNGs")
    parser.add_argument("--topics", type=int, default=12, help="number of LDA topics")
    parser.add_argument("--min_df", type=int, default=5, help="LDA min_df")
    parser.add_argument("--max_features", type=int, default=3000, help="LDA max features")
    parser.add_argument("--assign_thresh", type=float, default=0.25, help="topic assignment threshold")
    args = parser.parse_args()

    eda = PreprocessEDA(args.input, args.outdir)
    eda.run(n_topics=args.topics, min_df=args.min_df, max_features=args.max_features, assign_thresh=args.assign_thresh)
