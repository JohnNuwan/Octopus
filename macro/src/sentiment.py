"""Analyse de sentiment des actualités financières.

Utilise NLP (FinBERT, TextBlob, ou règles) pour analyser le sentiment
des news financières en temps réel.

Sources :
- Flux RSS Investing.com / Reuters / Bloomberg
- Analyse par lots sur les headlines
- Scoring : bullish / bearish / neutral lié à XAUUSD

Références:
    FinBERT — https://huggingface.co/ProsusAI/finbert
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import requests
from xml.etree import ElementTree

logger = logging.getLogger(__name__)


class SentimentLabel(Enum):
    """Label de sentiment."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class NewsArticle:
    """Article de news financière.

    Attributes:
        title: Titre de l'article.
        source: Source (Investing.com, Reuters, etc.).
        url: Lien vers l'article.
        published: Date de publication.
        summary: Résumé / extrait.
        sentiment: Score de sentiment (-1 à +1).
        gold_relevance: Pertinence pour l'or (0-1).
    """
    title: str
    source: str
    url: str
    published: datetime
    summary: str = ""
    sentiment: float = 0.0
    gold_relevance: float = 0.0


@dataclass
class SentimentResult:
    """Résultat d'analyse de sentiment agrégé.

    Attributes:
        overall_sentiment: Sentiment global (-1 à +1).
        bullish_count: Nombre d'articles bullish.
        bearish_count: Nombre d'articles bearish.
        neutral_count: Nombre d'articles neutres.
        gold_sentiment: Sentiment spécifique à l'or.
        top_articles: Articles les plus pertinents.
        timestamp: Horodatage de l'analyse.
    """
    overall_sentiment: float = 0.0
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    gold_sentiment: float = 0.0
    top_articles: List[NewsArticle] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


# Mots-clés XAUUSD pour le scoring de pertinence
GOLD_KEYWORDS = [
    "gold", "xau", "xauusd", "precious metal", "bullion",
    "fomc", "federal reserve", "fed", "inflation",
    "nonfarm payrolls", "nfp", "cpi", "gdp",
    "dollar", "dxy", "usd", "dollar index",
    "interest rate", "rate hike", "rate cut",
    "safe haven", "避险", "黄金",
    "central bank", "treasury yield", "real yield",
    "geopolitical", "conflict", "sanctions",
    "recession", "stagflation", "inflation",
]

# Mots-clés bullish pour l'or
BULLISH_GOLD_KEYWORDS = [
    "rate cut", "dovish", "weaker dollar",
    "inflation rises", "inflation higher",
    "geopolitical tension", "conflict escalates",
    "recession fears", "safe haven demand",
    "gold rises", "gold rallies", "gold surges",
    "central bank buying gold",
    "uncertainty", "volatility spikes",
    "tariff", "trade war",
]

# Mots-clés bearish pour l'or
BEARISH_GOLD_KEYWORDS = [
    "rate hike", "hawkish", "stronger dollar",
    "inflation falls", "inflation lower",
    "risk on", "risk appetite",
    "gold falls", "gold declines", "gold drops",
    "treasury yields rise", "yields surge",
    "economic recovery", "growth accelerates",
    "trade deal", "ceasefire",
    "dollar strengthens", "dollar rallies",
]


class RuleBasedSentiment:
    """Analyseur de sentiment basé sur des règles lexicales.

    Utilise des dictionnaires de mots-clés bullish/bearish
    pour scorer rapidement le sentiment sans modèle ML.
    """

    # Mots bullish général
    BULLISH_WORDS = {
        "surge", "rally", "soar", "jump", "climb", "rise", "gain",
        "bullish", "positive", "strong", "growth", "expansion",
        "improve", "recovery", "boost", "momentum", "outperform",
        "breakout", "upside", "optimistic", "upgrade",
    }

    # Mots bearish général
    BEARISH_WORDS = {
        "plunge", "crash", "slump", "drop", "fall", "decline", "loss",
        "bearish", "negative", "weak", "slowdown", "contraction",
        "deteriorate", "recession", "downturn", "risk", "fear",
        "breakdown", "downside", "pessimistic", "downgrade",
        "selloff", "correction", "panic",
    }

    @classmethod
    def score(cls, text: str) -> float:
        """Score de sentiment par règles lexicales.

        Args:
            text: Texte à analyser.

        Returns:
            Score entre -1 (bearish) et +1 (bullish).
        """
        words = set(re.findall(r'\w+', text.lower()))
        bullish_count = len(words & cls.BULLISH_WORDS)
        bearish_count = len(words & cls.BEARISH_WORDS)
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
        return (bullish_count - bearish_count) / total


class FinBERTSentiment:
    """Analyseur de sentiment via FinBERT (HuggingFace).

    Modèle BERT fine-tuné sur le texte financier.
    Fallback sur l'analyseur lexical si FinBERT n'est pas disponible.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert") -> None:
        """Initialise FinBERT.

        Args:
            model_name: Nom du modèle HuggingFace.
        """
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._available = False

    def _load(self) -> bool:
        """Charge le modèle FinBERT.

        Returns:
            True si le modèle est chargé avec succès.
        """
        if self._available:
            return True
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self._model.eval()
            self._available = True
            logger.info(f"✅ FinBERT chargé: {self.model_name}")
        except Exception as e:
            logger.warning(f"FinBERT non disponible: {e}. Utilisation règles lexicales.")
            self._available = False
        return self._available

    def score(self, text: str) -> float:
        """Analyse le sentiment d'un texte avec FinBERT.

        Args:
            text: Texte à analyser.

        Returns:
            Score entre -1 (bearish) et +1 (bullish).
        """
        if not self._load():
            return RuleBasedSentiment.score(text)

        try:
            import torch
            inputs = self._tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # FinBERT: 0=positive, 1=negative, 2=neutral
            pos = probs[0, 0].item()
            neg = probs[0, 1].item()
            return pos - neg  # -1 à +1
        except Exception as e:
            logger.warning(f"Erreur FinBERT inference: {e}")
            return RuleBasedSentiment.score(text)


class NewsSentimentAnalyzer:
    """Analyseur de sentiment des actualités financières.

    Agrège les news de plusieurs sources et calcule un score
    de sentiment global, avec un focus sur l'or (XAUUSD).

    Attributes:
        sentiment_engine: Moteur de sentiment (FinBERT ou règles).
        rss_sources: Sources RSS à surveiller.
        cache_duration: Durée de validité du cache.
    """

    DEFAULT_RSS_SOURCES = {
        "investing": "https://www.investing.com/rss/news.rss",
        "reuters_markets": "https://www.investing.com/rss/market_overview.rss",
        "forex": "https://www.investing.com/rss/forex.rss",
        "commodities": "https://www.investing.com/rss/commodities.rss",
    }

    def __init__(
        self,
        use_finbert: bool = True,
        rss_sources: Optional[Dict[str, str]] = None,
        max_articles: int = 50,
        cache_minutes: int = 15,
    ) -> None:
        """Initialise l'analyseur de sentiment.

        Args:
            use_finbert: Utiliser FinBERT (sinon règles lexicales).
            rss_sources: Sources RSS.
            max_articles: Nombre max d'articles à analyser.
            cache_minutes: Durée du cache.
        """
        self.sentiment_engine = (
            FinBERTSentiment() if use_finbert else RuleBasedSentiment()
        )
        self.rss_sources = rss_sources or self.DEFAULT_RSS_SOURCES
        self.max_articles = max_articles
        self.cache_duration = timedelta(minutes=cache_minutes)
        self._cache: Optional[SentimentResult] = None
        self._cache_time: Optional[datetime] = None

    def _compute_gold_relevance(self, text: str) -> float:
        """Calcule la pertinence d'un article pour l'or.

        Args:
            text: Texte de l'article.

        Returns:
            Score de pertinence (0-1).
        """
        text_lower = text.lower()
        matches = sum(
            1 for kw in GOLD_KEYWORDS if kw.lower() in text_lower
        )
        return min(matches / 5.0, 1.0)

    def _fetch_rss(self, url: str, source_name: str) -> List[NewsArticle]:
        """Récupère les articles depuis un flux RSS.

        Args:
            url: URL du flux RSS.
            source_name: Nom de la source.

        Returns:
            Liste des articles.
        """
        articles = []
        try:
            resp = requests.get(url, timeout=10, headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko)"
                )
            })
            resp.raise_for_status()

            root = ElementTree.fromstring(resp.content)
            for item in root.iter("item"):
                title = item.findtext("title", "")
                link = item.findtext("link", "")
                pub_date_str = item.findtext("pubDate", "")
                description = item.findtext("description", "")

                # Parser la date
                published = datetime.now()
                if pub_date_str:
                    try:
                        published = datetime.strptime(
                            pub_date_str[:25], "%a, %d %b %Y %H:%M:%S"
                        )
                    except (ValueError, TypeError):
                        pass

                articles.append(NewsArticle(
                    title=title,
                    source=source_name,
                    url=link,
                    published=published,
                    summary=description,
                ))

        except Exception as e:
            logger.warning(f"Erreur RSS {source_name}: {e}")

        return articles

    def analyze(self, force_refresh: bool = False) -> SentimentResult:
        """Analyse le sentiment des actualités.

        Args:
            force_refresh: Force le rafraîchissement.

        Returns:
            Résultat d'analyse de sentiment agrégé.
        """
        now = datetime.now()

        # Vérifier le cache
        if (
            not force_refresh
            and self._cache
            and self._cache_time
            and (now - self._cache_time) < self.cache_duration
        ):
            return self._cache

        # Récupérer les articles
        all_articles: List[NewsArticle] = []
        for name, url in self.rss_sources.items():
            articles = self._fetch_rss(url, name)
            all_articles.extend(articles)

        # Trier par date et limiter
        all_articles.sort(key=lambda a: a.published, reverse=True)
        all_articles = all_articles[:self.max_articles]

        # Analyser chaque article
        for article in all_articles:
            text = f"{article.title} {article.summary}"
            article.sentiment = self.sentiment_engine.score(text)
            article.gold_relevance = self._compute_gold_relevance(text)

        # Calculer les métriques agrégées
        if not all_articles:
            result = SentimentResult()
        else:
            # Sentiment pondéré par la pertinence or
            weighted_sentiments = [
                a.sentiment * a.gold_relevance
                for a in all_articles
            ]
            total_weight = sum(a.gold_relevance for a in all_articles) or 1.0

            overall = np.mean([a.sentiment for a in all_articles])
            gold_sent = sum(weighted_sentiments) / total_weight

            bullish = sum(1 for a in all_articles if a.sentiment > 0.2)
            bearish = sum(1 for a in all_articles if a.sentiment < -0.2)
            neutral = len(all_articles) - bullish - bearish

            # Top articles (les plus pertinents pour l'or)
            top = sorted(
                all_articles,
                key=lambda a: a.gold_relevance,
                reverse=True,
            )[:10]

            result = SentimentResult(
                overall_sentiment=float(overall),
                bullish_count=bullish,
                bearish_count=bearish,
                neutral_count=neutral,
                gold_sentiment=float(gold_sent),
                top_articles=top,
                timestamp=now,
            )

        self._cache = result
        self._cache_time = now
        logger.info(
            f"📰 Sentiment: overall={result.overall_sentiment:.2f} "
            f"gold={result.gold_sentiment:.2f} "
            f"({result.bullish_count}B/{result.bearish_count}S/{result.neutral_count}N)"
        )
        return result

    def get_gold_signal(self) -> Tuple[float, str]:
        """Obtient un signal de trading basé sur le sentiment.

        Returns:
            Tuple (signal -1 à +1, description).
        """
        result = self.analyze()
        gold = result.gold_sentiment

        if gold > 0.3:
            return (gold, f"🟢 Sentiment or positif ({gold:.2f})")
        elif gold < -0.3:
            return (gold, f"🔴 Sentiment or négatif ({gold:.2f})")
        else:
            return (gold, f"⚪ Sentiment or neutre ({gold:.2f})")