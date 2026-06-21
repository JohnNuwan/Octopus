"""
Agent de recherche académique Octopus.

Cherche les derniers papiers arXiv sur les topics pertinents,
analyse leurs contributions et suggère des améliorations
pour l'architecture d'Octopus.
"""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Tentative d'import hermes_tools (optionnel)
try:
    from hermes_tools import web_search, web_extract
    HERMES_TOOLS_DISPONIBLES = True
except ImportError:
    HERMES_TOOLS_DISPONIBLES = False

# Tentative d'import httpx (optionnel)
try:
    import httpx
    HTTPX_DISPONIBLE = True
except ImportError:
    HTTPX_DISPONIBLE = False


TOPICS_PAR_DEFAUT = [
    "world models trading",
    "JEPA reinforcement learning",
    "MCTS finance",
    "deep reinforcement learning portfolio management",
]


class ResearchAgent:
    """Agent de recherche de papiers académiques.

    Attributes:
        topics: Liste des sujets de recherche.
        max_papers: Nombre maximum de papiers par recherche.
    """

    def __init__(self, topics: Optional[List[str]] = None,
                 max_papers: int = 5):
        """Initialise l'agent de recherche.

        Args:
            topics: Sujets à rechercher (sinon défaut).
            max_papers: Nombre max de papiers par sujet.
        """
        self.topics = topics or TOPICS_PAR_DEFAUT
        self.max_papers = max_papers

    def search_arxiv(self, topics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Cherche les derniers papiers sur arXiv.

        Utilise hermes_tools si disponible, sinon l'API arXiv via httpx.

        Args:
            topics: Sujets à rechercher (sinon utilise ceux par défaut).

        Returns:
            Liste des papiers trouvés avec titre, résumé, url.
        """
        topics = topics or self.topics
        papers = []

        if HERMES_TOOLS_DISPONIBLES:
            papers = self._search_via_hermes(topics)
        elif HTTPX_DISPONIBLE:
            papers = self._search_via_arxiv_api(topics)
        else:
            logger.warning(
                "Aucun moteur de recherche disponible. "
                "Installer hermes_tools ou httpx."
            )

        # Déduplication
        seen = set()
        unique_papers = []
        for p in papers:
            key = p.get("title", p.get("id", ""))
            if key not in seen:
                seen.add(key)
                unique_papers.append(p)

        return unique_papers[:self.max_papers * len(topics)]

    def analyze_paper(self, arxiv_id: str) -> Dict[str, Any]:
        """Analyse et résume un papier arXiv spécifique.

        Args:
            arxiv_id: ID arXiv du papier (ex: '2304.12345').

        Returns:
            Dict avec résumé, contributions et pertinence pour Octopus.
        """
        paper = {"id": arxiv_id, "summary": "", "contributions": [],
                 "relevance": 0}

        if HERMES_TOOLS_DISPONIBLES:
            try:
                content = web_extract(f"https://arxiv.org/abs/{arxiv_id}")
                paper["summary"] = self._extract_abstract(content)
                paper["contributions"] = self._extract_contributions(content)
                paper["relevance"] = self._score_relevance(
                    paper["summary"] + " " + " ".join(paper["contributions"])
                )
            except Exception as e:
                logger.error("Erreur analyse papier %s: %s", arxiv_id, e)
        elif HTTPX_DISPONIBLE:
            try:
                resp = httpx.get(
                    f"https://export.arxiv.org/api/query?id_list={arxiv_id}",
                    timeout=15
                )
                paper["summary"] = self._parse_arxiv_xml_summary(resp.text)
                paper["relevance"] = self._score_relevance(paper["summary"])
            except Exception as e:
                logger.error("Erreur API arXiv %s: %s", arxiv_id, e)

        return paper

    def suggest_improvements(self, papers: List[Dict[str, Any]]) -> List[str]:
        """Propose des améliorations pour Octopus à partir des papiers.

        Args:
            papers: Liste des papiers analysés.

        Returns:
            Liste de suggestions d'amélioration.
        """
        improvements = []
        keywords_improvements = {
            "world model": "Intégrer un module World Model pour la prédiction d'état latent",
            "jepa": "Remplacer l'encodeur par une architecture JEPA pour meilleure généralisation",
            "mcts": "Remplacer epsilon-greedy par MCTS pour l'exploration",
            "transformer": "Ajouter des couches Transformer dans le backbone du réseau",
            "attention": "Implémenter un mécanisme d'attention temporelle",
            "macro": "Fusionner les features macro dans l'observation du RL",
            "reward shaping": "Utiliser le reward shaping pour accélérer la convergence",
            "curriculum": "Implémenter un curriculum learning pour l'entraînement",
            "ensemble": "Utiliser un ensemble de modèles pour réduire la variance",
            "distributional": "Passer à un RL distributionnel pour meilleure estimation du risque",
        }

        for paper in papers:
            text = (paper.get("title", "") + " " +
                    paper.get("summary", "") + " " +
                    " ".join(paper.get("contributions", []))).lower()

            for keyword, suggestion in keywords_improvements.items():
                if keyword in text and suggestion not in improvements:
                    improvements.append(suggestion)

        return improvements

    def run(self, **kwargs) -> Dict[str, Any]:
        """Point d'entrée principal (interface AgentManager).

        Returns:
            Rapport complet avec papiers trouvés et améliorations.
        """
        papers = self.search_arxiv()
        analyzed = []
        for p in papers[:self.max_papers]:
            if "id" in p:
                analyzed.append(self.analyze_paper(p["id"]))
            else:
                analyzed.append(p)

        improvements = self.suggest_improvements(analyzed)

        summary_lines = [
            f"🔬 Research Agent — {len(analyzed)} papiers analysés",
        ]
        for i, p in enumerate(analyzed[:3], 1):
            title = p.get("title", p.get("id", f"Papier {i}"))
            relevance = p.get("relevance", 0)
            star = "⭐" if relevance > 5 else "📄"
            summary_lines.append(f"  {star} {i}. {title} (pertinence: {relevance}/10)")

        if improvements:
            summary_lines.append("\n💡 Améliorations suggérées:")
            for imp in improvements[:5]:
                summary_lines.append(f"  • {imp}")

        return {
            "summary": "\n".join(summary_lines),
            "metrics": {
                "papers_found": len(papers),
                "papers_analyzed": len(analyzed),
                "improvements_found": len(improvements),
            },
            "papers_found": analyzed,
            "improvements": improvements,
        }

    def _search_via_hermes(self, topics: List[str]) -> List[Dict[str, Any]]:
        """Recherche via hermes_tools (web_search + web_extract)."""
        papers = []
        for topic in topics:
            try:
                query = f"arxiv {topic} 2025 2026"
                results = web_search(query, max_results=self.max_papers)
                for r in results:
                    papers.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "summary": r.get("snippet", ""),
                        "id": self._extract_arxiv_id(r.get("url", "")),
                    })
            except Exception as e:
                logger.error("Erreur hermes_tools pour '%s': %s", topic, e)
        return papers

    def _search_via_arxiv_api(self, topics: List[str]) -> List[Dict[str, Any]]:
        """Recherche via l'API REST d'ArXiv."""
        papers = []
        for topic in topics:
            try:
                query = re.sub(r"[^a-zA-Z0-9\s]", "", topic)
                query = "+AND+".join(query.split())
                url = (
                    f"https://export.arxiv.org/api/query?"
                    f"search_query=all:{query}"
                    f"&sortBy=submittedDate&sortOrder=descending"
                    f"&max_results={self.max_papers}"
                )
                resp = httpx.get(url, timeout=15)
                papers.extend(self._parse_arxiv_xml(resp.text))
            except Exception as e:
                logger.error("Erreur API arXiv pour '%s': %s", topic, e)
        return papers

    def _parse_arxiv_xml(self, xml_text: str) -> List[Dict[str, Any]]:
        """Parse la réponse XML de l'API arXiv."""
        import xml.etree.ElementTree as ET
        ns = {"atom": "http://www.w3.org/2005/Atom",
              "arxiv": "http://arxiv.org/schemas/atom"}
        papers = []
        try:
            root = ET.fromstring(xml_text)
            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns)
                summary = entry.find("atom:summary", ns)
                link = entry.find("atom:id", ns)
                papers.append({
                    "title": title.text.strip() if title is not None else "",
                    "summary": summary.text.strip()[:500] if summary is not None else "",
                    "url": link.text.strip() if link is not None else "",
                    "id": self._extract_arxiv_id(
                        link.text.strip() if link is not None else ""
                    ),
                })
        except ET.ParseError as e:
            logger.error("Erreur parse XML arXiv: %s", e)
        return papers

    def _parse_arxiv_xml_summary(self, xml_text: str) -> str:
        """Extrait le résumé d'un papier depuis la réponse XML."""
        import xml.etree.ElementTree as ET
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        try:
            root = ET.fromstring(xml_text)
            summary = root.find(".//atom:summary", ns)
            return summary.text.strip()[:1000] if summary is not None else ""
        except ET.ParseError:
            return ""

    @staticmethod
    def _extract_arxiv_id(url: str) -> str:
        """Extrait l'ID arXiv d'une URL."""
        match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)", url)
        return match.group(1) if match else ""

    @staticmethod
    def _extract_abstract(content: str) -> str:
        """Extrait l'abstract du contenu HTML d'une page arXiv."""
        match = re.search(
            r'<blockquote class="abstract[^"]*"[^>]*>(.*?)</blockquote>',
            content, re.DOTALL
        )
        if match:
            return re.sub(r"<[^>]+>", "", match.group(1)).strip()[:1000]
        return ""

    @staticmethod
    def _extract_contributions(content: str) -> List[str]:
        """Extrait les contributions potentielles du contenu."""
        contributions = []
        sections = re.split(r"<h[1-3][^>]*>", content)
        for section in sections:
            if any(kw in section.lower() for kw in
                   ["contribution", "method", "approach", "novel",
                    "introduction"]):
                text = re.sub(r"<[^>]+>", "", section)
                contributions.append(text[:300])
        return contributions[:3]

    @staticmethod
    def _score_relevance(text: str) -> int:
        """Score de pertinence pour Octopus (0-10)."""
        keywords = {
            "reinforcement learning": 3, "rl": 2,
            "world model": 3, "jepa": 3, "mcts": 3,
            "trading": 2, "finance": 2, "portfolio": 2,
            "deep learning": 1, "neural network": 1,
            "attention": 1, "transformer": 1,
            "market": 1, "forex": 2, "xauusd": 3,
            "reward": 2, "policy gradient": 2,
            "multi-agent": 2, "hierarchical": 1,
        }
        text_lower = text.lower()
        score = sum(weight for keyword, weight in keywords.items()
                    if keyword in text_lower)
        return min(score, 10)