"""
agents/layer2_scoring.py
════════════════════════
Layer 2 — Intent Classification & Scoring Agent

Research basis:
  • Frontiers in AI — González-Flores et al. (2025), DOI: 10.3389/frai.2025.1554325
    "The relevance of lead prioritization: a B2B lead scoring model based on ML"
    → Gradient Boosting Classifier delivers best ROC AUC across 15 algorithms

  • arXiv:2410.01627 — Arora, Jain & Merugu (Amazon Research, 2024)
    "Intent Detection in the Age of LLMs"
    → Hybrid LLM + SetFit routing: within 2% LLM accuracy, 50% lower latency

Architecture:
  Stage 1 — GBC scores structured firmographic features
  Stage 2 — LLM intent router processes unstructured signals
  Output   — Composite score, tier (HOT/WARM/COLD), confidence, reasoning

Author : Dr. Debendra Ray, DBA — Independent AI Researcher
Licence: MIT
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Feature extraction ─────────────────────────────────────────────────────
def extract_features(prospect: dict, icp: dict) -> np.ndarray:
    """
    Extract structured features for the GBC model.
    Maps prospect attributes to numeric feature vector.

    Feature index map:
      0 — employee_fit        (1.0 if in ICP range, 0.5 if adjacent, 0.0 if outside)
      1 — industry_fit        (1.0 if exact match, 0.5 if adjacent)
      2 — geography_fit       (1.0 if in target list)
      3 — signal_strength     (from discovery agent, 0.0-1.0)
      4 — signal_count        (normalised: count / 5)
      5 — has_funding_signal  (binary)
      6 — has_hiring_signal   (binary)
      7 — has_tech_signal     (binary)
    """
    f = icp.get("firmographics", {})
    emp_range  = f.get("employee_range",  [50, 500])
    industries = [i.lower() for i in f.get("industry", [])]
    geos       = [g.lower() for g in f.get("geography", [])]

    emp = prospect.get("employee_estimate", 0)
    emp_fit = (
        1.0 if emp_range[0] <= emp <= emp_range[1]
        else 0.5 if emp_range[0] * 0.5 <= emp <= emp_range[1] * 1.5
        else 0.0
    )

    ind_fit = (
        1.0 if prospect.get("industry", "").lower() in industries
        else 0.5
    )

    geo_fit = (
        1.0 if any(g in prospect.get("location", "").lower() for g in geos)
        else 0.5
    )

    signals = prospect.get("signals", [])
    sig_text = " ".join(signals).lower()

    return np.array([
        emp_fit,
        ind_fit,
        geo_fit,
        float(prospect.get("signal_strength", 0.5)),
        min(len(signals) / 5.0, 1.0),
        1.0 if any(kw in sig_text for kw in ["funding", "series", "raised"]) else 0.0,
        1.0 if any(kw in sig_text for kw in ["hired", "hiring", "sdr", "head of"]) else 0.0,
        1.0 if any(kw in sig_text for kw in ["crm", "tech", "platform", "migration"]) else 0.0,
    ], dtype=float)


# ── GBC Model (lightweight fallback when sklearn model not trained) ─────────
class GradientBoostingScorer:
    """
    Production: loads a trained GBC model from disk.
    Demo/fallback: uses weighted feature scoring aligned with
    the González-Flores et al. (2025) feature importance findings —
    'source' and 'lead status' are top predictors.
    """

    WEIGHTS = np.array([0.20, 0.15, 0.10, 0.25, 0.10, 0.08, 0.07, 0.05])

    def __init__(self) -> None:
        self.model = self._load_model()

    def _load_model(self):
        model_path = os.getenv("GBC_MODEL_PATH", "data/models/gbc_lead_scorer.pkl")
        if os.path.exists(model_path):
            import pickle
            with open(model_path, "rb") as f:
                logger.info("Loaded GBC model from %s", model_path)
                return pickle.load(f)
        logger.info("No trained GBC model found — using weighted feature scoring")
        return None

    def score(self, features: np.ndarray) -> float:
        if self.model is not None:
            prob = self.model.predict_proba(features.reshape(1, -1))[0][1]
            return float(prob)
        return float(np.dot(features, self.WEIGHTS))


# ── LLM Intent Router ──────────────────────────────────────────────────────
class LLMIntentRouter:
    """
    Hybrid uncertainty-based router (arXiv:2410.01627).
    Uses LLM only for signals that the GBC classifier is uncertain about.
    Threshold: if GBC confidence 0.45-0.65 → route to LLM for enrichment.
    """

    UNCERTAINTY_BAND = (0.45, 0.65)

    def __init__(self) -> None:
        self.llm = self._init_llm()

    def _init_llm(self):
        provider = os.getenv("LLM_PROVIDER", "azure").lower()
        if provider == "azure":
            from langchain_openai import AzureChatOpenAI
            return AzureChatOpenAI(
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
                api_version="2024-02-01",
                temperature=0.0,
            )
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        else:
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "phi4-mini"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            )

    def needs_routing(self, gbc_score: float) -> bool:
        lo, hi = self.UNCERTAINTY_BAND
        return lo <= gbc_score <= hi

    def classify_intent(self, prospect: dict, gbc_score: float) -> dict:
        """Ask LLM to classify buying intent from unstructured signals."""
        from langchain.schema import HumanMessage
        prompt = f"""
        You are a B2B sales intent classifier.

        Prospect: {prospect.get('company_name')} — {prospect.get('industry')}
        Signals: {prospect.get('signals', [])}
        Initial score: {gbc_score:.2f}

        Classify this prospect's buying intent:
        - HOT: actively in a buying window, strong signals
        - WARM: interested but not yet actively buying
        - COLD: weak signals, poor timing

        Respond ONLY as JSON:
        {{"intent": "HOT|WARM|COLD", "confidence": 0.0-1.0, "reasoning": "..."}}
        """
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            import json, re
            match = re.search(r"\{.*\}", response.content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.warning("LLM intent routing failed: %s", e)
        return {"intent": "WARM", "confidence": 0.5, "reasoning": "LLM fallback"}


# ── Main scoring agent ─────────────────────────────────────────────────────
class IntentScoringAgent:
    """
    Two-stage scoring pipeline.
    Stage 1: GBC on structured features  (fast, deterministic)
    Stage 2: LLM intent router           (only for uncertain cases)
    """

    HOT_THRESHOLD  = 0.72
    WARM_THRESHOLD = 0.45

    def __init__(self, icp: dict) -> None:
        self.icp    = icp
        self.gbc    = GradientBoostingScorer()
        self.router = LLMIntentRouter()

    def run(self, prospects: list[dict]) -> list[dict]:
        scored = []
        for p in prospects:
            result = self._score_one(p)
            scored.append(result)
        return sorted(scored, key=lambda x: x["overall_score"], reverse=True)

    def _score_one(self, prospect: dict) -> dict:
        features  = extract_features(prospect, self.icp)
        gbc_score = self.gbc.score(features)

        # Hybrid routing: LLM only when GBC is uncertain
        llm_result = None
        if self.router.needs_routing(gbc_score):
            llm_result = self.router.classify_intent(prospect, gbc_score)
            # Blend scores: 60% GBC, 40% LLM confidence
            llm_conf   = llm_result.get("confidence", 0.5)
            final_score = (gbc_score * 0.6) + (llm_conf * 0.4)
        else:
            final_score = gbc_score

        tier = (
            "HOT"  if final_score >= self.HOT_THRESHOLD  else
            "WARM" if final_score >= self.WARM_THRESHOLD else
            "COLD"
        )

        return {
            **prospect,
            "overall_score":    round(final_score, 3),
            "gbc_score":        round(gbc_score, 3),
            "tier":             tier,
            "llm_routing_used": llm_result is not None,
            "llm_intent":       llm_result,
            "feature_vector":   features.tolist(),
        }
