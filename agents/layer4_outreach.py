"""
agents/layer4_outreach.py
═════════════════════════
Layer 4 — Personalised Outreach Agent (Hybrid ML + RAG)

Research basis: arXiv:2603.14173 — Shanivendra (2026)
"Hybrid Intent-Aware Personalization with Machine Learning and RAG-Enabled
Large Language Models for Financial Services Marketing"
→ ML classifier predicts: channel, timing, personalisation level, message framing
→ RAG-grounded LLM generates: unique, contextual outreach message per prospect

Two-stage architecture:
  Stage 1 (ML Classifier) — predicts optimal channel, timing, and framing
  Stage 2 (RAG + LLM)     — generates unique grounded message per prospect

Author : Dr. Debendra Ray, DBA — Independent AI Researcher
Licence: MIT
"""

from __future__ import annotations

import logging
import os
import random
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


# ── Stage 1: Channel / Timing / Framing Classifier ────────────────────────
class OutreachClassifier:
    """
    Lightweight rule-based classifier aligned with ML findings from
    Shanivendra (2026). In production: replace with trained sklearn pipeline.

    Predicts:
      - channel:              linkedin_inmail | email | phone | multi
      - optimal_day:          Monday-Friday
      - optimal_hour:         8-11 AM local time (highest B2B response rates)
      - personalisation_level: 1-5
      - message_framing:      pain_point | aspiration | social_proof | urgency
    """

    CHANNEL_RULES = {
        "HOT":  "multi",          # Phone + LinkedIn + Email sequence
        "WARM": "linkedin_inmail", # Start warm on LinkedIn
        "COLD": "email",           # Low-friction cold email
    }

    FRAMING_RULES = {
        "funding": "aspiration",    # Company just raised → frame around growth
        "hiring":  "pain_point",    # Hiring surge → headcount pain
        "crm":     "social_proof",  # CRM migration → reference customers
        "default": "pain_point",
    }

    OPTIMAL_TIMES = {
        "linkedin_inmail": ("Tuesday", "10:00 AM"),
        "email":           ("Wednesday", "9:00 AM"),
        "phone":           ("Thursday", "11:00 AM"),
        "multi":           ("Tuesday", "10:00 AM"),
    }

    def classify(self, lead: dict) -> dict:
        tier    = lead.get("tier", "WARM")
        signals = " ".join(lead.get("signals", [])).lower()
        score   = lead.get("overall_score", 0.5)

        channel = self.CHANNEL_RULES.get(tier, "email")

        # Framing based on dominant signal
        framing = self.FRAMING_RULES["default"]
        for kw, frame in self.FRAMING_RULES.items():
            if kw != "default" and kw in signals:
                framing = frame
                break

        day, hour = self.OPTIMAL_TIMES.get(channel, ("Tuesday", "10:00 AM"))
        p_level   = min(5, max(1, int(score * 5)))  # 1-5 from score

        return {
            "recommended_channel":    channel,
            "optimal_send_day":       day,
            "optimal_send_time":      hour,
            "personalisation_level":  p_level,
            "message_framing":        framing,
        }


# ── Stage 2: RAG-Grounded Message Generator ────────────────────────────────
class RAGOutreachGenerator:
    """
    Generates a unique, contextual outreach message per prospect.
    RAG step: retrieves similar successful outreach from vector store
    to ground the LLM generation (prevents generic output).
    """

    SYSTEM_PROMPT = """You are an expert B2B sales copywriter.
Write a highly personalised, concise outreach message.

Rules:
- Reference a SPECIFIC signal from the prospect's data (funding, hire, news)
- Do NOT use generic openers ("I hope this finds you well", "I noticed...")
- Maximum 120 words for the message body
- Subject line: under 8 words, no clickbait
- Sound human, not like an AI template
- End with ONE clear call-to-action
- Never mention the prospect's employee count or revenue estimate
"""

    def __init__(self) -> None:
        self.llm        = self._init_llm()
        self.vector_store = self._init_vector_store()

    def _init_llm(self):
        provider = os.getenv("LLM_PROVIDER", "azure").lower()
        if provider == "azure":
            from langchain_openai import AzureChatOpenAI
            return AzureChatOpenAI(
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
                api_version="2024-02-01", temperature=0.7)
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o", temperature=0.7)
        else:
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "phi4-mini"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=0.7)

    def _init_vector_store(self):
        """Load FAISS/ChromaDB vector store of successful outreach examples."""
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_openai import AzureOpenAIEmbeddings
            index_path = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
            if os.path.exists(index_path):
                embeddings = AzureOpenAIEmbeddings()
                return FAISS.load_local(index_path, embeddings,
                                        allow_dangerous_deserialization=True)
        except Exception as e:
            logger.debug("Vector store not available: %s", e)
        return None

    def _retrieve_examples(self, lead: dict, framing: str) -> str:
        """RAG retrieval: find similar successful outreach messages."""
        if self.vector_store is None:
            return ""
        query = f"{lead.get('industry')} {framing} {lead.get('signals', [''])[0]}"
        try:
            docs = self.vector_store.similarity_search(query, k=2)
            return "\n---\n".join(d.page_content for d in docs)
        except Exception:
            return ""

    def generate(self, lead: dict, classifier_output: dict) -> dict:
        from langchain.schema import HumanMessage, SystemMessage

        framing  = classifier_output.get("message_framing", "pain_point")
        channel  = classifier_output.get("recommended_channel", "email")
        examples = self._retrieve_examples(lead, framing)

        qual     = lead.get("qualification", {})
        signals  = lead.get("signals", [])
        top_signal = signals[0] if signals else "your recent company news"

        prompt = f"""
Prospect Details:
- Company: {lead.get('company_name')}
- Industry: {lead.get('industry')}
- Location: {lead.get('location')}
- Top signal: {top_signal}
- Pain point identified: {qual.get('bd_specialist_notes', 'Not specified')}
- Recommended CTA: {qual.get('recommended_cta', 'Schedule a call')}

Outreach parameters:
- Channel: {channel}
- Message framing: {framing}
- Personalisation level: {classifier_output.get('personalisation_level', 3)}/5

{f"Reference examples of successful similar outreach:{chr(10)}{examples}" if examples else ""}

Write:
1. Subject line (under 8 words)
2. Message body (under 120 words)

Respond ONLY as JSON:
{{"subject": "...", "body": "..."}}
"""
        try:
            response = self.llm.invoke([
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            import json, re
            match = re.search(r"\{.*\}", response.content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.warning("Outreach generation failed: %s", e)

        # Fallback
        return {
            "subject": f"Quick question — {lead.get('company_name', 'your company')}",
            "body": f"Hi, I noticed {top_signal}. I'd love to share how we've helped similar companies. Would you have 20 minutes this week?",
        }


# ── Main outreach agent ────────────────────────────────────────────────────
class OutreachPersonalisationAgent:
    """
    Orchestrates both stages:
    1. Classify optimal channel/timing/framing (ML)
    2. Generate unique RAG-grounded message (LLM)
    """

    def __init__(self, icp: dict) -> None:
        self.icp        = icp
        self.classifier = OutreachClassifier()
        self.generator  = RAGOutreachGenerator()

    def run(self, qualified_leads: list[dict]) -> list[dict]:
        ready = []
        for lead in qualified_leads:
            enriched = self._process_one(lead)
            ready.append(enriched)
        return ready

    def _process_one(self, lead: dict) -> dict:
        classification = self.classifier.classify(lead)
        message        = self.generator.generate(lead, classification)

        return {
            **lead,
            "outreach": {
                **classification,
                "generated_subject":  message.get("subject", ""),
                "generated_body":     message.get("body", ""),
                "generated_at":       datetime.utcnow().isoformat(),
                "rag_grounded":       self.generator.vector_store is not None,
            },
        }
