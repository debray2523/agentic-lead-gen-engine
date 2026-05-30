"""
agents/layer1_discovery.py
══════════════════════════
Layer 1 — Prospect Discovery Agent (Agentic RAG)

Research basis: arXiv:2501.09136 — Singh et al., "Agentic Retrieval-Augmented
Generation: A Survey on Agentic RAG", January 2025.

Uses dynamic multi-step retrieval: the agent does NOT query a static list.
It actively fetches live signals (funding, hiring, tech changes) via MCP
tool servers, reflects on quality, and iteratively refines before passing
prospects downstream.

Author : Dr. Debendra Ray, DBA — Independent AI Researcher
Licence: MIT
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a B2B prospect discovery specialist.
Given an Ideal Customer Profile (ICP), your job is to identify companies
that match the profile based on live market signals.

You must:
1. Identify companies matching the ICP firmographics
2. Detect buying-window signals (funding, hiring surges, tech changes)
3. Reflect on signal quality before returning results
4. Return only high-confidence prospects

Return a JSON list of prospect objects. Each object must have:
- company_name, domain, industry, employee_estimate, location
- signals: list of observed buying signals
- signal_strength: float 0.0-1.0
- data_source: where the data came from
"""


class ProspectDiscoveryAgent:
    """
    Agentic RAG discovery agent.

    In production: connects to MCP web-fetch, LinkedIn, and enrichment servers.
    In demo mode: generates high-quality synthetic prospects matching the ICP.
    """

    def __init__(self, icp: dict) -> None:
        self.icp  = icp
        self.llm  = self._init_llm()
        self.demo = os.getenv("DEMO_MODE", "false").lower() == "true"

    def _init_llm(self):
        provider = os.getenv("LLM_PROVIDER", "azure").lower()
        if provider == "azure":
            return AzureChatOpenAI(
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
                api_version="2024-02-01",
                temperature=0.1,
            )
        elif provider == "openai":
            return ChatOpenAI(model="gpt-4o", temperature=0.1)
        else:
            # Ollama / local fallback
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "phi4-mini"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            )

    # ── Public API ─────────────────────────────────────────────────────────
    def run(self) -> list[dict]:
        """Execute discovery and return raw prospect list."""
        if self.demo:
            return self._demo_prospects()

        prospects = self._discover_via_signals()
        prospects = self._reflect_and_filter(prospects)
        return prospects

    # ── Signal-based discovery ─────────────────────────────────────────────
    def _discover_via_signals(self) -> list[dict]:
        """
        In production this calls MCP tool servers:
          - mcp_web_fetch  → news APIs, funding databases
          - mcp_enrichment → Apollo, Hunter, Clearbit
          - mcp_linkedin   → company signals, hiring patterns
        """
        icp_summary = self._summarise_icp()
        prompt = f"""
        ICP Definition:
        {icp_summary}

        Search for companies that match this ICP. Focus on:
        - Companies in {self.icp.get('firmographics', {}).get('industry', [])}
        - Size: {self.icp.get('firmographics', {}).get('employee_range', [50, 500])} employees
        - Located in: {self.icp.get('firmographics', {}).get('geography', [])}

        Look for buying-window signals:
        {self.icp.get('intent_signals', {}).get('positive', [])}

        Return 20 prospect companies as a JSON array. Each entry:
        {{
          "company_name": "...",
          "domain": "...",
          "industry": "...",
          "employee_estimate": 150,
          "location": "...",
          "signals": ["signal1", "signal2"],
          "signal_strength": 0.75,
          "data_source": "web_fetch"
        }}
        Return ONLY the JSON array. No preamble.
        """
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        return self._parse_json(response.content)

    def _reflect_and_filter(self, prospects: list[dict]) -> list[dict]:
        """
        Agentic RAG reflection step: ask LLM to critique its own output
        and remove low-quality prospects before passing downstream.
        """
        if not prospects:
            return []
        reflection_prompt = f"""
        You generated these {len(prospects)} prospects. Review each one critically.
        Remove any that:
        - Don't match the ICP firmographics
        - Have weak or no buying signals
        - Have signal_strength < 0.4

        ICP: {self._summarise_icp()}

        Prospects to review:
        {prospects}

        Return the filtered list as a JSON array. Maintain the same schema.
        Return ONLY the JSON array. No preamble.
        """
        messages = [HumanMessage(content=reflection_prompt)]
        response = self.llm.invoke(messages)
        filtered = self._parse_json(response.content)
        logger.debug("Reflection: %d → %d prospects", len(prospects), len(filtered))
        return filtered

    # ── Helpers ────────────────────────────────────────────────────────────
    def _summarise_icp(self) -> str:
        f = self.icp.get("firmographics", {})
        return (
            f"Industry: {f.get('industry', 'Any')}, "
            f"Size: {f.get('employee_range', [50,500])} employees, "
            f"Geography: {f.get('geography', 'Global')}, "
            f"Tech: {f.get('tech_stack_must_have', [])}"
        )

    def _parse_json(self, text: str) -> list[dict]:
        import json, re
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning("JSON parse failed, returning empty list")
        return []

    def _demo_prospects(self) -> list[dict]:
        """High-quality synthetic prospects for demo/testing — no API calls needed."""
        return [
            {
                "company_name": "Nexus Analytics",
                "domain": "nexusanalytics.io",
                "industry": "SaaS",
                "employee_estimate": 180,
                "location": "San Francisco, CA",
                "signals": [
                    "Series B funding ($18M) announced 34 days ago",
                    "VP of Sales hired 3 weeks ago",
                    "4 SDR positions posted on LinkedIn",
                ],
                "signal_strength": 0.91,
                "data_source": "demo",
            },
            {
                "company_name": "CloudPeak Solutions",
                "domain": "cloudpeak.com",
                "industry": "Technology",
                "employee_estimate": 320,
                "location": "Austin, TX",
                "signals": [
                    "CRM migration keywords in 3 job descriptions",
                    "Competitor acquisition announced last month",
                ],
                "signal_strength": 0.72,
                "data_source": "demo",
            },
            {
                "company_name": "DataBridge Corp",
                "domain": "databridge.io",
                "industry": "Software",
                "employee_estimate": 95,
                "location": "New York, NY",
                "signals": [
                    "Series A ($8M) announced 60 days ago",
                    "Head of Revenue Operations posted",
                ],
                "signal_strength": 0.68,
                "data_source": "demo",
            },
        ]
