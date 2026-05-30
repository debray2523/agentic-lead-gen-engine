"""
agents/layer5_evaluator.py
══════════════════════════
Layer 5 — LLM-as-Judge Evaluator (REEMD Self-Optimising Loop)

Research basis: arXiv:2412.17149 — Yuksel & Sawaf (2024)
"A Multi-AI Agent System for Autonomous Optimization of Agentic AI Solutions
via Iterative Refinement and LLM-Driven Feedback Loops"

REEMD = Refinement → Execution → Evaluation → Modification → Documentation

A separate judge LLM scores every output for:
  • Business alignment score  (target: > 91%)
  • Data accuracy score       (target: > 90%)
  • Personalisation quality   (1-5 Likert)
  • ICP fit confidence        (0.0-1.0)

If scores fall below threshold → pipeline auto-adjusts configuration.
Every change is documented (Documentation agent role).

Author : Dr. Debendra Ray, DBA — Independent AI Researcher
Licence: MIT
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are an expert B2B sales quality evaluator.
Your role is to objectively score AI-generated lead qualification and
outreach content for quality, accuracy, and business alignment.

Be strict and precise. Base scores only on what is actually present.
Do NOT inflate scores. The system will use your scores to self-improve.
"""


class LLMJudgeEvaluator:
    """
    Separate judge LLM (intentionally uses a different/cheaper model
    than the generation LLM to avoid self-congratulatory scoring).
    """

    ALIGNMENT_THRESHOLD    = float(os.getenv("ALIGNMENT_THRESHOLD", "0.80"))
    ACCURACY_THRESHOLD     = float(os.getenv("ACCURACY_THRESHOLD",  "0.85"))

    def __init__(self) -> None:
        self.llm      = self._init_judge_llm()
        self.run_id   = str(uuid.uuid4())[:8]
        self.log_path = os.path.join(
            os.getenv("OUTPUT_DIR", "output"),
            f"eval_log_{self.run_id}.jsonl"
        )
        os.makedirs(os.path.dirname(self.log_path) if os.path.dirname(self.log_path) else ".", exist_ok=True)

    def _init_judge_llm(self):
        """
        Use a cheaper/faster model for the judge role — intentional.
        GPT-4o-mini or phi4-mini is sufficient for scoring.
        """
        provider = os.getenv("LLM_PROVIDER", "azure").lower()
        judge_model = os.getenv("LLM_JUDGE_MODEL", "gpt-4o-mini")

        if provider == "azure":
            from langchain_openai import AzureChatOpenAI
            return AzureChatOpenAI(
                azure_deployment=judge_model,
                api_version="2024-02-01",
                temperature=0.0,
            )
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=judge_model, temperature=0.0)
        else:
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "phi4-mini"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=0.0,
            )

    # ── Public API ─────────────────────────────────────────────────────────
    def run(self, leads: list[dict]) -> tuple[list[dict], dict]:
        """
        Evaluate all leads and return (evaluated_leads, run_metadata).
        run_metadata includes aggregate scores for the REEMD loop.
        """
        evaluated = []
        scores = {"alignment": [], "accuracy": [], "personalisation": [], "icp_fit": []}

        for lead in leads:
            result = self._evaluate_one(lead)
            evaluated.append(result)
            e = result["evaluation"]
            scores["alignment"].append(e["alignment_score"])
            scores["accuracy"].append(e["accuracy_score"])
            scores["personalisation"].append(e["personalisation_quality"])
            scores["icp_fit"].append(e["icp_fit_confidence"])
            self._log_evaluation(result)

        avg = {k: round(sum(v) / len(v), 3) if v else 0.0
               for k, v in scores.items()}

        passed = sum(1 for l in evaluated if l["evaluation"]["judge_passed"])
        meta = {
            "run_id":            self.run_id,
            "evaluated_at":      datetime.utcnow().isoformat(),
            "total_leads":       len(evaluated),
            "judge_passed":      passed,
            "pass_rate":         round(passed / max(len(evaluated), 1), 3),
            "avg_alignment":     avg["alignment"],
            "avg_accuracy":      avg["accuracy"],
            "avg_personalisation": avg["personalisation"],
            "avg_icp_fit":       avg["icp_fit"],
            "reemd_triggered":   avg["alignment"] < self.ALIGNMENT_THRESHOLD,
        }

        if meta["reemd_triggered"]:
            logger.warning(
                "REEMD loop triggered — avg alignment %.2f below threshold %.2f",
                avg["alignment"], self.ALIGNMENT_THRESHOLD
            )

        return evaluated, meta

    # ── Single lead evaluation ─────────────────────────────────────────────
    def _evaluate_one(self, lead: dict) -> dict:
        from langchain.schema import HumanMessage, SystemMessage

        outreach = lead.get("outreach", {})
        qual     = lead.get("qualification", {})

        prompt = f"""
Evaluate this lead qualification and outreach output.

LEAD:
- Company: {lead.get('company_name')} ({lead.get('industry')})
- Tier: {lead.get('tier')}  |  Score: {lead.get('overall_score')}
- Signals: {lead.get('signals', [])}

QUALIFICATION:
- Market analysis: {qual.get('market_analyst_notes', '')}
- BD recommendation: {qual.get('bd_specialist_notes', '')}
- Data status: {qual.get('data_validator_status', '')}
- Fit score: {qual.get('fit_score', 0)}

OUTREACH GENERATED:
- Channel: {outreach.get('recommended_channel', '')}
- Subject: {outreach.get('generated_subject', '')}
- Body: {outreach.get('generated_body', '')}

Score each dimension strictly (0.0 to 1.0 unless noted):

1. alignment_score     — Does the outreach align with the qualification analysis?
2. accuracy_score      — Is the data accurate and consistent throughout?
3. personalisation_quality — How personalised is the message? (1=generic, 5=highly specific)
4. icp_fit_confidence  — How well does this lead fit the ICP?
5. judge_passed        — Does this lead meet quality bar? (true/false)
6. improvement_notes   — One specific improvement if judge_passed=false, else "None"

Respond ONLY as JSON:
{{
  "alignment_score": 0.0,
  "accuracy_score": 0.0,
  "personalisation_quality": 1,
  "icp_fit_confidence": 0.0,
  "judge_passed": false,
  "improvement_notes": "..."
}}
"""
        try:
            response = self.llm.invoke([
                SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            match = re.search(r"\{.*\}", response.content, re.DOTALL)
            if match:
                scores = json.loads(match.group())
            else:
                scores = self._default_scores()
        except Exception as e:
            logger.warning("Judge LLM failed for %s: %s", lead.get("company_name"), e)
            scores = self._default_scores()

        # Override judge_passed with hard thresholds
        scores["judge_passed"] = (
            scores.get("alignment_score", 0) >= self.ALIGNMENT_THRESHOLD and
            scores.get("accuracy_score",  0) >= self.ACCURACY_THRESHOLD
        )

        return {
            **lead,
            "evaluation": {
                **scores,
                "evaluated_at": datetime.utcnow().isoformat(),
                "run_id":       self.run_id,
            },
        }

    def _default_scores(self) -> dict:
        return {
            "alignment_score":        0.70,
            "accuracy_score":         0.70,
            "personalisation_quality": 3,
            "icp_fit_confidence":     0.65,
            "judge_passed":           False,
            "improvement_notes":      "Judge LLM unavailable — manual review needed",
        }

    # ── REEMD Documentation ────────────────────────────────────────────────
    def _log_evaluation(self, result: dict) -> None:
        """Documentation agent role: log every evaluation to JSONL audit trail."""
        record = {
            "timestamp":    datetime.utcnow().isoformat(),
            "run_id":       self.run_id,
            "company":      result.get("company_name"),
            "tier":         result.get("tier"),
            "scores":       result.get("evaluation", {}),
        }
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.debug("Eval log write failed: %s", e)

    def generate_run_report(self, meta: dict) -> str:
        """Generate a human-readable run summary report."""
        lines = [
            "═" * 60,
            f"  PIPELINE EVALUATION REPORT — Run {meta.get('run_id', 'N/A')}",
            "═" * 60,
            f"  Total leads evaluated : {meta.get('total_leads', 0)}",
            f"  Judge passed          : {meta.get('judge_passed', 0)} ({meta.get('pass_rate', 0)*100:.1f}%)",
            f"  Avg alignment score   : {meta.get('avg_alignment', 0):.2f}  (target ≥ {self.ALIGNMENT_THRESHOLD})",
            f"  Avg accuracy score    : {meta.get('avg_accuracy', 0):.2f}  (target ≥ {self.ACCURACY_THRESHOLD})",
            f"  Avg personalisation   : {meta.get('avg_personalisation', 0):.1f}/5",
            f"  Avg ICP fit           : {meta.get('avg_icp_fit', 0):.2f}",
            f"  REEMD loop triggered  : {'YES — pipeline will retry' if meta.get('reemd_triggered') else 'No'}",
            "═" * 60,
        ]
        report = "\n".join(lines)
        logger.info(report)
        return report
