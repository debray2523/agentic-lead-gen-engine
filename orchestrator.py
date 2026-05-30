"""
orchestrator.py
═══════════════
LangGraph-based orchestrator for the Agentic Lead Generation Engine.
Manages state, workflow graph, and coordinates all 5 agent layers.

Author : Dr. Debendra Ray, DBA — Independent AI Researcher
Licence: MIT
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agents.layer1_discovery   import ProspectDiscoveryAgent
from agents.layer2_scoring     import IntentScoringAgent
from agents.layer3_qualification import QualificationCrew
from agents.layer4_outreach    import OutreachPersonalisationAgent
from agents.layer5_evaluator   import LLMJudgeEvaluator
from tools.icp_loader          import load_icp
from tools.output_writer       import write_output

logger = logging.getLogger(__name__)


# ── Shared pipeline state ──────────────────────────────────────────────────
class PipelineState(TypedDict):
    icp:            dict
    raw_prospects:  list[dict]
    scored_leads:   list[dict]
    qualified_leads: list[dict]
    outreach_ready: list[dict]
    evaluated:      list[dict]
    run_metadata:   dict
    iteration:      int


# ── Node functions (one per layer) ────────────────────────────────────────
def discover(state: PipelineState) -> PipelineState:
    logger.info("▶ Layer 1 — Prospect Discovery")
    agent = ProspectDiscoveryAgent(icp=state["icp"])
    state["raw_prospects"] = agent.run()
    logger.info("  Found %d raw prospects", len(state["raw_prospects"]))
    return state


def score(state: PipelineState) -> PipelineState:
    logger.info("▶ Layer 2 — Intent Scoring")
    agent = IntentScoringAgent(icp=state["icp"])
    state["scored_leads"] = agent.run(state["raw_prospects"])
    hot = sum(1 for l in state["scored_leads"] if l["tier"] == "HOT")
    warm = sum(1 for l in state["scored_leads"] if l["tier"] == "WARM")
    logger.info("  Scored: %d HOT  %d WARM", hot, warm)
    return state


def qualify(state: PipelineState) -> PipelineState:
    logger.info("▶ Layer 3 — Qualification Crew (A2A)")
    crew = QualificationCrew(icp=state["icp"])
    hot_warm = [l for l in state["scored_leads"] if l["tier"] in ("HOT", "WARM")]
    state["qualified_leads"] = crew.run(hot_warm)
    logger.info("  Qualified %d leads", len(state["qualified_leads"]))
    return state


def personalise(state: PipelineState) -> PipelineState:
    logger.info("▶ Layer 4 — Personalised Outreach")
    agent = OutreachPersonalisationAgent(icp=state["icp"])
    state["outreach_ready"] = agent.run(state["qualified_leads"])
    logger.info("  Generated outreach for %d leads", len(state["outreach_ready"]))
    return state


def evaluate(state: PipelineState) -> PipelineState:
    logger.info("▶ Layer 5 — LLM-as-Judge (REEMD)")
    judge = LLMJudgeEvaluator()
    state["evaluated"], meta = judge.run(state["outreach_ready"])
    state["run_metadata"].update(meta)
    state["iteration"] += 1
    passed = sum(1 for l in state["evaluated"] if l["evaluation"]["judge_passed"])
    logger.info("  Evaluation: %d/%d passed", passed, len(state["evaluated"]))
    return state


def should_retry(state: PipelineState) -> str:
    """REEMD loop: retry once if average alignment score below threshold."""
    if state["iteration"] >= 2:
        return "write_output"
    avg = (
        sum(l["evaluation"]["alignment_score"] for l in state["evaluated"])
        / max(len(state["evaluated"]), 1)
    )
    return "personalise" if avg < 0.80 else "write_output"


def write(state: PipelineState) -> PipelineState:
    write_output(state["evaluated"], state["run_metadata"])
    logger.info("✓ Pipeline complete — %d leads written", len(state["evaluated"]))
    return state


# ── Graph assembly ─────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    g = StateGraph(PipelineState)

    g.add_node("discover",      discover)
    g.add_node("score",         score)
    g.add_node("qualify",       qualify)
    g.add_node("personalise",   personalise)
    g.add_node("evaluate",      evaluate)
    g.add_node("write_output",  write)

    g.set_entry_point("discover")
    g.add_edge("discover",    "score")
    g.add_edge("score",       "qualify")
    g.add_edge("qualify",     "personalise")
    g.add_edge("personalise", "evaluate")
    g.add_conditional_edges("evaluate", should_retry,
                            {"personalise": "personalise", "write_output": "write_output"})
    g.add_edge("write_output", END)

    return g.compile()


# ── Entry point ────────────────────────────────────────────────────────────
def run_pipeline(icp_path: str) -> list[dict]:
    icp = load_icp(icp_path)
    graph = build_graph()
    initial: PipelineState = {
        "icp":            icp,
        "raw_prospects":  [],
        "scored_leads":   [],
        "qualified_leads": [],
        "outreach_ready": [],
        "evaluated":      [],
        "run_metadata":   {"icp_name": icp.get("name", "unknown")},
        "iteration":      0,
    }
    final = graph.invoke(initial)
    return final["evaluated"]
