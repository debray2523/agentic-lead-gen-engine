#!/usr/bin/env python3
"""
main.py — Agentic Lead Generation Engine entry point

Usage:
  python main.py --icp configs/sample_icp.yaml
  python main.py --icp configs/sample_icp.yaml --max-leads 50 --output-dir ./output
  python main.py --icp configs/sample_icp.yaml --demo   # no API keys needed

Author : Dr. Debendra Ray, DBA — Independent AI Researcher
Licence: MIT
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()

def parse_args():
    p = argparse.ArgumentParser(description="Agentic Lead Generation Engine")
    p.add_argument("--icp",        required=True,  help="Path to ICP YAML config")
    p.add_argument("--max-leads",  type=int, default=100, help="Max leads per run")
    p.add_argument("--output-dir", default="./output", help="Output directory")
    p.add_argument("--demo",       action="store_true", help="Run with synthetic data (no API keys)")
    p.add_argument("--log-level",  default="INFO", choices=["DEBUG","INFO","WARNING"])
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    if args.demo:
        os.environ["DEMO_MODE"] = "true"
    os.environ["OUTPUT_DIR"] = args.output_dir

    console.print(Panel.fit(
        "[bold cyan]Agentic Lead Generation Engine[/bold cyan]\n"
        "[dim]Dr. Debendra Ray, DBA · Independent AI Researcher[/dim]\n"
        "[dim]github.com/debray2523/agentic-lead-gen-engine[/dim]",
        border_style="cyan"
    ))

    from orchestrator import run_pipeline
    leads = run_pipeline(args.icp)

    console.print(f"\n[bold green]✓ Pipeline complete — {len(leads)} leads processed[/bold green]")
    passed = sum(1 for l in leads if l.get("evaluation", {}).get("judge_passed"))
    console.print(f"  Judge passed  : [bold]{passed}/{len(leads)}[/bold]")
    console.print(f"  Output dir    : [dim]{args.output_dir}[/dim]\n")


if __name__ == "__main__":
    main()
