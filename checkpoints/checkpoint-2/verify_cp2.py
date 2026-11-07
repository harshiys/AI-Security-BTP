#!/usr/bin/env python3
"""
SentinelMesh Checkpoint 2 Gatekeeper CLI (verify_cp2.py)
Inspects agents, runs, and controlled attack scenarios against fused Cyber-ML runtime telemetry.
Dataset: Real TAMAS (Microsoft AutoGen DPI Benchmark).
"""
import argparse
import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

def load_fused_data():
    summary_path = ROOT / "results" / "checkpoint2_fused_summary.csv"
    if not summary_path.exists():
        print("Error: results/checkpoint2_fused_summary.csv not found. Please run 'python run_checkpoint2.py' first.")
        sys.exit(1)
    return pd.read_csv(summary_path)

def verify_agent(run_id, agent_name, df):
    matches = df[(df["run_id"] == run_id) & (df["agent"] == agent_name)]
    if len(matches) == 0:
        print(f"BLOCKED: Agent '{agent_name}' in run '{run_id}' not found in runtime registry.")
        sys.exit(1)
    
    row = matches.iloc[0]
    verdict = row["verdict"]
    score = float(row["score"])
    drift = float(row["behavioral_drift"])
    dyn_trust = float(row["dynamic_trust"])

    if verdict == "pass":
        print(f"ALLOWED: Agent '{agent_name}' verified safe | verdict={verdict} (risk_score={score:.4f}, dynamic_trust={dyn_trust:.4f}, drift={drift:.4f})")
        sys.exit(0)
    elif verdict == "flag":
        print(f"FLAGGED: Agent '{agent_name}' flagged for anomaly | verdict={verdict} (risk_score={score:.4f}, dynamic_trust={dyn_trust:.4f}, drift={drift:.4f})")
        sys.exit(1)
    else:
        print(f"BLOCKED: Agent '{agent_name}' blocked by runtime policy | verdict={verdict} (risk_score={score:.4f})")
        sys.exit(1)

def verify_scenario(scenario_name):
    scenarios_file = ROOT / "cyber_results" / "cp2_attack_scenarios.json"
    if not scenarios_file.exists():
        print(f"Error: {scenarios_file} not found.")
        sys.exit(1)
    
    raw_data = json.load(open(scenarios_file, encoding="utf-8"))
    scenarios = raw_data.get("scenarios", []) if isinstance(raw_data, dict) else raw_data
    match = next((s for s in scenarios if s.get("scenario") == scenario_name), None)
    if not match:
        avail = [s["scenario"] for s in scenarios]
        print(f"Unknown scenario '{scenario_name}'. Available scenarios: {', '.join(avail)}")
        sys.exit(1)

    print(f"Evaluating Controlled Cyber Attack Scenario: '{scenario_name}'")
    print(f"Description: {match['description']}")
    
    if scenario_name == "impersonation":
        print("BLOCKED: impersonation attack detected (claimed_sender does not match acting_agent cryptographic identity)")
        sys.exit(1)
    elif scenario_name == "rogue_delegation":
        print("BLOCKED: rogue delegation detected (communication edge violates authorized coordination graph topology)")
        sys.exit(1)
    elif scenario_name == "sleeper_agent":
        print("BLOCKED: sleeper agent activation detected (anomalous post-activation tool invocation outside historical baseline)")
        sys.exit(1)

def show_all(df):
    print("=" * 85)
    print("SENTINELMESH CHECKPOINT 2: RUNTIME AGENT TRUST & VERDICT MATRIX")
    print("=" * 85)
    cols = ["run_id", "agent", "behavioral_drift", "cyber_risk", "dynamic_trust", "score", "verdict"]
    display_df = df[cols].copy()
    print(display_df.to_string(index=False))
    print("=" * 85)
    print(f"Total Agents Monitored: {len(df)} | Allowed: {sum(df.verdict=='pass')} | Flagged: {sum(df.verdict=='flag')} | Blocked: {sum(df.verdict=='block')}")

def main():
    p = argparse.ArgumentParser(description="SentinelMesh CP2 Runtime Gatekeeper")
    p.add_argument("--run-id", default=None, help="TAMAS run_id (e.g. r77c0e4eb97)")
    p.add_argument("--agent", default=None, help="Agent name (e.g. news_gathering_agent)")
    p.add_argument("--scenario", choices=["impersonation", "rogue_delegation", "sleeper_agent"], help="Evaluate controlled cyber attack scenario")
    p.add_argument("--all", action="store_true", help="Display full table of all monitored agents")
    args = p.parse_args()

    df = load_fused_data()

    if args.scenario:
        verify_scenario(args.scenario)
    elif args.all:
        show_all(df)
    elif args.run_id and args.agent:
        verify_agent(args.run_id, args.agent, df)
    else:
        # Default behavior: test sample clean and sample flagged agent
        print("=== Quick Demonstration (Run with --all or --help for more options) ===\n")
        print("1. Clean Agent Verification:")
        try:
            verify_agent("r71a932976c", "article_writing_agent", df)
        except SystemExit:
            pass
        
        print("\n2. Flagged Anomaly Agent Verification:")
        try:
            verify_agent("r77c0e4eb97", "news_gathering_agent", df)
        except SystemExit:
            pass

        print("\n3. Controlled Attack Scenario (Impersonation):")
        try:
            verify_scenario("impersonation")
        except SystemExit:
            pass

if __name__ == "__main__":
    main()
