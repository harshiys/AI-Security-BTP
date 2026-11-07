import asyncio
import os
import json
import argparse
import importlib
import inspect
from autogen_agentchat.teams import MagenticOneGroupChat, RoundRobinGroupChat, Swarm
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_agentchat.conditions import (
    ExternalTermination, TextMentionTermination, MaxMessageTermination,
)
from autogen_ext.models.openai import OpenAIChatCompletionClient

# ---------------------------------------------------------------------------
# SentinelMesh patch (Groq): this is TAMAS's own autogen_run.py, modified only
# to (a) use Groq's OpenAI-compatible endpoint with GROQ_API_KEY, and (b) wire
# the correct tool module to each agent in ALL FIVE DPI scenarios, not only
# News. No tool function is invented: each agent is given exactly the functions
# that are actually defined in its mapped module.
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Input user query")
parser.add_argument("--json_data", type=str, required=True, help="Path to JSON input file")
parser.add_argument("--model", type=str, required=True, help="Model name to be used")
parser.add_argument("--config", type=str, required=True, help="Configuration name")
args = parser.parse_args()


def _build_model_client():
    """Groq via the OpenAI-compatible API. Fails loudly if the key is missing."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise SystemExit(
            "GROQ_API_KEY is not set. This patched runner talks to Groq's "
            "OpenAI-compatible API.\n"
            "  PowerShell:  $env:GROQ_API_KEY = 'gsk_...'\n"
            "  Git Bash:    export GROQ_API_KEY=gsk_...\n"
            "Get a key at https://console.groq.com/keys"
        )
    return OpenAIChatCompletionClient(
        model=args.model,
        api_key=key,
        base_url="https://api.groq.com/openai/v1",
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "gpt-oss",
            "structured_output": True,
        },
    )


model_client = _build_model_client()

# Agent display name -> tool module name, for every DPI scenario.
# (Display names verified against TAMAS data/DPI/<scenario>_DPI.json.)
AGENT_MODULE = {
    # News
    "News Gathering Agent": "tools_news_gathering_agent",
    "Fact Checking Agent": "tools_fact_checking_agent",
    "Article Writing Agent": "tools_article_writing_agent",
    "Distribution Agent": "tools_distribution_agent",
    # Education
    "Adaptive Learning Agent": "tools_adaptive_learning_agent",
    "Assessment Agent": "tools_assessment_agent",
    "Progress Tracking Agent": "tools_progress_tracking_agent",
    "Concept Clarification Agent": "tools_concept_clarification_agent",
    # Finance
    "Identity Verification Agent": "tools_identity_verification_agent",
    "Market Analysis Agent": "tools_market_analysis_agent",
    "Trade Execution Agent": "tools_trade_execution_agent",
    "Risk Management Agent": "tools_risk_management_agent",
    # Healthcare
    "Diagnosis Agent": "tools_diagnosis_agent",
    "Medication Adherence Agent": "tools_medical_adherence_agent",
    "Real time Monitoring Agent": "tools_realtime_monitoring_agent",
    "Patient Interaction Agent": "tools_patient_interaction_agent",
    # Legal
    "Legal Research Agent": "tools_research_agent",
    "Document Drafting Agent": "tools_doc_draft_agent",
    "Case Analysis Agent": "tools_analysis_agent",
    "Compliance Check Agent": "tools_compliance_agent",
}


def _strip_ab_suffix(name: str) -> str:
    """News Swarm/collusion configs use 'Agent A' / 'Agent B' variants."""
    for suf in (" A", " B"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def _module_functions(module):
    """Only functions DEFINED in this module (not imported helpers)."""
    out = []
    for fn_name, fn in inspect.getmembers(module, inspect.isfunction):
        if getattr(fn, "__module__", None) == module.__name__:
            out.append(fn)
    return out


def tools_for(agent_name: str):
    key = agent_name if agent_name in AGENT_MODULE else _strip_ab_suffix(agent_name)
    mod_name = AGENT_MODULE.get(key)
    if not mod_name:
        return []
    try:
        module = importlib.import_module(mod_name)
    except Exception as e:
        print(f"[patch] could not import {mod_name} for '{agent_name}': {e}")
        return []
    return _module_functions(module)


initialized_agents = {}
agent_list = []


async def main() -> None:
    try:
        with open(args.json_data, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    for agent in data["agents"]:
        var_name = agent["agent_name"].lower().replace(" ", "_")
        agent_name = var_name
        system_message = agent["agent_description"]
        tools = tools_for(agent["agent_name"])

        initialized_agents[var_name] = AssistantAgent(
            name=agent_name.replace(" ", "_"),
            model_client=model_client,
            tools=tools,
            system_message=system_message,
        )
        agent_list.append(initialized_agents[var_name])

    termination = MaxMessageTermination(50) | TextMentionTermination("TERMINATE")

    if args.config == "RoundRobin":
        team = RoundRobinGroupChat(agent_list, termination_condition=termination)
    elif args.config == "Swarm":
        team = Swarm(agent_list, termination_condition=termination)
    elif args.config == "Magentic_one":
        team = MagenticOneGroupChat(agent_list, model_client=model_client)
    else:
        print(f"Unknown config: {args.config}")
        return

    await Console(team.run_stream(task="Task: " + data["user query"]))


if __name__ == "__main__":
    asyncio.run(main())
