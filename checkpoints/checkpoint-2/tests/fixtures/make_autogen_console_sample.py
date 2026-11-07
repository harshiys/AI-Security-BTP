# Produces tests/fixtures/autogen_console_sample.txt: REAL AutoGen (autogen-agentchat 0.7.5) Console output with a SCRIPTED model
# (ReplayChatCompletionClient) and real TAMAS tool functions. It checks the log FORMAT only. It is not a TAMAS execution.
# Usage: TAMAS_TOOLS=TAMAS/data/tools/autogen python make_autogen_console_sample.py > autogen_console_sample.txt
import asyncio, os, sys
sys.path.insert(0, os.environ.get("TAMAS_TOOLS", "TAMAS/data/tools/autogen"))
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_core import FunctionCall
from autogen_core.models import CreateResult, RequestUsage
from autogen_ext.models.replay import ReplayChatCompletionClient
import tools_news_gathering_agent as tn, tools_article_writing_agent as tw

def call(name):
    return CreateResult(finish_reason="function_calls", content=[FunctionCall(id="c1", arguments="{}", name=name)],
                        usage=RequestUsage(prompt_tokens=1, completion_tokens=1), cached=False)
def text(t):
    return CreateResult(finish_reason="stop", content=t, usage=RequestUsage(prompt_tokens=1, completion_tokens=1), cached=False)

async def main():
    c1 = ReplayChatCompletionClient([call("fetch_latest_rss_feeds")]+[text("Feeds fetched, passing on.")]*6, model_info={"vision":False,"function_calling":True,"json_output":False,"family":"unknown","structured_output":False})
    c2 = ReplayChatCompletionClient([text("Drafting the article.")]*6, model_info={"vision":False,"function_calling":True,"json_output":False,"family":"unknown","structured_output":False})
    a = AssistantAgent("news_gathering_agent", model_client=c1, tools=[tn.fetch_latest_rss_feeds], system_message="gather")
    b = AssistantAgent("article_writing_agent", model_client=c2, tools=[tw.write_headline], system_message="write")
    team = RoundRobinGroupChat([a, b], termination_condition=MaxMessageTermination(6))
    await Console(team.run_stream(task="Task: write a news summary"))
asyncio.run(main())
