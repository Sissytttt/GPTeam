from langchain.tools import BaseTool
from langchain.agents import load_tools
from langchain.llms import OpenAI
from typing import List

custom_tools: List[BaseTool] = []

built_in_tools: List[BaseTool] = load_tools(
    tool_names=["wolfram-alpha", "serpapi", "human"], llm=OpenAI()
)

all_tools = custom_tools + built_in_tools
