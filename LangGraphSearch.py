from dotenv import load_dotenv 
load_dotenv()

import functools, operator, requests, os, json 
from bs4 import BeautifulSoup 
from duckduckgo_search import DDGS 
from langchain.schema import HumanMessage
from langchain_community.chat_models import ChatLiteLLM
from langchain.agents import AgentExecutor, create_react_agent 
from langchain_core.messages import BaseMessage, HumanMessage 
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langgraph.graph import StateGraph, END
from langchain.tools import tool 
from langchain_openai import ChatOpenAI 
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
import gradio as gr 

# Set environment variables 

# Initialize the Model
chat = ChatLiteLLM(model="gemini/gemini-pro")

# 1. Define Custom Tools 
@tool("internet_search", return_direct=False)
def internet_search(query: str) -> str:
    """Search the internet using DuckDuckGo."""
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=5)]
        return results if results else "No results found."
    
@tool("process_content", return_direct=False)
def process_content(url: str) -> str:
    """Process content from a webpage."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

tools = [internet_search, process_content]


# 2. Agents
# Helper function for creating agents
def create_agent(llm: ChatLiteLLM, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        
    ])
    agent = create_openai_tools_agent(llm, tools, prompt) 
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor 
