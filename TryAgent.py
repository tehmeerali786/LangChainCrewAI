from dotenv import load_dotenv 
load_dotenv()

import litellm
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
litellm.add_function_to_prompt = True

tools = [TavilySearchResults(max_results=1)]

# Get the prompt to use - you can modify this!
# prompt = hub.pull("hwchase17/openai-tools-agent")
prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert"),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad")])
        
        
# Choose the LLM that will drive the agent
# Only certain models support this
llm = ChatLiteLLM(model="gemini/gemini-pro")

# Construct the OpenAI Tools agent
agent = create_openai_tools_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


agent_executor.invoke({"input": "what is LangChain?"})

