from crewai import Agent, Task, Crew, Process 
import os 
from langchain_google_genai import ChatGoogleGenerativeAI
from llamaapi import LlamaAPI
from langchain.tools import DuckDuckGoSearchRun
from langchain_experimental.llms import ChatLlamaAPI



os.environ["GOOGLE_API_KEY"] = 


llm = ChatGoogleGenerativeAI(model="gemini-pro", verbose=True, temperature=0.1)


search_tool = DuckDuckGoSearchRun()



researcher = Agent(
    role='Researcher',
    goal = 'Research new AI insights',
    backstory = 'You are an ai research assistant',
    verbose = True,
    allow_delegation = False,
    tools=[search_tool],
    llm = llm
                )

writer = Agent(
    role = 'Writer',
    goal = 'Write compelling and engaging blog posts about AI trends and insights. ',
    backstory = 'You are an AI blog post writer who specializes in writing about AI topics.',
    verbose = True,
    allow_delegation = False,
    tools=[search_tool],
    llm = llm
)

task1 = Task(description='Investigate the latest AI trends.', agent=researcher)
task2 = Task(description='Write a compelling blog post based on the latest AI trends.', agent=writer)

crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=2,
    process=Process.sequential
)

result = crew.kickoff()