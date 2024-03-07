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
    role= "Artificial Intelligence Researcher",
    goal = "Research by searching on Internet for new applications of Internet of Things and Internet of Medical Things in Healtcare.",
    backstory = """You are Artificial Intelligence Researcher with 10 years of experience of doing research for Hedge Funds company. 
    You always been comming up with insightful research with new trends in various industry. You have helped your
    Hedge Fund Company very successful by finding new applications in industry.""",
    verbose = True,
    allow_delegation = False,
    tools=[search_tool],
    llm = llm
                )

writer = Agent(
    role = "Financial Analysis Writer",
    goal = "Write compelling and engaging report for Hedge Fund Company about new Internet of Things and Internet of Medical Things.",
    backstory = """You are an astonishing financial analysis writer for Hedge Fund Company. You always have made your Hedge Fund Company invest in
    right stocks by writing insightful reports based on real updated research.""",
    verbose = True,
    allow_delegation = False,
    tools=[search_tool],
    llm = llm
)

task1 = Task(description='Investigate the latest applications of Internt of Things and Internet of Medical Things in HealthCare.', agent=researcher)
task2 = Task(description='Write a report on cuting edge applications of Internt of Things and Internet of Medical Things in HealthCare.', agent=writer)

crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=2,
    process=Process.sequential
)

result = crew.kickoff()