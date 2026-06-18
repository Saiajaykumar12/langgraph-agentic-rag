import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

model = ChatGroq(model="llama-3.3-70b-versatile")

search_provider = os.getenv("SEARCH_PROVIDER", "serper").lower()
if search_provider == "tavily":
    search_tool = TavilySearchResults(max_results=5)
    tools = [search_tool]
else:
    search = GoogleSerperAPIWrapper()
    tools = [search.run]

agent = create_agent(
    model = model,
    tools = tools,
    system_prompt = "You are an agent and can search for any question on google",
    checkpointer = MemorySaver()
)

while True:
    query = input("User: ")
    if(query.lower() == "exit"):
        print("Goodbye!")
        break

    response = agent.invoke({"messages":[{"role":"user","content":query}]},
                            {"configurable":{"thread_id":"Ajay"}})
    print("AI: ", response["messages"][-1].content)