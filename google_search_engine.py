from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

model = ChatGroq(model="llama-3.3-70b-versatile")
search = GoogleSerperAPIWrapper()

agent = create_agent(
    model = model,
    tools = [search.run],
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