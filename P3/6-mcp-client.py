import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
import dotenv
dotenv.load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-robotics-er-1.5-preview")
client = MultiServerMCPClient(
    {
        "math": {
            "command": "python",
            "args": ["P3/5-mcp-server.py"],
            "transport": "stdio",
        }
    }
)

async def main():
    tools = await client.get_tools()    # asynchronous call to get tools
    from langchain_core.messages import HumanMessage
    from langchain.agents import create_agent
    agent = create_agent(llm, tools=tools)
    response = await agent.ainvoke({"messages": [HumanMessage("2 더하기 3은?")]})
    print(response)
    print("=============================")

asyncio.run(main())
