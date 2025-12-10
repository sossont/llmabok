import datetime

import dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

dotenv.load_dotenv()


@tool
def get_today() -> str:
    """Return today's date in ISO format (YYYY-MM-DD)."""
    return datetime.date.today().isoformat()


llm = ChatGoogleGenerativeAI(model="gemini-robotics-er-1.5-preview")
agent = create_agent(
    llm,
    tools=[get_today],
    system_prompt=(
        "현재 날짜가 필요하면 get_today 도구를 사용해서 확인한 뒤 답하세요. "
        "응답은 한국어로 간단히."
    ),
)

question = ("내년 크리스마스는 무슨 요일이야?")

response = agent.invoke(
    {
        "messages": [
            HumanMessage(content=question),
        ]
    }
)
print(response["messages"][-1].content)
