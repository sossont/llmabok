# import dotenv
# dotenv.load_dotenv()
# from langchain_experimental.tools import PythonREPLTool


# """
# 요구사항 : 내가 코드를 작성하는게 아니라, PythonREPLTool을 사용해서 코드를 실행하고 결과를 반환하는 도구만 주고
#             코드는 LLM이 작성하도록 함. 그리고 코드 작성 시에 주석을 달아야 함.
# """

# from langchain_core.prompts import PromptTemplate
# context = """
# 당신은 프로그래머 입니다.
# 제가 질문한 내용을 해결하기 위한 Python 코드를 작성해주세요.
# 코드 작성 시에 주석을 달아야 하고, Python 코드를 실행해서 결과를 반환해줘야 함.
# Python 코드를 실행하는 tool은 내가 PythonREPLTool을 사용할 것이므로, 너는 코드만 작성하면 돼.
# 그리고 답변과 함께 작성한 코드를 깔끔하게 출력해줘.

# ---
# 질문 예시 :
# 5 팩토리얼을 계산해줘

# ---
# 너가 작성할 코드 예시 :
# def factorial(n):
#     if n == 0:
#         return 1
#     else:
#         return n * factorial(n-1)

# """

# question = """
# 종을 치면 일어나는 사건은 아래와 같아.
# 1. 암탉이 있으면 암탉 1마리가 알을 1개 낳는다.
# 2. 알이 있으면 알은 병아리가 된다.
# 3. 병아리가 있으면 병아리는 암탉이 된다.

# 암탉이 처음에 1마리 있을 때, 하루에 1번 씩 종을 칠 경우, 10일 지나면 암탉, 병아리, 알은 각각 몇 개일까?
# """

# from langchain_google_genai import ChatGoogleGenerativeAI
# llm = ChatGoogleGenerativeAI(model="gemini-robotics-er-1.5-preview")

# tool = PythonREPLTool()


# # response = tool.invoke("print(100 + 200)")

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

llm = ChatGoogleGenerativeAI(model="gemini-robotics-er-1.5-preview")
agent = create_agent(
    llm,
    tools=[tool],
    system_prompt=context,
)


response = agent.invoke(
    {
        "messages": [
            HumanMessage(content=question),
        ]
    }
)
print(response["messages"][-1].content)


chicks = 1
eggs = 0
hens = 0

for bell_count in range(10):
    eggs_laid = chicks
    chicks_hatched = hens
    hens_matured = eggs
    eggs = eggs_laid
    hens = hens_matured
    chicks += chicks_hatched
    print(f"Day {bell_count+1}: {chicks} chickens, {eggs} eggs, {hens} hens")
