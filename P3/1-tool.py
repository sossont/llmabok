from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """
    Add two numbers.

    Args:
        a (int): the left operand.
        b (int): the right operand.

    Returns:
        The sum of the two operands.
    """

    return a + b

# print(type(add))
# print(add.name)
# print(add.description)
# print(add.args)

# response = add.invoke({"a": 2, "b": 3})
# print(response)

@tool
def increase(n: int) -> int:
    """Increase the number by 1.
    
    args:
        n (int): the number to increase.
    returns:
        The increased number.
    """
    return n + 1

@tool
def substract(a: int, b: int) -> int:
    """Substract two numbers.
    
    args:
        a (int): the left operand.
        b (int): the right operand.
    returns:
        The difference of the two operands.
    """
    return a - b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.
    
    args:
        a (int): the left operand.
        b (int): the right operand.
    returns:
        The product of the two operands.
    """
    return a * b

@tool
def divide(a: int, b: int) -> int:
    """Divide two numbers.
    
    args:
        a (int): the left operand.
        b (int): the right operand.
    returns:
        The quotient of the two operands.
    """
    if b == 0:
        raise ValueError("Division by zero")
    
    return a / b

# import dotenv
# dotenv.load_dotenv()

# from langchain_google_genai import ChatGoogleGenerativeAI
# llm = ChatGoogleGenerativeAI(model="gemini-robotics-er-1.5-preview")

# llm_with_tools = llm.bind_tools([add, increase, substract, multiply, divide]) # Tool Calling Agent

# from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
# messages = [SystemMessage(content="You are a calculator. You have to use the tools provided to you to calculate the result."),
#             HumanMessage(content="Add 2 and 3, then multiply the result by 4, then divide the result by 2")]
# response = llm_with_tools.invoke(messages)
# # print(response)
# tool_map = {
#     "add": add,
#     "multiply": multiply,
#     "divide": divide,   
#     "increase": increase,
#     "substract": substract,
# }

# #Agent Executor
# while response.content == "" and response.tool_calls:
#     messages.append(response)
#     for tc in response.tool_calls:
#         tool_result = tool_map[tc["name"]].invoke(tc["args"])
#         messages.append(
#             ToolMessage(content=str(tool_result), tool_call_id=tc["id"])
#         )
#         print(f"The result of {tc['name']} is {tool_result}")
#         print('============================')
#     response = llm_with_tools.invoke(messages)
 
# print(response)

# print('============================ Tool Calling Agent ============================')
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a calculator. You have to use the tools provided to you to calculate the result."),
#     ("user", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad")])

# from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
# agent = create_tool_calling_agent(llm, tools=[add, increase, substract, multiply, divide], prompt=prompt)
# agent_executor = AgentExecutor(agent=agent, tools=[add, increase, substract, multiply, divide])
# response = agent_executor.invoke({"input": "Add 2 and 3, then multiply the result by 4, then divide the result by 2"})
# print(response)


# 아래가 가장 최신의 방법

import dotenv
dotenv.load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

llm = ChatGoogleGenerativeAI(model="gemini-robotics-er-1.5-preview")
agent = create_agent(
    llm,
    tools=[add, increase, substract, multiply, divide],
    system_prompt="You are a calculator. You have to use the tools provided to you to calculate the result.",
)

response = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content="Add 2 and 3, then multiply the result by 4, then divide the result by 2"
            )
        ]
    }
)
print(response)
print(f'final response: {response["messages"][-1].content}')