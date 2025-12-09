import dotenv
dotenv.load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [ # (role, message)
        ("system", "당신은 친절한 AI 어시스턴트입니다. 당신의 이름은 {name} 입니다."),
        ("human", "반가워요!"),
        ("ai", "안녕하세요! 무엇을 도와드릴까요?"),
        ("human", "{user_input}"),
    ]
)

prompt2 = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 대화 내용을 요악해주는 도우미입니다."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "대화 내용을 요약해 주세요.")
    ]
)

print(prompt)
print('============================')
print(prompt2)
print('============================')


from langchain.chat_models import init_chat_model
llm = init_chat_model("gemini-robotics-er-1.5-preview", model_provider="google_genai")
# chain = prompt | llm
# # response = prompt.invoke({ "name":"테디", "user_input":"당신의 이름은 무엇입니까?" })
# # print(response)
# print(chain.invoke({"name": "테디", "user_input": "당신의 이름은 무엇입니까?"}))

chain2 = prompt2 | llm
print(chain2.invoke({"history": [("human", "한국의 수도는?"), ("ai", "부산입니다."), ("human", "한국의 대통령은?"), ("ai", "이재명입니다.")]}))