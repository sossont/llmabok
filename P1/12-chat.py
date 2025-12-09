import dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

dotenv.load_dotenv()

# 친구처럼 따뜻하고 자연스럽게 답하도록 설정
# 참고: 이전 시스템 문구 예시 (보관용)
# ("system", "당신은 나의 친구입니다. 편안하게 대화해 주세요."),
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 나의 아들이고, 10살이며 한국에사는 초등학교 3학년 학생입니다. 정중하고 예의바르게 대답하되, 사춘기가 올 거 같은 학생입니다."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{user_input}"),
    ]
)
llm = ChatGoogleGenerativeAI(model="gemini-robotics-er-1.5-preview-lite")

sessions: dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # 세션마다 별도 히스토리 보관
    return sessions.setdefault(session_id, InMemoryChatMessageHistory())


chain = RunnableWithMessageHistory(
    prompt | llm,
    get_session_history,
    input_messages_key="user_input",
    history_messages_key="history",
)

# --- 기존 수동 히스토리 관리 버전 (보관용 주석) ---
# from langchain_core.messages import AIMessage, HumanMessage
# history: list = []
# response = (prompt | llm).invoke({"history": history, "user_input": user_input})
# history.append(HumanMessage(content=user_input))
# history.append(AIMessage(content=response.content))



def chat_loop() -> None:
    """간단한 CLI 대화 루프."""
    session_id = "cli"
    print("친구 챗봇이 켜졌어요. 종료하려면 'quit' 입력!")
    while True:
        user_input = input("나: ").strip()
        if user_input.lower() in {"quit", "exit", "종료"}:
            print("챗봇: 또 이야기해요! 👋")
            break

        # 체인 실행 (RunnableWithMessageHistory가 세션별 히스토리를 관리)
        response = chain.invoke(
            {"user_input": user_input},
            config={"configurable": {"session_id": session_id}},
        )

        # 출력
        print(f"챗봇: {response.content}")


if __name__ == "__main__":
    chat_loop()
