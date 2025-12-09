import dotenv
dotenv.load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

llm = ChatGoogleGenerativeAI(model="gemini-robotics-er-1.5-preview-lite")

prompt = PromptTemplate.from_template("""
        다음 context를 근거로 질문에 답하세요. 
        답변할 때 context를 참고하되 context를 언급하지 말고 요약해서 대답하세요.
        context: {context}
        question: {question}
        """)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """
            You are a helpful assistant that answers questions based on the given context.
            You should answer the question based on the context and summarize the context in your answer.
            You should not mention the context in your answer.
            You should not use any other information than the context.
            """),
    ("human", "{question}"),
])
# chain = prompt | llm
chain = chat_prompt | llm
response = chain.invoke({"context": """
                            2025년 6월 3일에 실시된 제21대 대통령 선거에서 민주화 이후 역대 대선 선거인수 대비 득표수 최대 비율인 38.943%의 비율로 역대 대선 최다 득표인 17,287,513표를 받으면서 국민의힘 김문수 후보를 상대로 8.27%p 차이로 승리하여 대통령에 당선되었다. 
                            그리고 49.42%의 득표율로 기존 민주당계 정당의 대선 최고 득표율을 경신했다.
                            이재명은 2025년 6월 4일 대한민국의 제21대 대통령으로 취임했으며, 2030년 6월 3일까지 대통령으로서 직무를 수행할 예정이다.
                        """, 
                        "question": "한국의 대통령은?"})
print(response.content)
