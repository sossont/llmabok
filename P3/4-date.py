import dotenv
dotenv.load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-robotics-er-1.5-preview")

response = llm.invoke("모레는 무슨 요일인가요?")
print(response.content)
