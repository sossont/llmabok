import dotenv
dotenv.load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-robotics-er-1.5-preview")

from typing import TypedDict
class State(TypedDict):
    subject: str
    story: str
    criticism: str
