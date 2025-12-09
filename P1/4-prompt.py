import dotenv
dotenv.load_dotenv()

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("{country}의 수도는?")
print(prompt)


# prompt = PromptTemplate(template="{country}의 수도는?", input_variables=["country"])

result = prompt.invoke({"country": "한국"})
print(result)

print("=============================\n")




prompt = PromptTemplate.from_template("""{date}이 생일인 유명한 한국 인물은 누가 있나요?
                                        {count} 명을 알려주세요.""")

from datetime import datetime
from langchain.chat_models import init_chat_model
llm = init_chat_model("gemini-robotics-er-1.5-preview", model_provider="google_genai")
prompt = prompt.partial(date=datetime.now().strftime("%m월 %d일"))
prompt.pretty_print()
chain = prompt | llm
# result = chain.invoke({"count": 5})
result = chain.invoke({"date": "3월 11일", "count": 10},)
print(result)