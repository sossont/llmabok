import dotenv
dotenv.load_dotenv()

from langchain.chat_models import init_chat_model
llm = init_chat_model("gemini-robotics-er-1.5-preview", model_provider="google_genai")

# countries = ["한국", "미국", "일본"]
# for country in countries:
    # response = llm.invoke(f'{country}의 수도는? 인구, 면적, 기후 등도 알려줘')
    # print(response)

message = 'message'
response = llm.invoke(message)
print(response)
