import dotenv
dotenv.load_dotenv()

from langchain_classic.output_parsers import DatetimeOutputParser
output_parser = DatetimeOutputParser()

from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model

prompt = PromptTemplate.from_template("""{company}의 창립기념일은?
                                        다음 형식으로 답변해주세요.
                                        {format}""")
prompt = prompt.partial(format=output_parser.get_format_instructions())
llm = init_chat_model("gemini-robotics-er-1.5-preview", model_provider="google_genai")
chain = prompt | llm | output_parser
response = chain.invoke({"company": "Google"})
print(response)
