import dotenv
dotenv.load_dotenv()

from enum import Enum
class Color(Enum):
    RED    = "빨강"
    BLUE   = "파랑"
    YELLOW = "노랑"

from langchain_classic.output_parsers import EnumOutputParser
output_parser = EnumOutputParser(enum=Color)

from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model

prompt = PromptTemplate.from_template("""{product}의 색상은?
                                        다음 형식으로 답변하되, 색상만 출력해주세요.
                                        {format}""")

prompt = prompt.partial(format=output_parser.get_format_instructions())
llm = init_chat_model("gemini-robotics-er-1.5-preview-lite", model_provider="google_genai")
chain = prompt | llm | output_parser

response = chain.invoke({"product": "물"})
print(response)
