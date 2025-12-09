import dotenv
dotenv.load_dotenv()

from pydantic import BaseModel, Field

class ResponseModel(BaseModel):
    country: str = Field(description="The name of the country")
    capital: str = Field(description="The Capital City of the country")

from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
output_parser = JsonOutputParser(pydantic_object=ResponseModel)

from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model

prompt = PromptTemplate.from_template("""{country}의 수도는?
                                        다음 형식으로 답변해주세요.
                                        {format}""")
prompt = prompt.partial(format=output_parser.get_format_instructions())
prompt.pretty_print()

llm = init_chat_model("gemini-robotics-er-1.5-preview", model_provider="google_genai")
chain = prompt | llm | output_parser

response = chain.invoke({"country": "중국"})
# print(f"{response.country}: {response.capital}")
print(response)