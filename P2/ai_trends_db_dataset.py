import dotenv
dotenv.load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-robotics-er-1.5-preview")

with open("AI 에이전트 동향.txt", "r", encoding="utf-8") as f:
    file = f.read()

from pydantic import BaseModel, Field
class TextSplitted(BaseModel):
    contents: list[str] = Field(description="The list of the text chunk.")

from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate.from_template("""
You are a text splitter.
Your task is to split the provided text into smaller, semantically coherent chunks.
Each chunk should be concise and maintain the original meaning of the text.
출력 포맷은 다음과 같습니다: {format}
---
content: {content}      
""")

from langchain_core.output_parsers import PydanticOutputParser
output_parser = PydanticOutputParser(pydantic_object=TextSplitted)

prompt = prompt.partial(format=output_parser.get_format_instructions())

chain = prompt | llm | output_parser
response = chain.invoke({"content": file})

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="../models/Qwen3-Embedding-0.6B")

from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)

vector_store.add_texts(response.contents)
vector_store.dump("vector_store.json")

from ragas.testset import TestsetGenerator
generator = TestsetGenerator.from_langchain(llm, embeddings)

from langchain_core.documents import Document
docs = [Document(page_content=text) for text in response.contents]

dataset = generator.generate_with_langchain_docs(docs, testset_size=10)

df = dataset.to_pandas()
df.to_csv('dataset.csv', index=False)
