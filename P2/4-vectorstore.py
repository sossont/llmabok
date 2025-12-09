import dotenv
dotenv.load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-robotics-er-1.5-preview")

from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)

vector_store.add_texts([
    "Hello, world!",
    "체인소맨은 덴지가 주인공인 일본 애니메이션 입니다.",
    "포치타는 귀엽습니다.",
    "주술회전에서 가장 강한 케릭터는 고죠 사토루입니다.",
    "고죠 사토루의 영역 전개는 최강입니다."
])

results = vector_store.similarity_search("고죠 사토루", k=2)

retriever = vector_store.as_retriever()

### RAG PROMPT
from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate.from_template("""
        다음 context를 근거로 질문에 답하세요. context를 언급하지 않고 요약하여 답하세요
        context: {context}
        question: {question}
        """)

### RAG CHAIN
from langchain_core.runnables import RunnablePassthrough
chain = { "context": retriever, "question": RunnablePassthrough() } | prompt | llm
response = chain.invoke("고죠 사토루는 누구인가?")
print(response.content)
