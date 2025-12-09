import dotenv
dotenv.load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-robotics-er-1.5-preview")

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
vector_store = InMemoryVectorStore(embeddings)

with open("ai_agent_rag.md", "r", encoding="utf-8") as f:
    texts = f.read().splitlines()

vector_store.add_texts(texts)
vector_store.dump("vectorstore.json")

embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")
vector_store = InMemoryVectorStore.load("vectorstore.json", embeddings)
cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
reranker = CrossEncoderReranker(model=cross_encoder)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=vector_store.as_retriever()
)
searched = compression_retriever.invoke("ISO/IEC는 AI 에이전트를 어떻게 정의하는가?")
print(searched)
print('===============================')