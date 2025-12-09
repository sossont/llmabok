import dotenv
dotenv.load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-robotics-er-1.5-preview")

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")

from langchain_chroma import Chroma
vector_store = Chroma(embedding_function=embeddings, persist_directory="chroma_db")

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
reranker = CrossEncoderReranker(model=encoder)

docs = [
"사과는 빨간색도 있고, 초록색도 있습니다. 나는 빨간 사과는 좋아하지만 초록 사과는 싫어합니다.",
"바나나는 노란색입니다. 바나나는 맛있습니다. 나는 바나나를 좋아합니다.",
"포도는 보라색입니다. 포도는 작고 달콤합니다. 포도는 씨가 있어서 먹기 불편합니다. 나는 포도를 싫어합니다.",
"나는 낚시를 좋아합니다. 낚시는 물고기를 잡는 재미가 있습니다. 낚시는 자연과 함께하는 활동입니다.",
"나는 여행을 좋아합니다. 여행을 통해 다양한 과일을 맛볼 수 있습니다.",
]
query = "나는 어떤 과일을 좋아할까요?"

vector_store.add_texts(docs)

from langchain_classic.retrievers import ContextualCompressionRetriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=vector_store.as_retriever()
)

results = compression_retriever.invoke(query)
print(results)
print('===============================')