import dotenv
dotenv.load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-robotics-er-1.5-preview")

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")

from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)

with open("AI 에이전트 동향_LLM_Splitted.txt", "r", encoding="utf-8") as f:
    texts = f.read().splitlines()

vector_store.add_texts(texts)
vector_store.dump("vectorstore.json")
