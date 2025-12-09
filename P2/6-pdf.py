# pip install pypdf langchain-community langchain-google-genai langchain-text-splitters
import dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

dotenv.load_dotenv()


def build_context(docs: list) -> str:
    if not docs:
        return "문서에서 관련된 내용을 찾지 못했습니다."
    return "\n\n".join(doc.page_content for doc in docs)


def main() -> None:
    query = input("질문을 입력하세요: ").strip()
    if not query:
        print("질문이 비어 있습니다. 종료합니다.")
        return

    loader = PyPDFLoader("data/ai_agent_flow.pdf", mode="single")
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    docs = splitter.split_documents(raw_docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = InMemoryVectorStore.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(model="gemini-robotics-er-1.5-preview")
    prompt = PromptTemplate.from_template(
        "다음 문서 조각을 참고해 질문에 한국어로 간단히 답변하세요.\n"
        "문맥:\n{context}\n\n질문: {question}"
    )

    chain = (
        {
            "context": retriever | RunnableLambda(build_context),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    response = chain.invoke(query)
    print("\n=== 답변 ===")
    print(response.content)


if __name__ == "__main__":
    main()
