import dotenv
dotenv.load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.retrievers import WikipediaRetriever


def build_context(docs: list) -> str:
    if not docs:
        return "검색 결과가 없습니다."
    return "\n\n".join(doc.page_content for doc in docs)


def main() -> None:
    query = input("검색어를 입력하세요: ").strip()
    if not query:
        print("검색어가 비어 있습니다. 종료합니다.")
        return

    # pip install wikipedia
    retriever = WikipediaRetriever()
    llm = ChatGoogleGenerativeAI(model="gemini-robotics-er-1.5-preview")

    prompt = PromptTemplate.from_template(
        "다음 위키 문서를 참고해 질문에 요약하여 한국어로 답하세요.\n"
        "문맥:\n{context}\n\n질문: {question}"
    )

    chain = (
        {
            "context": retriever,
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
