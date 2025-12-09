"""ai_agent_rag.md를 RAG 벡터 DB에 올려 질의응답하는 예시.

기본 흐름(P2/6-pdf.py와 동일 구조):
1) Markdown 로드 → 헤더/길이 기반 청크 분할
2) Google Embedding → 인메모리 벡터스토어
3) Retriever + Gemini LLM으로 답변

venv 예시:
python -m venv .venv
.venv\\Scripts\\activate
pip install -U langchain-community langchain-google-genai langchain-text-splitters python-dotenv
"""

from pathlib import Path
import dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


def build_context(docs: list) -> str:
    """선택된 문서 조각을 하나의 문자열로 합칩니다."""
    if not docs:
        return "문서에서 관련된 내용을 찾지 못했습니다."
    return "\n\n".join(doc.page_content for doc in docs)


def load_markdown_docs(md_path: Path):
    """Markdown을 헤더/길이 기반으로 쪼개 LangChain Document 리스트로 반환."""
    text = md_path.read_text(encoding="utf-8")

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Chapter"), ("##", "Section"), ("###", "Subsection")],
        strip_headers=False,
    )
    header_docs = header_splitter.split_text(text)

    chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    return chunk_splitter.split_documents(header_docs)


def main() -> None:
    dotenv.load_dotenv()

    question = input("질문을 입력하세요: ").strip()
    if not question:
        print("질문이 비어 있습니다. 종료합니다.")
        return

    md_path = Path(__file__).resolve().parent.parent / "ai_agent_rag.md"
    if not md_path.exists():
        print(f"Markdown 파일을 찾을 수 없습니다: {md_path}")
        return

    docs = load_markdown_docs(md_path)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = InMemoryVectorStore.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(model="gemini-robotics-er-1.5-preview")
    prompt = PromptTemplate.from_template(
        """
        다음 문서의 내용을 바탕으로 간결하고 명확한 답변을 해주세요.
        기술적인 내용을 중심으로, 중요한 포인트는 강조하되 설명은 간단하고 직설적으로 해주세요.
        이모지는 사용하지 마세요.
        문서 내용: {context}
        질문: {question}
        """
    )

    chain = (
        {
            "context": retriever | RunnableLambda(build_context),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    response = chain.invoke(question)
    print("\n=== 답변 ===")
    print(response.content)


if __name__ == "__main__":
    main()

# --- 기존 코드 (요청에 따라 주석 처리) ---
# with open("data/langchain.md", "r", encoding="utf-8") as f:
#     file = f.read()
#
# from langchain_text_splitters import MarkdownHeaderTextSplitter
# splitter = MarkdownHeaderTextSplitter(
#              headers_to_split_on=[("#", "Chapter"), ("##", "Section")],
#              strip_headers=False,
# )
#
# docs = splitter.split_text(file)
#
# print(f"Number of splitted documents: {len(docs)}")
# for doc in docs:
#     print(f"\n\n---\n\n{doc}")
