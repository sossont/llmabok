import dotenv
dotenv.load_dotenv()

with open("AI 에이전트 동향.txt", "r", encoding="utf-8") as f:
    file = f.read()

# from langchain_text_splitters import CharacterTextSplitter
# splitter = CharacterTextSplitter()

# docs = splitter.create_documents([file])

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

prompt = PromptTemplate.from_template(f"""
            당신은 Vecotr DB에 저장할 최적의 문서를 추출하기 위한 전문적인 분석가입니다.
            당신의 목표는 제공된 원문 텍스트에서 중요한 정보를 추출하고, 이를 최대한 잘 보존하면서 가능한 짧은 문서 조각으로 분할하는 것입니다.
            1. 복잡한 문장은 가능한 짧은 문서 조각으로 분할하세요.
            2. 가능한 원본 문장은 유지합니다.
            3. 지시 대명사는 제거합니다.
            4. 중복된 문장은 제거합니다.
            5. 중복된 단어는 제거합니다.
            6. 중복된 문장 구조는 제거합니다.

            -----
            입력 예시는 아래와 같습니다.
            AI 에이전트 동향– 빅테크 기업의 AI 에이전트 사례를 중심으로  최근 생성형 AI의 확산과 함께 인간과 상호작용이 가능한 AI 에이전트에 대한 관심이 급증하고 있으며, 향후 몇 년 안에 관련 시장이 급속히 성장할 것으로 예상된다. 빅테크 기업들이 AI 에이전트 시장에 잇따라 진출하며 다양한 수익 모델을 창출하고 있는 상황이다. 그러나 AI 에이전트는 기술적, 사회적, 윤리적, 법적 문제를 초래할 가능성도 내포하고 있다. 앞으로 AI 에이전트는 기술 혁신과 사회 변화를 주도하는 핵심 요소로 자리 잡으며, 기업의 업무방식과 개인의 삶에 깊숙이 통합되어 획기적인 변화를 이끌어갈 것으로 전망된다.1. AI 에이전트(AI Agent)의 도입 및 부상 1) AI 에이전트의 정의 £AI 에이전트에 대한 합의된 학술적 정의는 부재하나, 관련 연구에서는 AI 에이전트를 단순한 AI 모델이나 알고리즘과 구별하여  설명하며, 특히 상호작용과 독립적인 의사결정 능력을 강조 ∙스튜어트 러셀(Stuart Russell)과 피터 노비그(Peter Norvig)는 2021년 출판한 저서 ‘인공지능 – 현대적 접근법’에서 에이전트(Agent)는 센서를 통해 환경을 인식하고 센서가 액추에이터를 통해 해당 환경에 작용하는 것으로, 합리적 에이전트(Rational Agent)는 최선의 결과를 달성하기 위해 행동하거나, 불확실성이 있는 경우 최선의 기대 결과를 얻기 위해 행동하는 행위자로 정의1) ∙알란 찬(ALan Chan) 등은 2024년 발간된 연구논문인 ‘AI 에이전트에 대한 가시성’2)에서 많은 AI 개발자들이 더 큰 자율성, 외부 도구나 서비스 접근, 장기적 목표 달성을 위해 안정적으로 적응하고 계획하며 지속적 행동할 수 있는 능력 향상을 갖춘 시스템을 제작하고 있다고 설명하면서 이러한 시스템에 대해 AI 에이전트(AI agents 또는 agentic systems)라고 지칭
1) Russell, S. J., & Norvig, P.  Artificial Intelligence: A Modern Approach (4th ed.). Pearson. 20212) Alan Chan 외, Visibility into AI Agents, 2024
SPRi AI Brief Special |  2024-12월호
2) 가트너는 AI 에이전트를 ‘에이젠틱 AI(Agentic AI)’로 칭하고, AI 기술을 사용하여 작업을 완료하고 목표를 달성하는 목표 중심 소프트웨어 엔터티로 정의3)∙‘에이젠틱 AI’는 명시적인 입력 없이 지침을 받고, 계획을 세우고, 도구를 사용하여 작업을 완료하며, 미리 정해진 출력을 생성하지 않고, 동적 출력을 생성할 수 있다고 설명∙AI 에이전 시 스 펙트럼 을 제 시하면 서 한쪽  끝에 는 특 정 작 업을 제 한적으 로 수 행하는  전통 적인 시스템 이 있고, 반대편 에는 환경에서 학습하고 독립적으로 결정 및 작업을 수행할 수 있는 완전한 에이전틱 AI 시스템이 있다고 설명 <AI 에이전시(AI Agency) 갭>
※ 출처: Gartner, Intelligent Agents in AI Really Can Work Alone. Here’s How, 2024.10.1£ISO/IEC는 AI 관련 표준에서 에이전트(Agent)와 AI 에이전트(AI Agent)를 구분하여 정의 ∙에이전트(Agent)는 ‘환경을 인식하고 목표를 달성하기 위해 조치를 취하는 자동화된 엔티티’라고 정의하고, AI 에이전트(AI Agent)는 “AI 기술을 사용하여 목표를 성공적으로 달성할 가능성을 최대화하는 에이전트”라고 정의(ISO/IEC DIS 22989)4)* ISO/IEC 22989에서 AI 시스템은 "인간이 정의한 목표에 대해 콘텐츠, 예측, 권장 사항 또는 결정과 같은 출력을 생성하는 엔지니어링 시스템‘이라고 정의 £AI 에이전트 관련 기능을 제공하는 기업들은 AI 에이전트(또는 유사 개념)를 기업의 특성을 반영하여 다양한 방식으로 정의 ∙세일즈포스는 AI 에이전트를 “인간의 개입 없이 고객 문의를 이해하고 응답할 수 있는 일종의 인공지능 시스템”5)으로 정의3) Gartner, Intelligent Agents in AI Really Can Work Alone. Here’s How, 2024.10.14) ISO, ISO/IEC DIS 22989(en), https://www.iso.org/obp/ui/#iso:std:iso-iec:22989:dis:ed-1:v1:en:sec:3.1.2
SPRi AI Brief Special |  2024-12월호

            -----
            출력 예시는 아래와 같습니다.
            - AI 에이전트에 대한 합의된 학술적 정의는 부재하다.
            - 스튜어트 러셀과 피터 노비그는 AI 에이전트를 '합리적 에이전트'로 정의하며, 환경을 인식하고 행동하는 시스템으로 설명했다.
            - Alan Chan은 2024년 연구에서 AI 에이전트가 자율성, 외부 도구 사용, 장기 목표 달성을 위해 적응하고 계획하며 지속적 행동할 수 있다고 설명했다.
            - 가트너는 AI 에이전트를 '에이젠틱 AI'로 칭하며, 목표 달성을 위해 동적으로 작업을 수행하는 소프트웨어로 정의했다.

            -----
            입력 : {file}
            """)

llm = ChatGoogleGenerativeAI(model="gemini-robotics-er-1.5-preview")
chain = prompt | llm
response = chain.invoke({"content": file})
print(response.content)
# with open("AI 에이전트 동향_Splitted.txt", "w", encoding="utf-8") as f:
#     f.write(f"Number of splitted documents: {len(docs)}")
#     for doc in docs:
#         f.write(f"\n\n---\n\n{doc.page_content}")
