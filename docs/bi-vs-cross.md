# Bi-Encoder vs Cross-Encoder 정리

## 왜 점수가 다를까?
- **아키텍처 차이**  
  - Bi-Encoder: 쿼리와 문서를 **독립적으로 임베딩**하고, 유사도(cosine/inner product)로 압축된 상호작용만 봄.  
  - Cross-Encoder: 쿼리와 문서를 **하나의 시퀀스로 합쳐** 토큰 간 상호작용을 직접 모델링(어텐션) → 더 풍부한 문맥 활용.
- **학습 목표 차이**  
  - Bi: 대조학습(in-batch negatives)로 “의미가 가까우면 벡터도 가깝게” 학습.  
  - Cross: 포인트/페어와이즈 랭킹 손실로 “올바른 쌍에 더 높은 점수” 학습.
- **표현력 vs 속도**  
  - Bi: 유사도 계산이 빠르지만, 상호작용 정보가 내적 점수 하나로 압축됨.  
  - Cross: 느리지만 토큰 레벨 상호작용을 살려 정밀도가 높음.

## 추천 파이프라인 (두 단계 혼합)
1) **1차 회수(Bi-Encoder + ANN)**: 문서 임베딩을 미리 캐싱 → 쿼리 임베딩 1회 → cosine/내적으로 top-k 후보.  
2) **2차 재랭킹(Cross-Encoder)**: top-k 후보 각각을 쿼리와 함께 넣어 점수 재계산 → 더 높은 정확도.

```mermaid
flowchart LR
  subgraph Stage1[1차 회수: Bi-Encoder]
    Q1[쿼리 인코딩]
    D1[문서 임베딩(오프라인)]
    ANN[벡터 검색 (cosine/inner product)]
  end
  subgraph Stage2[2차 재랭킹: Cross-Encoder]
    Pair[(쿼리, top-k 문서)]
    CE[Cross-Encoder]
    Rerank[최종 정렬]
  end
  Q1 --> ANN
  D1 --> ANN
  ANN -->|top-k| Pair
  Pair --> CE --> Rerank
```

## ContextualCompressionRetriever 흐름
- 구성: `base_retriever`(벡터 검색, Bi-Encoder 임베딩) + `base_compressor`(재평가/정렬, Cross-Encoder reranker 등).
- 단계:
  1) 쿼리 임베딩 → `base_retriever`가 top-k 후보 회수.
  2) 후보들을 쿼리와 짝지어 `base_compressor`가 점수화/필터링/재정렬.
  3) 점수 높은 순으로 반환.  
- 즉, **Retrieval(Bi-Encoder) → Re-Ranker(Cross-Encoder)** 2단계를 한 번에 수행하는 래퍼.

## 임베딩 함수의 역할 (Bi-Encoder 관점)
- 임베딩 함수는 **문서/쿼리를 벡터로 만드는 단계**가 전부다. 벡터 간 유사도 점수(코사인/내적)는 벡터스토어·리트리버가 계산한다.
- Bi-Encoder 임베딩은 “점수까지 매긴다”가 아니라, “점수를 계산할 수 있는 표현을 만든다”가 정확한 설명이다.
- 실무 팁: 임베딩 모델을 바꾸면 (1) 저장된 문서 벡터를 다시 만들고, (2) 쿼리 임베딩과의 스페이스 일관성을 확인해야 한다.

## Vector Store 선택 가이드
- InMemoryVectorStore
  - 메모리 상주, 영속성 없음. 소량 데이터·데모·실험용으로 간단히 사용.
  - 고급 인덱스나 필터링, 운영 기능(분산/복제/백업/보안) 없음.
- ANN/영속 백엔드(FAISS, Chroma/Qdrant, Pinecone, Milvus, Elasticsearch/OpenSearch, Redis, PGVector 등)
  - 디스크/클러스터에 저장, 재시작 후 유지. HNSW/IVF 등 고성능 인덱스 제공.
  - 메타데이터 필터, 하이브리드 검색(BM25+벡터), ACL, 모니터링 등 운영 기능 지원.
  - 비용·설정·운영 복잡도가 있지만, 대규모·저지연 요구에 필수적.
- 요약: 벡터스토어는 “저장소”를 넘어 검색 품질·속도·운영 특성을 결정하는 핵심 컴포넌트다. 데이터 규모, 지연 요구, 필터/보안/운영 요건에 맞춰 선택한다.

## 언제 무엇을 쓰나?
- 대규모 코퍼스, 낮은 지연/비용이 중요: **Bi-Encoder**로 1차 회수.
- 품질 최우선, 후보 수 적음: **Cross-Encoder**로 재랭킹.
- 실무 표준: Bi로 top-k 회수 → Cross로 재랭킹.

## 비용/성능 트레이드오프
- Bi: 오프라인 문서 임베딩 + 온라인 쿼리 1회 → 매우 빠름, ANN(HNSW/FAISS)와 궁합 좋음.
- Cross: 쿼리마다 (쿼리,문서) 쌍을 모두 인코딩 → 느리지만 정밀.
- distillation: Cross의 강한 점수를 teacher로 사용해 Bi를 student로 개선.

## 코드 참고
- `P2/9-embedding.py`: 코사인 유사도 top-k 회수 후 `BAAI/bge-reranker-v2-m3` Cross-Encoder로 재랭킹 예제.

