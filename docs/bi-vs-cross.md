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

## 언제 무엇을 쓰나?
- 대규모 코퍼스, 낮은 지연/비용이 중요: **Bi-Encoder**로 1차 회수.
- 품질 최우선, 후보 수 적음: **Cross-Encoder**로 재랭킹.
- 실무 표준: Bi로 top-k 회수 → Cross로 재랭킹.

## 비용/성능 트레이드오프
- Bi: 오프라인 문서 임베딩 + 온라인 쿼리 1회 → 매우 빠름, ANN(HNSW/FAISS)와 궁합 좋음.
- Cross: 쿼리마다 (쿼리,문서) 쌍을 모두 인코딩 → 느리지만 정밀.
- distillation: Cross의 강한 점수를 teacher로 사용해 Bi를 student로 개선.

## 코드 참고
- `P2/9-embedding.py`: 코사인 유사도 top-k 회수 후 `BAAI/bge-reranker-v2-m3` Cross-Encoder로 재랭킹 예제.***

