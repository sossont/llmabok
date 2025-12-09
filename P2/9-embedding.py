# pip install sentence-transformers
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")

# docs = [
#     "한국의 수도는 서울입니다.",
#     "중국의 수도는 북경입니다.",
#     "삼성전자의 본사는 수원에 있습니다.",
#     "파이썬은 프로그래밍 언어입니다.",
#     "부산은 한국의 제2수도라 불립니다.",
#     "대한민국의 수도는 서울입니다.",
#     "미국의 수도는 워싱턴입니다.",
#     "겨울에는 수도관 동파가 자주 일어납니다.",
#     "대한민국은 수도권에 사람이 매우 많습니다."
# ]

# query = "한국의 수도는?"

# embedded_docs = embeddings.embed_documents(docs)
# embedded_query = embeddings.embed_query(query)

# from sklearn.metrics.pairwise import cosine_similarity
# similarity = cosine_similarity([embedded_query], embedded_docs)
# print(similarity)

docs = [
"사과는 빨간색도 있고, 초록색도 있습니다. 나는 빨간 사과는 좋아하지만 초록 사과는 싫어합니다.",
"바나나는 노란색입니다. 바나나는 맛있습니다. 나는 바나나를 좋아합니다.",
"포도는 보라색입니다. 포도는 작고 달콤합니다. 포도는 씨가 있어서 먹기 불편합니다. 나는 포도를 싫어합니다.",
"나는 낚시를 좋아합니다. 낚시는 물고기를 잡는 재미가 있습니다. 낚시는 자연과 함께하는 활동입니다.",
"나는 여행을 좋아합니다. 여행을 통해 다양한 과일을 맛볼 수 있습니다.",
]
query = "나는 어떤 과일을 좋아할까요?"

embedded_docs = embeddings.embed_documents(docs)
embedded_query = embeddings.embed_query(query)

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity([embedded_query], embedded_docs)[0]

top_k = min(5, len(docs))  # 여기서는 5문장 모두
top_k_indices = sorted(range(len(similarity)), key=lambda i: similarity[i], reverse=True)[:top_k]
top_k_docs = [docs[i] for i in top_k_indices]

cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
cross_scores = cross_encoder.score([(query, doc) for doc in top_k_docs])

reranked = sorted(zip(top_k_docs, cross_scores), key=lambda x: x[1], reverse=True)

print("코사인 유사도 top-k 인덱스:", top_k_indices)
print("코사인 유사도 점수:", [similarity[i] for i in top_k_indices])
print("Cross-Encoder 재랭킹 결과:")
for doc, score in reranked:
    print(f"{score:.4f} | {doc[:60]}...")