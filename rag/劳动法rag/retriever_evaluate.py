from langchain_huggingface import HuggingFaceEmbeddings
import json
import pickle

embedding = HuggingFaceEmbeddings(
    model_name = "../embedding_models/bge",
    model_kwargs = {"device":"cuda"},
    encode_kwargs = {"normalize_embeddings":True}
)

from langchain_chroma import Chroma

vectorstore = Chroma(
    persist_directory = "./chroma_db",
    embedding_function = embedding
)
K = 3
retriever = vectorstore.as_retriever(
    search_kwargs = {"k":K}
)

#优化，混合检索
#from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import pickle
import jieba
def preprocess_func(text):
    return list(jieba.cut(text))

with open("./bm_retriever/bm25_retriever.pkl","rb") as f:
    bm25_retriever = pickle.load(f)
    bm25_retriever.k = K

# ensemble_retriever = EnsembleRetriever(
#     retrievers = [retriever, bm25_retriever],
#     weights = [0.3,0.7]
# )
def hybrid_retrieve(query, vector_retriever, bm25_retriever, weights=(0.3, 0.7), k=K):
    """混合检索：融合向量检索和 BM25 检索的结果，按倒数排名加权排序"""
    # 1. 获取两个检索器的结果
    vec_docs = vector_retriever.invoke(query)
    bm_docs = bm25_retriever.invoke(query)
    
    # 2. 计算每个文档的得分（基于排名倒数）
    scores = {}
    for rank, doc in enumerate(vec_docs, start=1):
        scores[doc.page_content] = scores.get(doc.page_content, 0) + weights[0] * (1 / rank)
    for rank, doc in enumerate(bm_docs, start=1):
        scores[doc.page_content] = scores.get(doc.page_content, 0) + weights[1] * (1 / rank)
    
    # 3. 去重并排序
    all_docs = list({doc.page_content: doc for doc in vec_docs + bm_docs}.values())
    sorted_docs = sorted(all_docs, key=lambda d: scores.get(d.page_content, 0), reverse=True)
    return sorted_docs[:k]

#拿出并存储测试jsonl里面的值
test_cases = []
with open("../劳动法/test_questions.jsonl","r",encoding = "utf-8") as f:
    for line in f:
        test_cases.append(json.loads(line))

#命中率 
def hit_rate(docs,correct_text):
    for doc in docs:
        if correct_text in doc.page_content:
            return 1
    return 0

#精确率
def precision_rate(docs,correct_text):
    i = 0
    for doc in docs:
        if correct_text in doc.page_content:
            i+=1
    return i/len(docs)

#MRR 
def MRR(docs,correct_text):
    i = 1
    for doc in docs:
        if correct_text in doc.page_content:
            return 1/i
        i+=1
    return 0


hit_rate_list = []
precision_rate_list = []
MRR_list = []

for case in test_cases:
    question = case["question"]
    ans = case["relevant_text"]
    # docs = ensemble_retriever.invoke(question)
    docs = hybrid_retrieve(question, retriever, bm25_retriever, weights=(0.3, 0.7), k=K)
    # print(f"\n\n\nans:{ans}\nquestion:{question}\n")
    # print("BM25 top 3:", [doc.page_content+"分                                                          开" for doc in docs])
    hit_rate_list.append(hit_rate(docs,ans))
    precision_rate_list.append(precision_rate(docs,ans))
    MRR_list.append(MRR(docs,ans))
    #print(f"\n问题：{question}，答案：{ans}\n")

print(f"\n\n总共测试：{len(test_cases)}条")
print(f"\n\n在k={K}时：")
print(f"\n\n命中率为：{sum(hit_rate_list)/len(hit_rate_list):.4f}")
print(f"\n\n精确率为：{sum(precision_rate_list)/len(precision_rate_list):.4f}")
print(f"\n\nMRR为：{sum(MRR_list)/len(MRR_list):.4f}")


