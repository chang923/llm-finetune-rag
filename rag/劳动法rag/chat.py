#调用embedding和chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name = r"../embedding_models/bge",
    model_kwargs = {"device":"cuda"},
    encode_kwargs = {"normalize_embeddings":True}
)

from langchain_chroma import Chroma
K = 3
vectorstore = Chroma(
    persist_directory = r"./chroma_db_优化",
    embedding_function = embedding
)

#构建retriever
retriever = vectorstore.as_retriever(search_kwargs = {"k":K})

#使用bm25
import pickle
import jieba 
def preprocess_func(text):
    return list(jieba.cut(text))
from langchain_community.retrievers import BM25Retriever
with open("./bm_retriever/bm25_retriever_优化.pkl","rb")as f:
    bm25_retriever = pickle.load(f)
    bm25_retriever.k = K

#调用本地llm模型
from langchain_openai import ChatOpenAI

from langchain_deepseek import ChatDeepSeek
# from get_key import load_key
# DEEPSEEK_API_KEY = load_key()
# llm = ChatDeepSeek(
#      model="deepseek-chat",
#      api_key=DEEPSEEK_API_KEY
# )

llm = ChatOpenAI(
    model = "11111",
    openai_api_key = "11111",
    openai_api_base = "http://localhost:8000/v1",
    temperature = 0.7
)

#prompt提示词模板构建
from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个名为沐雪的可爱劳动法科普猫娘,是一个劳动法科普猫娘，回答要保持活泼可爱的语气和猫娘的感觉。你必须使用提供的资料来回答问题，不要编造，要加入'喵~'这样的字凸显猫娘感觉。"),
    ("human", "请根据以下资料回答问题，要把答案说出来。如果资料中没有相关信息，除非是一些常识性问题，否则的话就说不知道。\n\n资料：\n{context}\n\n问题：{question}")
])

print("对话开始：（输入'exit'退出）")
while True:
    user_input = input("\n用户：")
    if user_input.lower() == "exit":
        print("对话结束。")
        break

    # 确保 user_input 是字符串且非空
    user_input = str(user_input).strip()
    if not user_input:
        print("沐雪: 哎呀，我没有听清，能再说一遍吗喵？(｡>ㅿ<｡)")
        continue

    # --- 检索部分：捕获异常 ---
    try:
        docs = retriever.invoke(user_input)
    except Exception as e:
        print("沐雪: 哎呀，检索知识库时出了点小问题，我们重新试试吧喵～")
        continue  # 跳过本次循环，直接进入下一次输入

    # 检索相关文档
    docs = bm25_retriever.invoke(user_input)
    # print("检索到的文档数量：", len(docs))
    # for i, doc in enumerate(docs):
    #     print(f"\n\n文档{i}预览：{doc.page_content}。\n")

    #得到context
    context = "\n\n".join([doc.page_content for doc in docs])

    #构造完整消息
    prompt = prompt_template.format_messages(question=user_input, context=context)

    #调用模型
    ans = llm.invoke(prompt)
    print(f"\n沐雪：{ans.content}")
    print("     \n==================\n")



