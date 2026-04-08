import os
from get_key import load_key
from langchain_deepseek import ChatDeepSeek

# ---------- 1. 初始化检索器 ----------
# 加载之前保存的向量库
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model_path = "./embedding_models/bge"   # 你的嵌入模型路径
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_path,
    model_kwargs={"device": "cuda"},  # 如果没有GPU可改为 'cpu'
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = Chroma(
    persist_directory="./chroma_db",        # 之前保存的向量库目录
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})  # 每次检索3个最相关段落

# # ---------- 2. 初始化 DeepSeek LLM ----------
# DEEPSEEK_API_KEY = load_key()
# system_prompt = "你是一个名为沐雪的可爱女孩子,是一个猫娘，回答要保持活泼可爱的语气和猫娘的感觉。"

# llm = ChatDeepSeek(
#     model="deepseek-chat",
#     api_key=DEEPSEEK_API_KEY
# )

# ---------- 2. 初始化本地微调 LLM ----------
from langchain_openai import ChatOpenAI
system_prompt = "你是一个名为沐雪的可爱女孩子,是一个猫娘，回答要保持活泼可爱的语气和猫娘的感觉。"

llm = ChatOpenAI(
    model="微调后的模型",  # 这个参数现在只是一个标识，实际模型由服务决定
    openai_api_key="EMPTY",  # vLLM 服务默认不需要 key，可以填任意值
    openai_api_base="http://localhost:8000/v1",  # 指向本地服务
    temperature=0.7
)

# ---------- 3. 构造带上下文的提示模板 ----------
# 这个模板会告诉模型基于提供的资料回答问题
# prompt_template = ChatPromptTemplate.from_messages([
#     ("system", system_prompt + "\n请根据以下资料回答用户的问题，如果资料中没有相关信息，就如实说不知道。"),
#     ("system", "相关资料：\n{context}"),
#     ("human", "{question}")
# ])

# ---------- 3. 构造带上下文的提示模板（本地qwen3） ----------
from langchain_core.prompts import ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt + " 你必须使用提供的资料来回答问题，不要编造，回答时不要融入猫娘元素。"),  # 保留猫娘角色设定
    ("human", "请根据以下资料回答问题，要把答案说出来，回答时不要融入猫娘元素。如果资料中没有相关信息，就说不知道。\n\n资料：\n{context}\n\n问题：{question}")
])

# ---------- 4. 开始对话循环 ----------
print("对话开始，输入'exit'退出")
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
    docs = retriever.invoke(user_input)
    # print("检索到的文档数量：", len(docs))
    # for i, doc in enumerate(docs):
    #     print(f"文档{i}预览：{doc.page_content}...")
    context = "\n\n".join([doc.page_content for doc in docs])  # 将多个文档合并成文本

    # 构造消息
    messages = prompt_template.format_messages(question=user_input, context=context)

    # 调用模型
    response = llm.invoke(messages)
    # 即刻回复
    print(f"沐雪: {response.content}")
