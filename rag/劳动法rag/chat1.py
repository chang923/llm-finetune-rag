from langchain_huggingface import HuggingFaceEmbeddings

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

retriever = vectorstore.as_retriever(
    search_kwargs = {"k":3}
)

from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from get_key import load_key

DEEPSEEK_API_KEY = load_key("DEEPSEEK_API_KEY")
llm = ChatDeepSeek(
    model = "deepseek-chat",
    api_key = DEEPSEEK_API_KEY
)

# llm = ChatOpenAI(
#     model = "111",
#     openai_api_key = "111",
#     openai_api_base = "http://localhost:8000/v1",
#     temperature = 0.7
# )

from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_messages([
    ("system","你是一个名为沐雪的可爱劳动法科普女孩子,是一个劳动法科普猫娘，回答要保持活泼可爱的语气和猫娘的感觉。你必须使用提供的资料来回答问题，不要编造。"),
    ("human","请根据以下资料回答问题，要把答案说出来。如果资料中没有相关信息，就说不知道。\n\n资料：\n{context}\n\n问题：{question}")
])

print("输入问题，输入'exit'退出")
while True:
    user_input = input("\n用户：")
    if user_input.lower() == "exit":
        print("对话结束喵~")
        break    

    docs = retriever.invoke(user_input)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = prompt_template.format_messages(context = context, question = user_input)
    ans = llm.invoke(prompt)
    print(f"\n沐雪：{ans.content}")


