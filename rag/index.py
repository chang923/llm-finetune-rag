#导入文本为documents
from langchain_community.document_loaders import TextLoader

loader = TextLoader("./退款.txt")
documents = loader.load()

#print(documents)

from langchain_text_splitters import CharacterTextSplitter

#制作拆分方式
text_spliter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0, separator="\n\n", keep_separator=True)

segments = text_spliter.split_documents(documents)

# print(len(segments))

# for segment in segments:
#     print(segment.page_content)
#     print("------")

#加载本地嵌入模型
from langchain_huggingface import HuggingFaceEmbeddings

model_path = "./embedding_models/bge"

embedding = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs={"device": "cuda"},  # 配置模型环境
    encode_kwargs={"normalize_embeddings": True}  # 配置向量化编码环境
)


# 导入 Chroma 向量库
from langchain_chroma import Chroma

# 4. 创建 Chroma 向量库，将文档块转换为向量并持久化
#    指定一个目录来保存向量数据（如果没有会自动创建）
persist_directory = "./chroma_db"

vectorstore = Chroma.from_documents(
    documents=segments,           # 上一步切分好的文档块列表
    embedding=embedding,          # 加载好的本地嵌入模型
    persist_directory=persist_directory  # 持久化目录
)

# print(f"向量库构建完成！共存储 {vectorstore._collection.count()} 个向量片段。")
# print(f"向量数据已保存到目录：{persist_directory}")

# # 5. （可选）测试检索功能
# #    将向量库转换为检索器，设置返回最相关的3个块
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# # 模拟用户提问
# query = "申请退款后，款项多久可以退回？"
# print(f"\n用户问题：{query}")
# print("检索到的相关段落：")

# docs = retriever.invoke(query)
# for i, doc in enumerate(docs, 1):
#     print(f"\n--- 第 {i} 段 ---")
#     print(doc.page_content)