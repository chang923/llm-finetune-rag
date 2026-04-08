"""导入"""
from langchain_community.document_loaders import TextLoader

loader = TextLoader(
    file_path = "../劳动法/data.json"
)

documents = loader.load()

# print(documents)





"""制作切分器"""

from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=0, 
    separator=",\n", 
    keep_separator=False
)





"""进行切分"""
segments = splitter.split_documents(documents)

# print(f"\n共得到{len(segments)}个块\n")
# for segment in segments:
#     print(segment.page_content)
#     print("===================")




"""制作bm25_retriever"""
from langchain_community.retrievers import BM25Retriever
import jieba
def preprocess_func(text):
    return list(jieba.cut(text))

bm25_retriever = BM25Retriever.from_documents(
    segments,
    preprocess_func = preprocess_func  
)
import pickle

with open("./bm_retriever/bm25_retriever.pkl","wb") as f:
    pickle.dump(bm25_retriever, f)
    print("\nbm25_retriever已保存\n")


"""制作embedding"""
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name = r"../embedding_models/bge",
    model_kwargs = {"device":"cuda"},
    encode_kwargs = {"normalize_embeddings":"True"}
)

"""制作chroma"""
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents = segments,
    persist_directory = r"./chroma_db",
    embedding = embedding
)

print(f"向量库构建完成！共存储 {vectorstore._collection.count()} 个向量片段。")
print(f"向量数据已保存到目录：./chroma_db")