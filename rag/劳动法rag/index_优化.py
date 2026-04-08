from langchain_community.document_loaders import TextLoader

loader = TextLoader("../劳动法/data.json")
documents = loader.load()

from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size = 60,
    chunk_overlap = 0,
    separator = ",\n",
    keep_separator = False
)

segments = splitter.split_documents(documents)

from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name = "../embedding_models/bge",
    model_kwargs = {"device":"cuda"},
    encode_kwargs = {"normalize_embeddings":"True"}
)

from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    persist_directory = "./chroma_db_优化",
    embedding = embedding,
    documents = segments
)

from langchain_community.retrievers import BM25Retriever
import jieba
def preprocess_func(text):
    return list(jieba.cut(text))

bm25_retriever = BM25Retriever.from_documents(
    segments,
    preprocess_func = preprocess_func
)

import pickle
with open("./bm_retriever/bm25_retriever_优化.pkl","wb") as f:
    pickle.dump(bm25_retriever , f)

print("完成！")