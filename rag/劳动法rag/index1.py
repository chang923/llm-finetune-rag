from langchain_community.document_loaders import TextLoader

loader = TextLoader(r"../劳动法/data.json")

documents = loader.load()

from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 0,
    separator = ",\n",
    keep_separator = True
)

segments = splitter.split_documents(documents)

from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name = r"../embedding_models/bge",
    model_kwargs = {"device":"cuda"},
    encode_kwargs = {"normalize_embeddings":True}
)

from langchain_chroma import Chroma

persist_directory = r"./chroma1_db"

vectorstore = Chroma.from_documents(
    persist_directory = persist_directory,
    documents = segments,
    embedding = embedding
)

print(f"拆分成功，一共拆分{vectorstore._collection.count()}个向量片段")
print(f"存储在{persist_directory}中！")