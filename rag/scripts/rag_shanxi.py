import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate   # 修改了这一行
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from qwen3_embedding_official import Qwen3Embeddings

# ====== 配置路径（保持不变）=======
base_model_path = "/home/chang/llama-factory/LLaMA-Factory/models/Qwen/Qwen3-0.6B"
lora_path = "/home/chang/llama-factory/LLaMA-Factory/saves/Qwen/Qwen3-0.6B/lora/sft"
embed_model_path = "/home/chang/rag/rag_models/Qwen3-Embedding-0.6B"
doc_path = "/home/chang/rag/rag_data/shanxi_geo.txt"
persist_dir = "/home/chang/rag/chroma_db_qwen"

# ====== 1. 加载文档并切分 ======
loader = TextLoader(doc_path, encoding='utf-8')
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
print(f"切分为 {len(docs)} 个文本块")

# ====== 2. 初始化嵌入模型 ======
embeddings = Qwen3Embeddings(embed_model_path, device="cuda")

# ====== 3. 创建或加载向量数据库 ======
if os.path.exists(persist_dir):
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    print("从本地加载已有向量数据库")
else:
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
    vectorstore.persist()
    print("创建并保存向量数据库")

# ====== 4. 加载你的微调模型 ======
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto"
)
model = PeftModel.from_pretrained(model, lora_path)
model.eval()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    repetition_penalty=1.2,
    do_sample=True,
    device_map="auto",
    return_full_text=False
)
llm = HuggingFacePipeline(pipeline=pipe)

# ====== 5. 定义沐雪角色提示模板 ======
template = """你是一个名为沐雪的可爱AI女孩子。请根据以下资料回答问题。如果资料中没有相关信息，你可以根据自己的知识回答，但请保持沐雪的语气。注意：只回答用户当前的问题，不要生成新的问题。

资料：
{context}

问题：{question}
沐雪的回答："""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# ====== 6. 构建 RAG 问答链（使用自定义prompt） ======
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
    verbose=False
)

# ====== 7. 交互式问答 ======
print("\nRAG 问答系统已启动（输入 'exit' 退出）")
while True:
    query = input("\n用户: ")
    if query.lower() in ['exit', 'quit']:
        break
    result = qa_chain({"query": query})
    print("\n沐雪:", result['result'])
    #print("\n参考文档:", [doc.page_content for doc in result['source_documents']])