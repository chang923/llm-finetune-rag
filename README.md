# LLM Fine-Tuning & RAG

本项目包含两个主要部分：
- **fine-tuning**：基于 LLaMA-Factory 进行大模型微调
- **rag**：检索增强生成（RAG）示例，用于知识库问答

## 项目结构
llm-finetune-rag/
├── finetune/ # 微调相关代码（LLaMA-Factory）
│ └── LLaMA-Factory/ # 官方微调框架（可链接或子模块）
├── rag/ # RAG 实现（如向量数据库、检索链）
└── .gitignore


## 环境要求

- Python 3.10+
- CUDA 11.8+（如需 GPU 微调）
- 依赖包：见各子目录下的 `requirements.txt`

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/chang923/llm-finetune-rag.git
cd llm-finetune-rag
```

### 2. LoRA微调

本项目使用的是 LLama-factory 框架，针对 qwen3-4B-Instruct 进行了微调，使得其回复风格带有了猫娘色彩
项目数据集存在于 ./finetune/LLaMA-Factory/my_data/muice-dataset/results_glm-4-flash.json 中，其符合 alpaca 格式，具体说就是：
prompt:
query:
response:
首先你需要做的是修改 data/dataset_info.json中的cat中的路径，进行你本地的注册
之后你需要做的是进入 my_yaml/train_lora/my_yaml/train_lora ，按照你的要求修改配置，配置的参数对应在里面已经给出
然后在 LLaMA-Factory 下进行 llamafactory-cli train ./my_yaml/train_lora/llama3_lora_sft.yaml
接下来你需要进行一下本地的 chat 测试，在这之后是 API 部署，这两个都是可以使用 my_yaml 下的 chat_lora
运行指令 llamafactory-cli chat/api ./my_yaml/train_lora/llama3_lora_sft.yaml
在这个里面你需要导入初始的基座模型和微调的 LoRA 矩阵。以及生成时候的 T 参数

### 3. RAG知识库接入

这里实际上分为两个部分
index.py和chat.py
下面分别看这两个的内容
index.py中是知识库嵌入的部分，即RAG中的index准备，这里使用了 reg 作为 embedding model，用作相似度检索，向量数据库使用了 chroma 数据库；用 bm25 retriever 进行关键词检索。
首先是对数据分块，先用TextLoarer导入为documents，然后用CharacterTextSplitter转换为segments。
首先是导入了 embedding 模型，然后通过其构建了数据库对应的 vectorstore，再调用 reg_retriever = vectorstore.as_retriever() 弄到存储检索
之后就是使用了 bm25retriever，在这之中需要使用分词器，使用的是 jieba 分词器，首先需要定义分词函数preprocess_func(text): return list(jieba.cut(text))，为了在后续的使用中能调用这个关键词检索器，我们需要调用 pickle 进行保存，二进制保存，之后二进制读取(pickle.dump(bm25_retriever , f)/pickle.load(bm25_retriever , f))

chat.py是增强生成部分的内容，首先是导入index.py中的向量数据库和关键词识别的bm25，这两个一个导入embedding，chroma；一个制作preprocess_func(text)，bm25_retriever。之后使用融合模型，给予其对应权重，采用平滑打分，x = 10

接下来就是ChatPromptTemplate用来提示词构建，提示词包括system和human，第一个是模型人格固定用，第二个是提示词，里面有问题和检索的文本，{question}和{context}。这是要后面补充。
模型调用，这里可以是之前微调的 api，ChatOpenAI符合open-ai格式，也可以是ChatDeepSeek调用deepseek，作为llm
之后是循环输入问题，然后使用混合模型.invoke得到top-k，之后作为{question}和{context}来prompt.format_messages(question=user_input, context=context)
调用ans = llm.invoke(prompt)
输出ans.context即可

### 4. 评价部分
这部分在 retriever_evaluate.py 中，主要是评价“用户问题 -> TOP-K文本的好坏”
这里面有四个评价标准：命中率，召回率，准确率，MMR
命中率是有为1 没有为0
召回率是 K中相关/所有相关
准确率是 K中相关/K中所有
MMR是 1/K中第一个相关的，没有的话为0
使用了30条文本作为“问题：答案”对
其中15条正向查询，15条反向查询，数据在“/劳动法/test_questions.jsonl”
