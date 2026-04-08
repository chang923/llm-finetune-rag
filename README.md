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
之后你需要做的是进入 my_yaml/train_lora/my_yaml/train_lora ，按照你的要求修改配置，配置的参数对应在里面已经给出，然后在 LLaMA-Factory 下进行 llamafactory-cli train ./my_yaml/train_lora/my_yaml/train_lora/my_yaml/train_lora/my_yaml/train_lora
接下来你需要进行一下本地的 chat 测试，在这之后是 API 部署，这两个都是可以使用 my_yaml 下的 chat_lora，运行指令 llamafactory-cli chat/api ./my_yaml/train_lora/my_yaml/chat_lora/my_yaml/train_lora/my_yaml/train_lora
在这个里面你需要导入初始的基座模型和微调的 LoRA 矩阵。以及生成时候的 T 参数

### 3. RAG知识库接入


