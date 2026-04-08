import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings import Embeddings
from typing import List

class Qwen3Embeddings(Embeddings):
    def __init__(self, model_path: str, device: str = "cuda", use_flash_attention: bool = False):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
        model_kwargs = {"trust_remote_code": True}
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        self.model = AutoModel.from_pretrained(model_path, **model_kwargs).to(device)
        self.model.eval()
        self.task_description = "Given a web search query, retrieve relevant passages that answer the query"

    def last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_detailed_instruct(self, query: str) -> str:
        return f'Instruct: {self.task_description}\nQuery:{query}'

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 确保每个元素都是字符串，并过滤掉 None 值
        texts = [str(t) if t is not None else "" for t in texts]
        if not texts:
            return []  # 空输入返回空列表
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = self.last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings = embeddings.to(torch.float32)  # 确保转换为 float32
        return embeddings.cpu().numpy().tolist()

    def embed_query(self, text: str) -> List[float]:
        text_with_instruct = self.get_detailed_instruct(text)
        return self.embed_documents([text_with_instruct])[0]