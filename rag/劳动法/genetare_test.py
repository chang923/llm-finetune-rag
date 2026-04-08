import json
import random

# 加载原始数据
with open("data.json", "r", encoding="utf-8") as f:
    law_data = json.load(f)

# 将两个对象合并为一个法条列表
articles = []
for law_obj in law_data:
    for key, value in law_obj.items():
        # key 格式如 "中华人民共和国劳动合同法 第三条"
        articles.append({
            "law": key,
            "content": value.strip()
        })

# 确保随机性一致
random.seed(42)

# 选取 15 条正向（问编号，找内容）和 15 条反向（问内容，找编号）
positive_articles = random.sample(articles, 15)
negative_articles = random.sample([a for a in articles if a not in positive_articles], 15)

test_cases = []

# 正向问题：问法条编号，期望检索到内容
for art in positive_articles:
    test_cases.append({
        "type": "forward",
        "question": f"{art['law']}的内容是什么？",
        "relevant_text": art["content"],   # 匹配用完整内容
        "law_id": art['law']
    })

# 反向问题：问内容，期望检索到法条编号
for art in negative_articles:
    # 提取一小段有代表性的内容作为问句
    # 这里简单取前100个字符（或整个内容，根据实际情况）
    content_excerpt = art["content"][:150]  # 取前150字作为问题
    test_cases.append({
        "type": "reverse",
        "question": f"“{content_excerpt}...”是哪一条法律？",
        "relevant_text": art["law"],        # 匹配时检查法条编号
        "law_id": art['law']
    })

# 打乱顺序
random.shuffle(test_cases)

# 保存为 JSONL 格式（每行一个 JSON）
with open("test_questions.jsonl", "w", encoding="utf-8") as f:
    for case in test_cases:
        f.write(json.dumps(case, ensure_ascii=False) + "\n")

print(f"已生成 {len(test_cases)} 条测试数据，保存至 test_questions.jsonl")