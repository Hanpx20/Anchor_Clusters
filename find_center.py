import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
# 读取tsv文件
df = pd.read_csv('../clusters/7/queries.train.tsv', sep='\t', names=['id', 'text'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载预训练模型和对应的tokenizer
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# 将句子转为embedding
embeddings = []
for sentence in tqdm(df['text'][:50000]):
    inputs = tokenizer(sentence, return_tensors='pt').to(device)
    outputs = model(**inputs)
    embeddings.append(outputs.last_hidden_state.mean(1).cpu().detach().numpy())

print("Tokenize Finished!")

embeddings = torch.Tensor(embeddings)

# 计算中心点
center = embeddings.mean(dim=0)

# 计算所有句子到中心点的距离
distances = cosine_distances(embeddings, center.reshape(1, -1))

# 将距离添加到df中
df['distance'] = distances

# 按照距离排序，选出前100个最近的句子
top_ = df.sort_values(by='distance')[:200]

# 输出代表性句子
with open("7new.txt", "w") as file:
    file.writelines(top_[['id', 'sentence']])
print(top_[['id', 'sentence']])