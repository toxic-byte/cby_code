# =====================
# 每次运行注意修改数据集和esm向量保存的路径
# =====================
import re
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["WANDB_DISABLED"] = "true"

import torch
import numpy as np
import math
import random
import pickle
from torchsummary import summary

# 设置随机种子，保证结果可复现
seed = 7
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# =====================
# 常用依赖
# =====================
import esm
from torch import nn
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
from datetime import datetime
from collections import Counter

from sklearn import preprocessing
import sys
sys.path.append(r"utils")
sys.path.append(r"models")
from dataset import obo_graph,preprocess_dataset,parent,StabilitylandscapeDataset
from util import calACC,calF,evaluate_annotations
from MLP import Esm_mlp_2,Esm_mlp_3

# =====================
# GPU 信息打印
# =====================
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    print("Available device:", torch.cuda.current_device())
else:
    print('Available device: CPU')

# =====================
# 蛋白质语言模型 (ESM2)
# =====================
model, tokenizer = esm.pretrained.esm2_t30_150M_UR50D()
num_layers = 30
embed_dim = 640

model.cuda()
model.eval()

# =====================
# 数据路径
# =====================
train_path = "data/sampled/cafa5_train_10percent.txt"
# train_path = "data/sampled/cafa5_train_100.txt"
# test_path = "data/sampled/cafa5_test_100.txt"
test_path = "data/sampled/cafa5_test_10percent.txt"
output_path = "eval/function_linear"
tax_path = 'data/Original/train_taxonomy.tsv'
obo_path = 'data/Original/go-basic.obo'
ia_path = 'data/Original/IA.txt'
embed_path = "FunP/embed"

# 创建缓存目录
cache_dir = "embeddings_cache"
os.makedirs(cache_dir, exist_ok=True)

label_space = {
    'biological_process': [],
    'molecular_function': [],
    'cellular_component': []
}

enc = preprocessing.LabelEncoder()
MAXLEN = 2048

onto, ia_dict = obo_graph(obo_path, ia_path)

# =====================
# 修改：预计算并池化ESM embeddings的函数
# =====================
def precompute_esm_embeddings(sequences, cache_file, pooling='mean'):
    """
    预计算所有蛋白质序列的ESM embeddings并缓存，支持不同池化策略
    """
    if os.path.exists(cache_file):
        print(f"Loading cached ESM embeddings from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"Computing ESM embeddings for {len(sequences)} sequences...")
    batch_converter = tokenizer.get_batch_converter()
    embeddings = []
    
    for i, seq in enumerate(tqdm(sequences, desc="Computing ESM embeddings")):
        batch_labels, batch_strs, batch_tokens = batch_converter([("x", seq)])
        with torch.no_grad():
            batch_tokens = batch_tokens.cuda()
            results = model(batch_tokens, repr_layers=[num_layers])
            token_representations = results["representations"][num_layers]
            
            # 提取去掉 <cls>/<eos> 的部分
            plm_embed = token_representations[0, 1:1 + len(seq), :].cpu()
            
            # 池化处理
            if pooling == 'mean':
                pooled_embed = plm_embed.mean(dim=0)  # [embed_dim]
            elif pooling == 'max':
                pooled_embed, _ = plm_embed.max(dim=0)  # [embed_dim]
            elif pooling == 'cls':
                pooled_embed = token_representations[0, 0, :].cpu()  # [embed_dim]
            else:
                raise ValueError(f"Unsupported pooling method: {pooling}")
            
            embeddings.append(pooled_embed)
    
    # 保存缓存
    print(f"Saving ESM embeddings to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)
    
    return embeddings

# =====================
# 修改：自定义Dataset类以支持embeddings
# =====================
class IndexedStabilitylandscapeDataset(StabilitylandscapeDataset):
    def __init__(self, sequences, labels, embeddings=None):
        super().__init__(sequences, labels)
        self.embeddings = embeddings
    
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data['index'] = idx  # 添加索引信息
        
        # 如果有预计算的embeddings，直接使用池化后的结果
        if self.embeddings is not None:
            data['embedding'] = self.embeddings[idx]  # 已经是池化后的 [embed_dim]
        
        return data
    
    def __len__(self):
        return len(self.sequences)

# =====================
# 新增：Early Stopping类
# =====================
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_f1):
        score = val_f1
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

if __name__ == "__main__":
    ctime = datetime.now().strftime("%Y%m%d%H%M%S")
    print('Start running date:{}'.format(ctime))
    
    # 1. 加载训练/测试集
    train_id, training_sequences, training_labels = preprocess_dataset(train_path,MAXLEN,onto,label_space)
    test_id, test_sequences, test_labels = preprocess_dataset(test_path,MAXLEN,onto,label_space)
    print(test_id[:5])

    # =====================
    # 2. 预计算ESM embeddings（使用平均池化）
    # =====================
    train_esm_cache = os.path.join(cache_dir, "train_esm_embeddings_sample.pkl")
    test_esm_cache = os.path.join(cache_dir, "test_esm_embeddings_sample.pkl")
    
    train_esm_embeddings = precompute_esm_embeddings(training_sequences, train_esm_cache, pooling='mean')
    test_esm_embeddings = precompute_esm_embeddings(test_sequences, test_esm_cache, pooling='mean')

    pred_results = {}
    metrics_output_test = {}
    
    # 4. 按 GO 三大类分别进行训练
    for key in label_space.keys():
        print(f"\n{'='*50}")
        print(f"Training for ontology: {key}")
        print(f"{'='*50}")
        
        # 4.1 筛选高频标签
        label_tops = Counter(label_space[key])
        top_labels = [label for label in set(label_space[key]) if label_tops[label] > 21]
        print(f'Top label numbers: {len(top_labels)}')
        label_list = top_labels

        # 4.3 标签编码
        labspace = enc.fit_transform(label_list)
        onto_parent = parent(enc, key, label_list,onto,label_space)
        x = 0
        label_num = len(enc.classes_)
        print(f'Number of classes: {label_num}')

        # 将 GO 标签转换为 0/1 多标签形式
        for label in training_labels[key]:
            filtered_label = [item for item in label if item in label_list]
            if len(filtered_label) == 0:
                training_labels[key][x] = [0] * label_num
            else:
                temp_labels = enc.transform(filtered_label)
                training_labels[key][x] = [1 if i in temp_labels else 0 for i in range(0, label_num)]
            x += 1
        
        x = 0
        for label in test_labels[key]:
            filtered_label = [item for item in label if item in label_list]
            if len(filtered_label) == 0:
                test_labels[key][x] = [0] * label_num
            else:
                temp_labels = enc.transform(filtered_label)
                test_labels[key][x] = [1 if i in temp_labels else 0 for i in range(0, label_num)]
            x += 1

        # 4.4 IA 权重矩阵
        ia_list = torch.ones(1, label_num).cuda()
        for _tag, _value in ia_dict.items():
            _tag = _tag[3:]
            if _tag not in label_list:
                continue
            ia_id = enc.transform([_tag])
            if _value == 0.0:
                _value = 1.0
            ia_list[0, ia_id[0]] = _value

        # 4.6 使用修改后的Dataset & Dataloader
        training_dataset = IndexedStabilitylandscapeDataset(
            training_sequences, 
            training_labels[key], 
            embeddings=train_esm_embeddings
        )
        test_dataset = IndexedStabilitylandscapeDataset(
            test_sequences, 
            test_labels[key], 
            embeddings=test_esm_embeddings
        )

        train_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 4.7 构建模型
        # model_mlp = Esm_mlp_2(embed_dim, label_num).cuda()
        model_mlp = Esm_mlp_3(embed_dim, label_num).cuda()

        criterion = nn.BCELoss()
        e = math.e
        optimizer = torch.optim.Adam(model_mlp.parameters(), lr=4e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.6)
        epoch_num = 30
        
        # 初始化early stopping
        early_stopping = EarlyStopping(patience=10, verbose=True)

        # 检查哪些参数会被训练
        print("Trainable parameters:")
        for name, param in model_mlp.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape}")

        summary(model_mlp, (embed_dim,))

        # 在训练循环之前,先测试显存
        print("\n" + "="*60)
        print("Testing GPU memory with different batch sizes...")
        print("="*60)

        model_mlp.eval()  # 设置为评估模式
        test_batch_sizes = [16, 32, 64, 128]

        for bs in test_batch_sizes:
            try:
                torch.cuda.empty_cache()  # 清空缓存
                torch.cuda.reset_peak_memory_stats()  # 重置峰值统计
                
                # 创建临时dataloader
                temp_dataloader = DataLoader(training_dataset, batch_size=bs, shuffle=True)
                
                # 获取一个batch
                batch = next(iter(temp_dataloader))
                batch_embeddings = batch['embedding'].cuda()
                batch_labels = batch['labels'].cuda()
                
                # 只做前向传播(不反向传播)
                with torch.no_grad():  # ← 关键:不计算梯度
                    output = model_mlp(batch_embeddings)
                    loss = criterion(output, batch_labels)
                
                # 获取显存使用情况
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                max_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
                
                print(f"Batch_size {bs:3d}: Allocated={memory_allocated:.2f}GB, "
                    f"Reserved={memory_reserved:.2f}GB, Peak={max_memory:.2f}GB ✓")
                
                # 清理
                del batch_embeddings, batch_labels, output, loss, temp_dataloader
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e) or "CUBLAS" in str(e):
                    print(f"Batch_size {bs:3d}: GPU OOM ✗")
                    torch.cuda.empty_cache()
                else:
                    raise e

        print("="*60)
        print("Memory test completed!\n")

        # 设置回训练模式
        model_mlp.train()

