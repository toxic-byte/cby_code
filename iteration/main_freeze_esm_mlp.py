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
    train_esm_cache = os.path.join(cache_dir, "train_esm_embeddings_test.pkl")
    test_esm_cache = os.path.join(cache_dir, "test_esm_embeddings_test.pkl")
    
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

        # 初始化记录指标
        if key not in metrics_output_test:
            metrics_output_test[key] = {
                'acc':[],
                'ma_f1':[],
                'mi_f1':[],
                'b_p':[],
                'b_r':[],
                'b_f1':[],
                'w_f1':[],
                'p':[],
                'r':[],
                'f1':[],
                'aupr':[],
            }
        best_f1 = 0
        best_model_weights = None
        optimizer_model_weights = None
        sigmoid = torch.nn.Sigmoid()

        # =====================
        # 训练/验证循环
        # =====================
        for epoch in range(epoch_num):
            # (1) 训练阶段
            model_mlp.train()
            loss_mean = 0
            
            for batch_idx, batch_data in tqdm(enumerate(train_dataloader), 
                                            desc=f"Epoch {epoch+1}/{epoch_num} Training",
                                            total=len(train_dataloader)):
                optimizer.zero_grad()
                
                # 获取batch数据
                batch_embeddings = batch_data['embedding'].cuda()  # [batch_size, embed_dim]
                batch_labels = batch_data['labels'].cuda()  # [batch_size, label_num]
                
                # 前向传播
                output = model_mlp(batch_embeddings)
                output = sigmoid(output)
                
                # 计算损失
                loss = criterion(output, batch_labels)
                loss.backward()
                optimizer.step()
                
                loss_mean += loss.item()
                
                if (batch_idx + 1) % 100 == 0:
                    print('{}  Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                        key, epoch + 1, epoch_num, batch_idx + 1,
                        len(train_dataloader),
                        loss_mean / (batch_idx + 1)))
            
            scheduler.step()

            # (2) 验证阶段
            model_mlp.eval()
            _labels = []
            _preds = []
            weight_preds = []
            
            with torch.no_grad():
                for batch_idx, batch_data in tqdm(enumerate(test_dataloader),
                                                desc=f"Epoch {epoch+1}/{epoch_num} Validation",
                                                total=len(test_dataloader)):
                    # 获取batch数据
                    batch_embeddings = batch_data['embedding'].cuda()
                    batch_labels = batch_data['labels']
                    
                    # 前向传播
                    output = model_mlp(batch_embeddings)
                    
                    # 加入 IA 权重
                    w_output = output * ia_list
                    output = sigmoid(output).cpu()
                    w_output = sigmoid(w_output).cpu()
                    
                    _labels.append(batch_labels)
                    _preds.append(output)
                    weight_preds.append(w_output)
            
            # 合并所有batch的结果
            all_labels = torch.cat(_labels, dim=0)
            all_preds = torch.cat(_preds, dim=0)
            all_weight_preds = torch.cat(weight_preds, dim=0)
            
            # 计算评估指标
            acc = calACC(all_preds, all_labels)
            b_f1, b_p, b_r, ma_f1, mi_f1 = calF(all_preds, all_labels)
            wb_f1, wb_p, wb_r, wma_f1, wmi_f1 = calF(all_weight_preds, all_labels)
            f, p, r, aupr = evaluate_annotations(all_labels, all_preds)
            
            print(
            '{}  Epoch: {}, Test w-macro-F1: {:.2f}%, Test F1:{:.2f}%, Test avg-F1:{:.2f}%, Test weight-F1:{:.2f}%, Test AUPR:{:.2f}%'.
            format(key, epoch + 1, 100 * wma_f1, 100 * b_f1, 100 * f, 100 * wb_f1, 100 * aupr))

            # 保存每一轮指标
            metrics_output_test[key]['acc'].append(acc)
            metrics_output_test[key]['ma_f1'].append(ma_f1)
            metrics_output_test[key]['mi_f1'].append(mi_f1)
            metrics_output_test[key]['b_f1'].append(b_f1)
            metrics_output_test[key]['b_p'].append(b_p)
            metrics_output_test[key]['b_r'].append(b_r)
            metrics_output_test[key]['w_f1'].append(wb_f1)
            metrics_output_test[key]['p'].append(p)
            metrics_output_test[key]['r'].append(r)
            metrics_output_test[key]['f1'].append(f)
            metrics_output_test[key]['aupr'].append(aupr)

            # 保存最佳模型
            if f > best_f1:
                best_f1 = f
                best_model_weights = model_mlp.state_dict().copy()
                optimizer_model_weights = optimizer.state_dict().copy()
                
                # 保存检查点
                ckpt_dir = './ckpt/cafa5/linear'
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"{ctime}Mlp_esm2_t30_150M_UR50D_{key}_best.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_weights,
                    'optimizer_state_dict': optimizer_model_weights,
                    'best_f1': best_f1,
                    'metrics': metrics_output_test[key]
                }, ckpt_path)
                print(f"Best model saved with F1: {best_f1:.4f}")
            
            # Early stopping检查
            early_stopping(f)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # 加载最佳模型进行最终测试
        if best_model_weights is not None:
            model_mlp.load_state_dict(best_model_weights)
            print(f"Loaded best model for {key} with F1: {best_f1:.4f}")

    # =====================
    # 结果写入文件
    # =====================
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"MLP_esm2_t30_150M_UR50D_{ctime}.txt")
    with open(output_file, 'w') as file_prec:
        file_prec.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file_prec.write(f"Seed: {seed}\n")
        file_prec.write(f"Model: ESM2_t30_150M_UR50D\n")
        file_prec.write(f"Embedding dim: {embed_dim}, Pooling: mean\n\n")
        
        for key in metrics_output_test.keys():
            file_prec.write(f"\n{'='*30} {key} {'='*30}\n")
            epochs_run = len(metrics_output_test[key]['f1'])
            for i in range(epochs_run):
                file_prec.write("Epoch={}; Val Accuracy={:.4f}; Val Precision={:.4f}; Val Recall={:.4f}; Val F1={:.4f}; Val macro-F1={:.4f}; Val micro-F1={:.4f}; Val weight-F1={:.4f}; Val avg-precision={:.4f}; Val avg-recall={:.4f}; Val avg-F1={:.4f}; Val AUPR={:.4f}\n".
                format(i+1, 
                      metrics_output_test[key]['acc'][i], 
                      metrics_output_test[key]['b_p'][i], 
                      metrics_output_test[key]['b_r'][i], 
                      metrics_output_test[key]['b_f1'][i], 
                      metrics_output_test[key]['ma_f1'][i], 
                      metrics_output_test[key]['mi_f1'][i], 
                      metrics_output_test[key]['w_f1'][i], 
                      metrics_output_test[key]['p'][i], 
                      metrics_output_test[key]['r'][i], 
                      metrics_output_test[key]['f1'][i], 
                      metrics_output_test[key]['aupr'][i]))

    print(f"\nTraining completed! Results saved to: {output_file}")
    print('End running date:{}'.format(datetime.now().strftime("%Y%m%d%H%M%S")))