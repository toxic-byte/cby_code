# =====================
# 每次运行注意修改数据集和esm向量保存的路径
# =====================
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["WANDB_DISABLED"] = "true"

import torch
import numpy as np
import math
import random
import pickle

# 设置随机种子,保证结果可复现
seed = 7
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

from torch import nn
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
from datetime import datetime
from collections import Counter

from sklearn import preprocessing
import sys
sys.path.append(r"utils")
sys.path.append(r"models")
from dataset import obo_graph,preprocess_dataset,parent,IndexedStabilitylandscapeDataset
from util import calACC,calF,evaluate_annotations
from Text_name import Text_Name
from trainer import EarlyStopping
from embed import precompute_esm_embeddings,nlp_embedding
from transformers import AutoTokenizer, AutoModel


nlp_path = '/e/cuiby/huggingface/hub/models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract/snapshots/d673b8835373c6fa116d6d8006b33d48734e305d'
nlp_tokenizer = AutoTokenizer.from_pretrained(nlp_path)
nlp_model = AutoModel.from_pretrained(nlp_path)
nlp_dim = 768
nlp_model.cuda()
nlp_model.eval()

#esm模型维度参数
embed_dim = 640

# train_path = "data/sampled/cafa5_train_10percent.txt"
train_path = "data/sampled/cafa5_train_100.txt"
test_path = "data/sampled/cafa5_test_100.txt"
# test_path = "data/sampled/cafa5_test_10percent.txt"
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

if __name__ == "__main__":
    ctime = datetime.now().strftime("%Y%m%d%H%M%S")
    print('Start running date:{}'.format(ctime))
    
    # 1. 加载训练/测试集
    train_id, training_sequences, training_labels = preprocess_dataset(train_path,MAXLEN,onto,label_space)
    test_id, test_sequences, test_labels = preprocess_dataset(test_path,MAXLEN,onto,label_space)
    print("Train IDs (first 5):", train_id[:5])
    print("Test IDs (first 5):", test_id[:5])
    print(f"Total train samples: {len(train_id)}, Total test samples: {len(test_id)}")

    # 2. 预计算ESM embeddings
    # train_esm_cache = os.path.join(cache_dir, "train_esm_embeddings_sample.pkl")
    # test_esm_cache = os.path.join(cache_dir, "test_esm_embeddings_sample.pkl")

    train_esm_cache = os.path.join(cache_dir, "train_esm_embeddings_mean.pkl")
    test_esm_cache = os.path.join(cache_dir, "test_esm_embeddings_mean.pkl")
    
    train_esm_embeddings = precompute_esm_embeddings(training_sequences, train_esm_cache, pooling='mean')
    test_esm_embeddings = precompute_esm_embeddings(test_sequences, test_esm_cache, pooling='mean')

    pred_results = {}
    metrics_output_test = {}
    
    # 4. 按 GO 三大类分别进行训练
    for key in label_space.keys():
        print(f"\n{'='*50}")
        print(f"Training for ontology: {key}")
        print(f"{'='*50}")
        
        # 定义缓存路径
        label_processing_cache = os.path.join(cache_dir, f"label_processed_{key}.pkl")
        
        if os.path.exists(label_processing_cache):
            print(f"Loading preprocessed labels for {key} from cache...")
            with open(label_processing_cache, 'rb') as f:
                cached = pickle.load(f)
            
            label_list = cached['label_list']
            training_labels_binary = cached['training_labels_binary']
            test_labels_binary = cached['test_labels_binary']
            enc = cached['encoder']
            ia_list = cached['ia_list']
            onto_parent = cached['onto_parent']
            label_num = cached['label_num']
            
            print(f"✓ Loaded cached labels for {key}, label_num={label_num}")
        else:
            print(f"Processing labels for {key} from scratch...")
            
            # 4.1 筛选高频标签
            label_tops = Counter(label_space[key])
            top_labels = sorted([label for label in set(label_space[key]) if label_tops[label] > 21])
            print(f'Top label numbers: {len(top_labels)}')
            label_list = top_labels
            print("Top labels (first 10):", label_list[:10])
            
            # 4.3 标签编码
            labspace = enc.fit_transform(label_list)
            onto_parent = parent(enc, key, label_list, onto, label_space)
            label_num = len(enc.classes_)
            print(f'Number of classes: {label_num}')
            
            # 创建label_list的set以加速查找
            label_set = set(label_list)
            
            # 优化后的标签转换(使用numpy加速)
            print("Converting training labels...")
            training_labels_binary = []
            for label in tqdm(training_labels[key], desc="Processing train labels"):
                filtered_label = [item for item in label if item in label_set]
                if len(filtered_label) == 0:
                    training_labels_binary.append([0] * label_num)
                else:
                    temp_labels = enc.transform(filtered_label)
                    binary_label = [0] * label_num
                    for idx in temp_labels:
                        binary_label[idx] = 1
                    training_labels_binary.append(binary_label)
            
            print("Converting test labels...")
            test_labels_binary = []
            for label in tqdm(test_labels[key], desc="Processing test labels"):
                filtered_label = [item for item in label if item in label_set]
                if len(filtered_label) == 0:
                    test_labels_binary.append([0] * label_num)
                else:
                    temp_labels = enc.transform(filtered_label)
                    binary_label = [0] * label_num
                    for idx in temp_labels:
                        binary_label[idx] = 1
                    test_labels_binary.append(binary_label)
            
            # 4.4 IA 权重矩阵
            print("Building IA weight matrix...")
            ia_list = torch.ones(1, label_num).cuda()
            for _tag, _value in ia_dict.items():
                _tag = _tag[3:]
                if _tag not in label_set:
                    continue
                ia_id = enc.transform([_tag])
                if _value == 0.0:
                    _value = 1.0
                ia_list[0, ia_id[0]] = _value
            
            # 保存处理结果
            print(f"Saving processed labels to {label_processing_cache}")
            with open(label_processing_cache, 'wb') as f:
                pickle.dump({
                    'label_list': label_list,
                    'training_labels_binary': training_labels_binary,
                    'test_labels_binary': test_labels_binary,
                    'encoder': enc,
                    'ia_list': ia_list,
                    'onto_parent': onto_parent,
                    'label_num': label_num,
                }, f)
            print("✓ Saved processed labels")

        # 4.2 获得 NLP 嵌入(使用缓存和平均池化)
        # 只为训练集生成真实的NLP embeddings
        train_nlp_cache = os.path.join(cache_dir, f"train_nlp_embeddings_{key}_name.pkl")
        
        print(f"\n--- Processing Train NLP Embeddings for {key} ---")
        train_nlp = nlp_embedding(
            nlp_model, 
            train_id,
            training_labels[key], 
            key, 
            label_list,
            cache_path=train_nlp_cache,
            onto=onto,
            pooling='mean',  # 使用平均池化,
            name_flag="name"  # 使用GO名称作为文本
        )
        
        # 测试集使用零向量作为占位符
        print(f"\n--- Creating placeholder NLP Embeddings for Test Set ---")
        test_nlp = torch.zeros(len(test_id), nlp_dim)
        print(f"Test NLP embeddings shape: {test_nlp.shape}")
        
        # 验证对齐性
        print(f"\nAlignment verification:")
        print(f"Train samples: ESM={len(train_esm_embeddings)}, NLP={len(train_nlp)}, IDs={len(train_id)}")
        print(f"Test samples: ESM={len(test_esm_embeddings)}, NLP={len(test_nlp)}, IDs={len(test_id)}")
        assert len(train_esm_embeddings) == len(train_nlp) == len(train_id), "Train data alignment error!"
        assert len(test_esm_embeddings) == len(test_nlp) == len(test_id), f"Test data alignment error!,{len(test_esm_embeddings)},{len(test_nlp)},{len(test_id)}"

        # 4.5 创建数据集
        training_dataset = IndexedStabilitylandscapeDataset(
            training_sequences, 
            training_labels_binary, 
            embeddings=train_esm_embeddings,
            nlp_embeddings=train_nlp  # 训练集使用真实的GO文本embedding
        )
        test_dataset = IndexedStabilitylandscapeDataset(
            test_sequences, 
            test_labels_binary, 
            embeddings=test_esm_embeddings,
            nlp_embeddings=test_nlp  # 测试集使用占位符(零向量或平均向量)
        )

        # 4.6 创建DataLoader
        train_dataloader = DataLoader(
            training_dataset, 
            batch_size=256, 
            shuffle=True
        )
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=512, 
            shuffle=False
        )

        # 4.7 构建模型
        model_text_name = Text_Name(embed_dim, label_num).cuda()

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model_text_name.parameters(), lr=4e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.6)
        epoch_num = 200
        
        
        early_stopping = EarlyStopping(patience=10, verbose=True)

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

        for epoch in range(epoch_num):
            # (1) 训练阶段
            model_text_name.train()
            loss_mean = 0
            
            for batch_idx, batch_data in tqdm(enumerate(train_dataloader), 
                                            desc=f"Epoch {epoch+1}/{epoch_num} Training",
                                            total=len(train_dataloader)):
                optimizer.zero_grad()
                
                # 获取batch数据
                batch_embeddings = batch_data['embedding'].cuda()  # [batch_size, embed_dim]
                batch_labels = batch_data['labels'].cuda()  # [batch_size, label_num]
                batch_nlp = batch_data['nlp_embedding'].cuda()  # [batch_size, nlp_dim] - 真实GO文本
                
                # 前向传播
                output = model_text_name(batch_embeddings, batch_nlp)
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
            model_text_name.eval()
            _labels = []
            _preds = []
            weight_preds = []
            
            with torch.no_grad():
                for batch_idx, batch_data in tqdm(enumerate(test_dataloader),
                                                desc=f"Epoch {epoch+1}/{epoch_num} Validation",
                                                total=len(test_dataloader)):
                    batch_embeddings = batch_data['embedding'].cuda()
                    batch_labels = batch_data['labels']
                    batch_nlp = batch_data['nlp_embedding'].cuda()  # [batch_size, nlp_dim] - 占位符
                    
                    output = model_text_name(batch_embeddings, batch_nlp)
                    
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
                best_model_weights = model_text_name.state_dict().copy()
                optimizer_model_weights = optimizer.state_dict().copy()
                
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
            
            early_stopping(f)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        if best_model_weights is not None:
            model_text_name.load_state_dict(best_model_weights)
            print(f"Loaded best model for {key} with F1: {best_f1:.4f}")

    # 保存结果
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"MLP_esm2_t30_150M_UR50D_{ctime}.txt")
    with open(output_file, 'w') as file_prec:
        file_prec.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file_prec.write(f"Seed: {seed}\n")
        file_prec.write(f"Model: ESM2_t30_150M_UR50D with NLP auxiliary\n")
        file_prec.write(f"Embedding dim: {embed_dim}, Pooling: mean\n")
        file_prec.write(f"NLP dim: {nlp_dim}, Test NLP: zeros\n\n")
        
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