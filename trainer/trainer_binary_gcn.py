import torch
import torch.nn as nn   
from tqdm import tqdm
from utils.util import calACC, calF, evaluate_annotations
import os
from datetime import datetime
from models.Binary_Mlp_adj import CustomModelWithGCN

def create_model_and_optimizer(config, adj_matrix=None,pos_weight=None):
    """创建模型、损失函数和优化器"""
    model = CustomModelWithGCN(
        esm_dim=config['embed_dim'],
        nlp_dim=config['nlp_dim'],
        adj=adj_matrix
    ).cuda()
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config['step_size'], 
        gamma=config['gamma']
    )

    # 检查哪些参数会被训练
    print("Trainable parameters:")
    trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")
            trainable_params += param.numel()
    print(f"Total trainable parameters: {trainable_params:,}")

    return model, criterion, optimizer, scheduler

def train_one_epoch_with_gcn(model, train_dataloader, list_embedding, criterion, 
                             optimizer, epoch, epoch_num, key, use_gcn=True):
    """
    使用GCN的训练函数
    """
    model.train()
    loss_mean = 0
    
    list_embedding = list_embedding.cuda()
    num_go_terms = list_embedding.shape[0]
    
    for batch_idx, batch_data in tqdm(enumerate(train_dataloader), 
                                    desc=f"Epoch {epoch+1}/{epoch_num} Training",
                                    total=len(train_dataloader)):
        optimizer.zero_grad()
        
        batch_embeddings = batch_data['embedding'].cuda()  # [batch_size, esm_dim]
        batch_labels = batch_data['labels'].cuda()  # [batch_size, num_go_terms]
        batch_size = batch_embeddings.shape[0]
        
        if use_gcn:
            # 使用GCN
            refined_logits, initial_logits = model.forward_with_gcn(
                batch_embeddings, list_embedding
            )  # [batch_size, num_go_terms]
            
            # 可以选择：
            # 选项1: 只用refined logits计算loss
            loss = criterion(refined_logits, batch_labels)
            
            # 选项2: 同时优化initial和refined（多任务学习）
            # loss_initial = criterion(initial_logits, batch_labels)
            # loss_refined = criterion(refined_logits, batch_labels)
            # loss = 0.3 * loss_initial + 0.7 * loss_refined
            
        else:
            # 不使用GCN（baseline）
            esm_expanded = batch_embeddings.unsqueeze(1).expand(-1, num_go_terms, -1)
            nlp_expanded = list_embedding.unsqueeze(0).expand(batch_size, -1, -1)
            
            esm_flat = esm_expanded.reshape(-1, esm_expanded.size(-1))
            nlp_flat = nlp_expanded.reshape(-1, nlp_expanded.size(-1))
            
            logits_flat = model(esm_flat, nlp_flat)
            outputs = logits_flat.reshape(batch_size, num_go_terms)
            
            loss = criterion(outputs, batch_labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        loss_mean += loss.item()
        
        if (batch_idx + 1) % 50 == 0:
            print(f'{key}  Epoch [{epoch+1}/{epoch_num}], Step [{batch_idx+1}/{len(train_dataloader)}], '
                  f'Loss: {loss_mean/(batch_idx+1):.4f}')
    
    return loss_mean / len(train_dataloader)


def validate_with_gcn(model, test_dataloader, list_embedding, ia_list, epoch, 
                     epoch_num, key, use_gcn=True):
    """验证函数"""
    model.eval()
    _labels = []
    _preds = []
    weight_preds = []
    sigmoid = torch.nn.Sigmoid()
    
    list_embedding = list_embedding.cuda()
    ia_list = ia_list.cuda()
    num_go_terms = list_embedding.shape[0]
    
    with torch.no_grad():
        for batch_data in tqdm(test_dataloader, desc=f"Epoch {epoch+1}/{epoch_num} Validation"):
            batch_embeddings = batch_data['embedding'].cuda()
            batch_labels = batch_data['labels']
            batch_size = batch_embeddings.shape[0]
            
            if use_gcn:
                # 使用GCN
                output, _ = model.forward_with_gcn(batch_embeddings, list_embedding)
            else:
                # 不使用GCN
                esm_expanded = batch_embeddings.unsqueeze(1).expand(-1, num_go_terms, -1)
                nlp_expanded = list_embedding.unsqueeze(0).expand(batch_size, -1, -1)
                
                esm_flat = esm_expanded.reshape(-1, esm_expanded.size(-1))
                nlp_flat = nlp_expanded.reshape(-1, nlp_expanded.size(-1))
                
                logits_flat = model(esm_flat, nlp_flat)
                output = logits_flat.reshape(batch_size, num_go_terms)
            
            w_output = output * ia_list
            output = sigmoid(output).cpu()
            w_output = sigmoid(w_output).cpu()
            
            _labels.append(batch_labels)
            _preds.append(output)
            weight_preds.append(w_output)
    
    all_labels = torch.cat(_labels, dim=0)
    all_preds = torch.cat(_preds, dim=0)
    all_weight_preds = torch.cat(weight_preds, dim=0)
    
    # 计算指标...
    from utils.util import calACC, calF, evaluate_annotations
    acc = calACC(all_preds, all_labels)
    b_f1, b_p, b_r, ma_f1, mi_f1 = calF(all_preds, all_labels)
    wb_f1, wb_p, wb_r, wma_f1, wmi_f1 = calF(all_weight_preds, all_labels)
    f, p, r, aupr = evaluate_annotations(all_labels, all_preds)
    
    print(f'{key}  Epoch: {epoch+1}, Test w-macro-F1: {100*wma_f1:.2f}%, '
          f'Test F1:{100*b_f1:.2f}%, Test avg-F1:{100*f:.2f}%, '
          f'Test weight-F1:{100*wb_f1:.2f}%, Test AUPR:{100*aupr:.2f}%')
    
    return {
        'acc': acc, 'ma_f1': ma_f1, 'mi_f1': mi_f1,
        'b_f1': b_f1, 'b_p': b_p, 'b_r': b_r,
        'w_f1': wb_f1, 'p': p, 'r': r, 'f1': f, 'aupr': aupr
    }

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
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def train_model_for_ontology(config, key, train_dataloader, test_dataloader, 
                            list_embedding, ia_list, ctime, 
                            metrics_output_test, adj_matrix=None,pos_weight=None):
    """为特定本体训练模型"""
    model, criterion, optimizer, scheduler = create_model_and_optimizer(
        config, adj_matrix, pos_weight)
    early_stopping = EarlyStopping(patience=config['patience'], verbose=True)
    
    if key not in metrics_output_test:
        metrics_output_test[key] = {
            'acc': [], 'ma_f1': [], 'mi_f1': [], 'b_p': [], 'b_r': [],
            'b_f1': [], 'w_f1': [], 'p': [], 'r': [], 'f1': [], 'aupr': []
        }
    
    best_f1 = 0
    best_model_weights = None
    optimizer_model_weights = None
    
    train_fn = train_one_epoch_with_gcn
    
    for epoch in range(config['epoch_num']):
        # 训练
        train_loss = train_fn(model, train_dataloader, list_embedding, criterion, 
                             optimizer, epoch, config['epoch_num'], key,use_gcn=True)
        scheduler.step()
        
        # 验证
        metrics = validate_with_gcn(model, test_dataloader, list_embedding, ia_list, epoch, 
                          config['epoch_num'], key)
        
        # 保存指标
        for metric_name, metric_value in metrics.items():
            metrics_output_test[key][metric_name].append(metric_value)
        
        # 保存最佳模型
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model_weights = model.state_dict().copy()
            optimizer_model_weights = optimizer.state_dict().copy()
            
            ckpt_dir = './ckpt/cafa5/binary_mlp'
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"{ctime}binary_mlp_{key}_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_weights,
                'optimizer_state_dict': optimizer_model_weights,
                'best_f1': best_f1,
                'metrics': metrics_output_test[key],
                'config': config
            }, ckpt_path)
            print(f"Best model saved with F1: {best_f1:.4f}")
        
        # 早停
        early_stopping(metrics['f1'])
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f"Loaded best model for {key} with F1: {best_f1:.4f}")
    
    return model


def save_results(config, metrics_output_test, seed, ctime):
    """保存训练结果"""
    os.makedirs(config['output_path'], exist_ok=True)
    output_file = os.path.join(config['output_path'], f"binary_mlp_{config.get('text_mode', 'default')}_{ctime}.txt")
    
    with open(output_file, 'w') as file_prec:
        file_prec.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file_prec.write(f"Seed: {seed}\n")
        file_prec.write(f"Model: CustomModel (Binary Classification)\n")
        file_prec.write(f"Hidden dim: {config.get('hidden_dim', 512)}, Dropout: {config.get('dropout', 0.3)}\n\n")

        for key in metrics_output_test.keys():
            file_prec.write(f"\n{'='*30} {key} {'='*30}\n")
            epochs_run = len(metrics_output_test[key]['f1'])
            for i in range(epochs_run):
                file_prec.write(
                    "Epoch={}; Val Accuracy={:.4f}; Val Precision={:.4f}; Val Recall={:.4f}; "
                    "Val F1={:.4f}; Val macro-F1={:.4f}; Val micro-F1={:.4f}; Val weight-F1={:.4f}; "
                    "Val avg-precision={:.4f}; Val avg-recall={:.4f}; Val avg-F1={:.4f}; Val AUPR={:.4f}\n".format(
                        i+1,
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
                        metrics_output_test[key]['aupr'][i]
                    ))
            
            # 添加最佳结果总结
            best_epoch = metrics_output_test[key]['f1'].index(max(metrics_output_test[key]['f1']))
            file_prec.write(f"\nBest Results at Epoch {best_epoch + 1}:\n")
            file_prec.write(f"  F1: {metrics_output_test[key]['f1'][best_epoch]:.4f}\n")
            file_prec.write(f"  AUPR: {metrics_output_test[key]['aupr'][best_epoch]:.4f}\n")
            file_prec.write(f"  Precision: {metrics_output_test[key]['b_p'][best_epoch]:.4f}\n")
            file_prec.write(f"  Recall: {metrics_output_test[key]['b_r'][best_epoch]:.4f}\n")
    
    print(f"\nTraining completed! Results saved to: {output_file}")