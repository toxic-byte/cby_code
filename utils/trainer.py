import torch
from tqdm import tqdm
from utils.util import calACC, calF, evaluate_annotations
import os
from datetime import datetime
# from models.Text_distill import create_model_and_optimizer,train_one_epoch
from models.Text_contrastive import create_model_and_optimizer,train_one_epoch


def validate(model, test_dataloader, ia_list, epoch, epoch_num, key):
    """验证模型"""
    model.eval()
    _labels = []
    _preds = []
    weight_preds = []
    sigmoid = torch.nn.Sigmoid()
    
    with torch.no_grad():
        for batch_idx, batch_data in tqdm(enumerate(test_dataloader),
                                        desc=f"Epoch {epoch+1}/{epoch_num} Validation",
                                        total=len(test_dataloader)):
            batch_embeddings = batch_data['embedding'].cuda()
            batch_labels = batch_data['labels']
            batch_nlp = batch_data['nlp_embedding'].cuda()
            
            output = model(batch_embeddings, mode='test')
            
            w_output = output * ia_list
            output = sigmoid(output).cpu()
            w_output = sigmoid(w_output).cpu()
            
            _labels.append(batch_labels)
            _preds.append(output)
            weight_preds.append(w_output)
    
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
    
    metrics = {
        'acc': acc,
        'ma_f1': ma_f1,
        'mi_f1': mi_f1,
        'b_f1': b_f1,
        'b_p': b_p,
        'b_r': b_r,
        'w_f1': wb_f1,
        'p': p,
        'r': r,
        'f1': f,
        'aupr': aupr
    }
    
    return metrics

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

def train_model_for_ontology(config, key, train_dataloader, test_dataloader, label_num, ia_list, ctime, metrics_output_test):
    """为特定本体训练模型"""
    model, criterion, optimizer, scheduler = create_model_and_optimizer(config, label_num)
    early_stopping = EarlyStopping(patience=config['patience'], verbose=True)
    
    if key not in metrics_output_test:
        metrics_output_test[key] = {
            'acc': [], 'ma_f1': [], 'mi_f1': [], 'b_p': [], 'b_r': [],
            'b_f1': [], 'w_f1': [], 'p': [], 'r': [], 'f1': [], 'aupr': []
        }
    
    best_f1 = 0
    best_model_weights = None
    optimizer_model_weights = None
    
    for epoch in range(config['epoch_num']):
        # 训练
        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, epoch, config['epoch_num'], key)
        scheduler.step()
        
        # 验证
        metrics = validate(model, test_dataloader, ia_list, epoch, config['epoch_num'], key)
        
        # 保存指标
        for metric_name, metric_value in metrics.items():
            metrics_output_test[key][metric_name].append(metric_value)
        
        # 保存最佳模型
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model_weights = model.state_dict().copy()
            optimizer_model_weights = optimizer.state_dict().copy()
            
            ckpt_dir = './ckpt/cafa5/linear'
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"{ctime}main_text_{config['text_mode']}_distill{key}_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_weights,
                'optimizer_state_dict': optimizer_model_weights,
                'best_f1': best_f1,
                'metrics': metrics_output_test[key]
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


# =====================
# 结果保存
# =====================
def save_results(config, metrics_output_test, seed, ctime):
    """保存训练结果"""
    os.makedirs(config['output_path'], exist_ok=True)
    output_file = os.path.join(config['output_path'], f"main_text_{config['text_mode']}_distill{ctime}.txt")
    
    with open(output_file, 'w') as file_prec:
        file_prec.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file_prec.write(f"Seed: {seed}\n")
        file_prec.write(f"Model: main_text_{config['text_mode']}_distill\n")
        file_prec.write(f"Embedding dim: {config['embed_dim']}, Pooling: mean\n")
                
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
    
    print(f"\nTraining completed! Results saved to: {output_file}")