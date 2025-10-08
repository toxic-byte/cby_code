import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class ProteinFunctionContrastiveDataset(Dataset):
    """对比学习数据集"""
    def __init__(self, esm_embeddings, labels_binary, text_embeddings):
        """
        Args:
            esm_embeddings: [num_proteins, esm_dim] 蛋白质序列嵌入
            labels_binary: [num_proteins, num_go] 蛋白质功能的0/1标签
            text_embeddings: [num_go, nlp_dim] 功能文本描述嵌入
        """
        self.esm_embeddings = esm_embeddings
        self.labels_binary = labels_binary
        self.text_embeddings = text_embeddings
        
    def __len__(self):
        return len(self.esm_embeddings)
    
    def __getitem__(self, idx):
        protein_emb = self.esm_embeddings[idx]
        protein_labels = self.labels_binary[idx]
        
        # 找到该蛋白质的正样本功能(标签为1的功能)
        positive_indices = torch.where(protein_labels == 1)[0]
        
        return {
            'protein_emb': protein_emb,
            'positive_indices': positive_indices,
            'protein_idx': idx
        }


class ProjectionHead(nn.Module):
    """投影头，将不同模态映射到统一空间"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class ProteinFunctionContrastiveModel(nn.Module):
    """蛋白质-功能对比学习模型"""
    def __init__(self, esm_dim, nlp_dim, projection_dim=256, hidden_dim=512):
        super().__init__()
        
        # 蛋白质序列投影头
        self.protein_projector = ProjectionHead(esm_dim, hidden_dim, projection_dim)
        
        # 功能文本投影头
        self.function_projector = ProjectionHead(nlp_dim, hidden_dim, projection_dim)
        
        # 可学习的温度参数
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self, protein_emb, function_emb):
        """
        Args:
            protein_emb: [batch, esm_dim]
            function_emb: [num_functions, nlp_dim]
        Returns:
            protein_features: [batch, projection_dim]
            function_features: [num_functions, projection_dim]
        """
        protein_features = self.protein_projector(protein_emb)
        function_features = self.function_projector(function_emb)
        
        return protein_features, function_features


class InfoNCELoss(nn.Module):
    """InfoNCE对比学习损失"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, protein_features, function_features, positive_mask, temperature):
        """
        Args:
            protein_features: [batch, dim] 归一化的蛋白质特征
            function_features: [num_go, dim] 归一化的功能特征
            positive_mask: [batch, num_go] 正样本mask (1表示正样本)
            temperature: 温度参数
        """
        # 计算相似度矩阵 [batch, num_go]
        logits = torch.matmul(protein_features, function_features.t()) / temperature.exp()
        
        # 对于每个蛋白质,计算其与所有功能的对比损失
        batch_size = logits.shape[0]
        
        # 使用BCE with logits的多标签版本
        # 将正样本的logits最大化,负样本的logits最小化
        
        # 方法1: 多标签对比学习损失
        # 对每个蛋白质,正样本的相似度应该高,负样本应该低
        positive_mask = positive_mask.float()
        
        # 计算log-sum-exp来数值稳定
        exp_logits = torch.exp(logits)
        
        # 对于每个蛋白质,计算InfoNCE损失
        loss = 0
        for i in range(batch_size):
            # 正样本的logits
            pos_mask = positive_mask[i]  # [num_go]
            if pos_mask.sum() == 0:
                continue
                
            # 分子: 正样本的exp(logits)之和
            numerator = (exp_logits[i] * pos_mask).sum()
            
            # 分母: 所有样本的exp(logits)之和
            denominator = exp_logits[i].sum()
            
            # InfoNCE损失
            loss += -torch.log(numerator / denominator + 1e-8)
        
        if self.reduction == 'mean':
            loss = loss / batch_size
            
        return loss


class AsymmetricContrastiveLoss(nn.Module):
    """非对称对比学习损失 - 处理多标签场景"""
    def __init__(self):
        super().__init__()
        
    def forward(self, protein_features, function_features, positive_mask, temperature):
        """
        对每个正样本对单独计算对比损失
        """
        # 计算相似度矩阵
        logits = torch.matmul(protein_features, function_features.t()) / temperature.exp()
        
        batch_size, num_functions = logits.shape
        
        # 对每个正样本对计算损失
        losses = []
        for i in range(batch_size):
            pos_indices = torch.where(positive_mask[i] == 1)[0]
            if len(pos_indices) == 0:
                continue
                
            for pos_idx in pos_indices:
                # 对于每个正样本,计算它相对于所有功能的对比损失
                pos_logit = logits[i, pos_idx]
                
                # 所有logits的logsumexp
                all_logits = logits[i]  # [num_functions]
                logsumexp_all = torch.logsumexp(all_logits, dim=0)
                
                # 对比损失: -log(exp(pos) / sum(exp(all)))
                loss = -pos_logit + logsumexp_all
                losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=logits.device)
            
        return torch.stack(losses).mean()


def collate_fn(batch):
    """自定义collate函数处理不定长的positive_indices"""
    protein_embs = torch.stack([item['protein_emb'] for item in batch])
    protein_indices = torch.tensor([item['protein_idx'] for item in batch])
    
    # 收集所有positive_indices
    positive_indices_list = [item['positive_indices'] for item in batch]
    
    return {
        'protein_embs': protein_embs,
        'protein_indices': protein_indices,
        'positive_indices_list': positive_indices_list
    }


def pretrain_contrastive(
    train_esm_embeddings,
    training_labels_binary, 
    list_nlp,
    key,
    config,
    esm_dim=1280,
    nlp_dim=768,
    projection_dim=256,
    hidden_dim=512,
    batch_size=128,
    num_epochs=100,
    learning_rate=1e-4,
    device='cuda'
):
    """
    对比学习预训练主函数
    
    Args:
        train_esm_embeddings: [num_proteins, esm_dim]
        training_labels_binary: [num_proteins, num_go]
        list_nlp: [num_go, nlp_dim]
    """
    
   # ========== 修复：安全的类型转换 ==========
    def safe_to_tensor(data):
        """安全地将数据转换为CPU上的FloatTensor"""
        if isinstance(data, torch.Tensor):
            # 如果已经是tensor，先移到CPU再确保是float类型
            return data.cpu().float()
        elif isinstance(data, np.ndarray):
            # 如果是numpy数组，直接转换
            return torch.from_numpy(data).float()
        else:
            # 其他类型（如list），转换为tensor
            return torch.FloatTensor(data)
    
    # 转换为CPU上的FloatTensor
    # train_esm_embeddings = safe_to_tensor(train_esm_embeddings)
    training_labels_binary = safe_to_tensor(training_labels_binary)
    # list_nlp = safe_to_tensor(list_nlp)
    
    # 创建数据集和数据加载器
    dataset = ProteinFunctionContrastiveDataset(
        train_esm_embeddings,
        training_labels_binary,
        list_nlp
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # 创建模型
    model = ProteinFunctionContrastiveModel(
        esm_dim=esm_dim,
        nlp_dim=nlp_dim,
        projection_dim=projection_dim,
        hidden_dim=hidden_dim
    ).to(device)
    
    # 将功能文本嵌入移到GPU
    function_embeddings = list_nlp.to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    
    # 损失函数 - 可以选择使用InfoNCE或Asymmetric
    criterion = AsymmetricContrastiveLoss()
    # criterion = InfoNCELoss()
    
    # 训练循环
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            protein_embs = batch['protein_embs'].to(device)
            protein_indices = batch['protein_indices'].to(device)
            positive_indices_list = batch['positive_indices_list']
            
            # 构建positive mask
            batch_size = len(protein_embs)
            num_functions = len(function_embeddings)
            positive_mask = torch.zeros(batch_size, num_functions, device=device)
            
            for i, pos_indices in enumerate(positive_indices_list):
                if len(pos_indices) > 0:
                    positive_mask[i, pos_indices] = 1
            
            # 前向传播
            protein_features, function_features = model(protein_embs, function_embeddings)
            
            # 计算损失
            loss = criterion(
                protein_features,
                function_features,
                positive_mask,
                model.temperature
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # 更新学习率
        scheduler.step()
        
        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, "
              f"Temperature: {model.temperature.exp().item():.4f}, "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 保存检查点
        ckpt_dir = f"./ckpt/cafa5/contrastive_{key}_{config['run_mode']}_{config['text_mode']}"
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f'contrastive_checkpoint_epoch_{epoch+1}.pt')
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
    
    # 保存最终模型
    ckpt_path = os.path.join(ckpt_dir, f'contrastive_pretrained_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'protein_projector': model.protein_projector.state_dict(),
        'function_projector': model.function_projector.state_dict(),
    }, ckpt_path)
    
    return model

