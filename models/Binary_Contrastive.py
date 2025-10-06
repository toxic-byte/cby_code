import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ContrastiveCustomModel(nn.Module):
    def __init__(self, esm_dim, nlp_dim, hidden_dim=512, projection_dim=256, dropout=0.3, temperature=0.07):
        """
        结合对比学习的二分类模型
        
        参数:
            esm_dim: ESM embedding维度
            nlp_dim: NLP embedding维度
            hidden_dim: 隐藏层维度
            projection_dim: 对比学习投影维度
            dropout: dropout概率
            temperature: 对比学习温度参数
        """
        super(ContrastiveCustomModel, self).__init__()
        
        self.temperature = temperature
        
        # ========== 特征编码器 ==========
        # ESM编码器
        self.esm_encoder = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # GO编码器
        self.nlp_encoder = nn.Sequential(
            nn.Linear(nlp_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # ========== 对比学习投影头 ==========
        self.esm_projection = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        self.nlp_projection = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # ========== 分类头 ==========
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def encode(self, esm_embedding, nlp_embedding):
        """编码特征"""
        esm_feat = self.esm_encoder(esm_embedding)  # [batch, hidden_dim]
        nlp_feat = self.nlp_encoder(nlp_embedding)  # [batch, hidden_dim]
        return esm_feat, nlp_feat
    
    def project(self, esm_feat, nlp_feat):
        """投影到对比学习空间"""
        esm_proj = F.normalize(self.esm_projection(esm_feat), dim=-1)  # [batch, projection_dim]
        nlp_proj = F.normalize(self.nlp_projection(nlp_feat), dim=-1)  # [batch, projection_dim]
        return esm_proj, nlp_proj
    
    def forward(self, esm_embedding, nlp_embedding, return_features=False):
        """
        前向传播
        
        参数:
            esm_embedding: [batch_size, esm_dim]
            nlp_embedding: [batch_size, nlp_dim]
            return_features: 是否返回中间特征（用于对比学习）
        
        返回:
            logits: [batch_size, 1]
            如果return_features=True，还返回 (esm_proj, nlp_proj)
        """
        # 编码
        esm_feat, nlp_feat = self.encode(esm_embedding, nlp_embedding)
        
        # 分类
        combined = torch.cat([esm_feat, nlp_feat], dim=-1)
        logits = self.classifier(combined)
        
        if return_features:
            # 投影（用于对比学习）
            esm_proj, nlp_proj = self.project(esm_feat, nlp_feat)
            return logits, esm_proj, nlp_proj
        
        return logits


class ContrastiveLoss(nn.Module):
    """对比学习损失（InfoNCE）"""
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, esm_proj, nlp_proj, labels):
        """
        计算对比学习损失
        
        参数:
            esm_proj: [batch_size, projection_dim] 蛋白质投影
            nlp_proj: [batch_size, projection_dim] GO投影
            labels: [batch_size] 二分类标签 (0或1)
        """
        batch_size = esm_proj.size(0)
        
        # 计算相似度矩阵
        similarity = torch.matmul(esm_proj, nlp_proj.T) / self.temperature  # [batch, batch]
        
        # 创建正样本mask（label=1的是正样本）
        positive_mask = labels.float()  # [batch]
        
        # 对角线是对应的配对
        diagonal_similarity = torch.diag(similarity)  # [batch]
        
        # 对于正样本，拉近距离
        positive_loss = -diagonal_similarity * positive_mask
        
        # 对于负样本，推远距离（通过softmax的分母实现）
        # exp(s_ii) / sum(exp(s_ij))
        exp_sim = torch.exp(similarity)
        log_prob = diagonal_similarity - torch.log(exp_sim.sum(dim=1) + 1e-8)
        
        # 只对正样本计算对比损失
        contrastive_loss = -(log_prob * positive_mask).sum() / (positive_mask.sum() + 1e-8)
        
        return contrastive_loss


class SupConLoss(nn.Module):
    """监督对比学习损失（SupCon）"""
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        """
        参数:
            features: [batch_size, projection_dim] 归一化后的特征
            labels: [batch_size] 标签
        """
        device = features.device
        batch_size = features.size(0)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # 相同标签为1
        
        # 去除对角线（自己和自己）
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # 计算损失
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # 只对正样本对计算
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        
        loss = -mean_log_prob_pos.mean()
        
        return loss


class HybridLoss(nn.Module):
    """混合损失：分类损失 + 对比学习损失"""
    def __init__(self, alpha=0.5, temperature=0.07, contrastive_type='infonce'):
        """
        参数:
            alpha: 对比学习损失的权重
            temperature: 温度参数
            contrastive_type: 'infonce' 或 'supcon'
        """
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        if contrastive_type == 'infonce':
            self.contrastive_loss = ContrastiveLoss(temperature)
        else:
            self.contrastive_loss = SupConLoss(temperature)
        
        self.contrastive_type = contrastive_type
        
    def forward(self, logits, labels, esm_proj=None, nlp_proj=None):
        """
        参数:
            logits: [batch_size, 1] 分类logits
            labels: [batch_size] 标签
            esm_proj: [batch_size, projection_dim] 蛋白质投影
            nlp_proj: [batch_size, projection_dim] GO投影
        """
        # 分类损失
        cls_loss = self.bce_loss(logits.squeeze(-1), labels.float())
        
        # 对比学习损失
        if esm_proj is not None and nlp_proj is not None:
            if self.contrastive_type == 'infonce':
                con_loss = self.contrastive_loss(esm_proj, nlp_proj, labels)
            else:  # supcon
                # 将蛋白质和GO特征都纳入对比学习
                all_features = torch.cat([esm_proj, nlp_proj], dim=0)
                all_labels = torch.cat([labels, labels], dim=0)
                con_loss = self.contrastive_loss(all_features, all_labels)
            
            total_loss = cls_loss + self.alpha * con_loss
            return total_loss, cls_loss, con_loss
        else:
            return cls_loss, cls_loss, torch.tensor(0.0).to(logits.device)