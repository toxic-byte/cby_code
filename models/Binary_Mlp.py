import torch
import torch.nn as nn
import math
from tqdm import tqdm

# Classifier
class CustomModel(nn.Module):
    def __init__(self, esm_dim, nlp_dim, hidden_dim=512, dropout=0.3):
        """
        二分类模型：判断蛋白质序列是否具有特定GO功能
        
        参数:
            esm_dim: ESM embedding维度
            nlp_dim: NLP embedding维度
            hidden_dim: 隐藏层维度
            dropout: dropout概率
        """
        super(CustomModel, self).__init__()
        
        # 投影层，将ESM和NLP embedding投影到相同维度
        self.esm_proj = nn.Linear(esm_dim, hidden_dim)
        self.nlp_proj = nn.Linear(nlp_dim, hidden_dim)
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 分类头
        self.classifier = nn.Linear(hidden_dim // 2, 1)
        
        # 归一化层
        self.esm_norm = nn.LayerNorm(hidden_dim)
        self.nlp_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, esm_embedding, nlp_embedding):
        """
        前向传播
        
        参数:
            esm_embedding: [batch_size, esm_dim] 或 [esm_dim]
            nlp_embedding: [batch_size, nlp_dim] 或 [nlp_dim]
        
        返回:
            logits: [batch_size, 1] 二分类logits
        """
        # 确保维度正确
        if esm_embedding.dim() == 1:
            esm_embedding = esm_embedding.unsqueeze(0)
        if nlp_embedding.dim() == 1:
            nlp_embedding = nlp_embedding.unsqueeze(0)
        
        # 投影和归一化
        esm_feat = self.esm_norm(self.esm_proj(esm_embedding))  # [batch_size, hidden_dim]
        nlp_feat = self.nlp_norm(self.nlp_proj(nlp_embedding))  # [batch_size, hidden_dim]
        
        # 拼接特征
        combined = torch.cat([esm_feat, nlp_feat], dim=-1)  # [batch_size, hidden_dim * 2]
        
        # 特征融合
        fused = self.fusion(combined)  # [batch_size, hidden_dim // 2]
        
        # 分类
        logits = self.classifier(fused)  # [batch_size, 1]
        
        return logits