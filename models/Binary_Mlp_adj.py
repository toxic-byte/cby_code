import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GraphConvolution(nn.Module):
    """图卷积层"""
    def __init__(self, input_dim, output_dim, bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        参数:
            x: [batch_size, num_go_terms, input_dim] 或 [num_go_terms, input_dim]
            adj: [num_go_terms, num_go_terms] 稀疏邻接矩阵
        """
        if x.dim() == 3:  # [batch_size, num_go_terms, input_dim]
            batch_size = x.size(0)
            # 对每个batch应用图卷积
            support = torch.matmul(x, self.weight)  # [batch_size, num_go_terms, output_dim]
            # 转置为 [num_go_terms, batch_size, output_dim]
            support_t = support.transpose(0, 1)
            # 应用邻接矩阵
            output = torch.sparse.mm(adj, support_t.reshape(support_t.size(0), -1))
            # 重塑回 [num_go_terms, batch_size, output_dim]
            output = output.reshape(support_t.size(0), batch_size, -1)
            # 转置回 [batch_size, num_go_terms, output_dim]
            output = output.transpose(0, 1)
        else:  # [num_go_terms, input_dim]
            support = torch.matmul(x, self.weight)
            output = torch.sparse.mm(adj, support)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class CustomModelWithGCN(nn.Module):
    def __init__(self, esm_dim, nlp_dim, adj, hidden_dim=512, 
                 gcn_hidden_dim=256, dropout=0.3, num_gcn_layers=2):
        """
        带图卷积的二分类模型
        
        参数:
            esm_dim: ESM embedding维度
            nlp_dim: NLP embedding维度
            adj: GO的邻接矩阵 [num_go_terms, num_go_terms]
            hidden_dim: 主网络隐藏层维度
            gcn_hidden_dim: GCN隐藏层维度
            dropout: dropout概率
            num_gcn_layers: GCN层数
        """
        super(CustomModelWithGCN, self).__init__()
        
        self.num_gcn_layers = num_gcn_layers
        
        # 注册邻接矩阵（不需要梯度）
        self.register_buffer('adj', adj)
        
        # ========== 二分类编码器 ==========
        self.esm_proj = nn.Linear(esm_dim, hidden_dim)
        self.nlp_proj = nn.Linear(nlp_dim, hidden_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 分类头 - 输出单个logit
        self.classifier = nn.Linear(hidden_dim // 2, 1)
        
        # ========== 图卷积层 ==========
        # 用于refine每个蛋白质的所有GO预测
        self.gcn_layers = nn.ModuleList()
        
        # 第一层GCN
        self.gcn_layers.append(GraphConvolution(1, gcn_hidden_dim))
        
        # 中间层GCN
        for _ in range(num_gcn_layers - 2):
            self.gcn_layers.append(GraphConvolution(gcn_hidden_dim, gcn_hidden_dim))
        
        # 最后一层GCN - 输出回1维
        if num_gcn_layers > 1:
            self.gcn_layers.append(GraphConvolution(gcn_hidden_dim, 1))
        
        self.gcn_dropout = nn.Dropout(dropout)
        
        # 归一化层
        self.esm_norm = nn.LayerNorm(hidden_dim)
        self.nlp_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, esm_embedding, nlp_embedding, use_gcn=True):
        """
        前向传播
        
        参数:
            esm_embedding: [batch_size, esm_dim]
            nlp_embedding: [batch_size, nlp_dim] 
                          (batch_size = batch_size * num_go_terms when flattened)
            use_gcn: 是否使用图卷积（训练时True，推理时可选）
        
        返回:
            logits: [batch_size, 1] 或 [batch_size, num_go_terms] (如果reshape)
        """
        # 确保维度正确
        if esm_embedding.dim() == 1:
            esm_embedding = esm_embedding.unsqueeze(0)
        if nlp_embedding.dim() == 1:
            nlp_embedding = nlp_embedding.unsqueeze(0)
        
        # 投影和归一化
        esm_feat = self.esm_norm(self.esm_proj(esm_embedding))
        nlp_feat = self.nlp_norm(self.nlp_proj(nlp_embedding))
        
        # 拼接和融合
        combined = torch.cat([esm_feat, nlp_feat], dim=-1)
        fused = self.fusion(combined)
        
        # 分类
        logits = self.classifier(fused)  # [batch_size * num_go_terms, 1]
        
        return logits
    
    def forward_with_gcn(self, esm_embedding, list_embedding):
        """
        完整的前向传播（包含GCN）
        
        参数:
            esm_embedding: [batch_size, esm_dim] 单个蛋白质的embedding
            list_embedding: [num_go_terms, nlp_dim] 所有GO的embedding
        
        返回:
            refined_logits: [batch_size, num_go_terms] 经过GCN refine的logits
        """
        batch_size = esm_embedding.size(0)
        num_go_terms = list_embedding.size(0)
        
        # 1. 获取所有GO的初始预测
        # 扩展维度
        esm_expanded = esm_embedding.unsqueeze(1).expand(-1, num_go_terms, -1)
        nlp_expanded = list_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Flatten
        esm_flat = esm_expanded.reshape(-1, esm_expanded.size(-1))
        nlp_flat = nlp_expanded.reshape(-1, nlp_expanded.size(-1))
        
        # 获取初始logits
        initial_logits = self.forward(esm_flat, nlp_flat, use_gcn=False)
        
        # Reshape为 [batch_size, num_go_terms, 1]
        logits_reshaped = initial_logits.reshape(batch_size, num_go_terms, 1)
        
        # 2. 对每个样本应用GCN
        # 将logits看作GO节点的特征，通过图卷积传播
        x = logits_reshaped  # [batch_size, num_go_terms, 1]
        
        for i, gcn in enumerate(self.gcn_layers):
            x = gcn(x, self.adj)  # [batch_size, num_go_terms, hidden_dim]
            
            if i < len(self.gcn_layers) - 1:  # 不是最后一层
                x = F.relu(x)
                x = self.gcn_dropout(x)
        
        # 最后一层输出 [batch_size, num_go_terms, 1]
        refined_logits = x.squeeze(-1)  # [batch_size, num_go_terms]
        
        return refined_logits, initial_logits.reshape(batch_size, num_go_terms)