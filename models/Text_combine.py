import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm

# 多头注意力模块
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x

# 交叉注意力模块 - 用于序列和文本的交互
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value):
        B, N_q, C = query.shape
        _, N_kv, _ = key_value.shape
        
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.out_proj(x)
        return x

# 前馈网络
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)

# Transformer编码器层
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), dropout)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# 主分类模型
class CustomModel(nn.Module):
    def __init__(self, seq_dim, nlp_dim, output_size, 
                 hidden_dim=1024, num_heads=8, num_layers=3, 
                 fusion_method='concat', dropout=0.1):
        """
        融合序列嵌入与文本嵌入的蛋白质功能分类模型
        
        参数:
            seq_dim: 序列嵌入维度
            nlp_dim: 文本嵌入维度
            output_size: 输出类别数
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            fusion_method: 融合方法 ('concat', 'add', 'cross_attention', 'gated')
            dropout: Dropout比率
        """
        super(CustomModel, self).__init__()
        
        self.fusion_method = fusion_method
        
        # 序列和文本的投影层，将它们映射到相同维度
        self.seq_proj = nn.Sequential(
            nn.Linear(seq_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.nlp_proj = nn.Sequential(
            nn.Linear(nlp_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 序列特征编码器
        self.seq_encoder = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 文本特征编码器
        self.nlp_encoder = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 不同的融合策略
        if fusion_method == 'concat':
            fusion_dim = hidden_dim * 2
        elif fusion_method == 'add' or fusion_method == 'multiply':
            fusion_dim = hidden_dim
        elif fusion_method == 'cross_attention':
            self.cross_attn_seq2nlp = CrossAttention(hidden_dim, num_heads, dropout)
            self.cross_attn_nlp2seq = CrossAttention(hidden_dim, num_heads, dropout)
            self.cross_norm1 = nn.LayerNorm(hidden_dim)
            self.cross_norm2 = nn.LayerNorm(hidden_dim)
            fusion_dim = hidden_dim * 2
        elif fusion_method == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
            fusion_dim = hidden_dim
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # 融合后的特征编码器
        self.fusion_encoder = nn.ModuleList([
            TransformerBlock(fusion_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 4, output_size)
        )
        
        # 全局平均池化
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x, y):
        """
        前向传播
        
        参数:
            x: 序列嵌入 [batch_size, seq_len, seq_dim] 或 [batch_size, seq_dim]
            y: 文本嵌入 [batch_size, text_len, nlp_dim] 或 [batch_size, nlp_dim]
        
        返回:
            分类logits [batch_size, output_size]
        """
        # 处理输入维度
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, seq_dim]
        if len(y.shape) == 2:
            y = y.unsqueeze(1)  # [batch_size, 1, nlp_dim]
        
        # 投影到统一维度
        seq_features = self.seq_proj(x)  # [batch_size, seq_len, hidden_dim]
        nlp_features = self.nlp_proj(y)  # [batch_size, text_len, hidden_dim]
        
        # 通过各自的编码器
        for layer in self.seq_encoder:
            seq_features = layer(seq_features)
        
        for layer in self.nlp_encoder:
            nlp_features = layer(nlp_features)
        
        # 特征融合
        if self.fusion_method == 'concat':
            # 池化后拼接
            seq_pooled = self.pool(seq_features.transpose(1, 2)).squeeze(-1)
            nlp_pooled = self.pool(nlp_features.transpose(1, 2)).squeeze(-1)
            fused = torch.cat([seq_pooled, nlp_pooled], dim=-1).unsqueeze(1)
            
        elif self.fusion_method == 'add':
            # 池化后相加
            seq_pooled = self.pool(seq_features.transpose(1, 2)).squeeze(-1)
            nlp_pooled = self.pool(nlp_features.transpose(1, 2)).squeeze(-1)
            fused = (seq_pooled + nlp_pooled).unsqueeze(1)
            
        elif self.fusion_method == 'multiply':
            # 池化后逐元素相乘
            seq_pooled = self.pool(seq_features.transpose(1, 2)).squeeze(-1)
            nlp_pooled = self.pool(nlp_features.transpose(1, 2)).squeeze(-1)
            fused = (seq_pooled * nlp_pooled).unsqueeze(1)
            
        elif self.fusion_method == 'cross_attention':
            # 交叉注意力融合
            seq_attended = self.cross_attn_seq2nlp(
                self.cross_norm1(seq_features), 
                self.cross_norm1(nlp_features)
            )
            nlp_attended = self.cross_attn_nlp2seq(
                self.cross_norm2(nlp_features), 
                self.cross_norm2(seq_features)
            )
            
            seq_pooled = self.pool(seq_attended.transpose(1, 2)).squeeze(-1)
            nlp_pooled = self.pool(nlp_attended.transpose(1, 2)).squeeze(-1)
            fused = torch.cat([seq_pooled, nlp_pooled], dim=-1).unsqueeze(1)
            
        elif self.fusion_method == 'gated':
            # 门控融合
            seq_pooled = self.pool(seq_features.transpose(1, 2)).squeeze(-1)
            nlp_pooled = self.pool(nlp_features.transpose(1, 2)).squeeze(-1)
            
            gate_input = torch.cat([seq_pooled, nlp_pooled], dim=-1)
            gate_weight = self.gate(gate_input)
            
            fused = (gate_weight * seq_pooled + (1 - gate_weight) * nlp_pooled).unsqueeze(1)
        
        # 融合特征编码
        for layer in self.fusion_encoder:
            fused = layer(fused)
        
        # 池化并分类
        fused = fused.squeeze(1)  # [batch_size, fusion_dim]
        output = self.classifier(fused)
        
        return output