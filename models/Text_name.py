import torch
import torch.nn as nn
import torch.nn.functional as F

class Text_Name(nn.Module):
    def __init__(self, embed_dim, label_num, nlp_dim=768, hidden_dim=1280, dropout=0.3):
        super(Text_Name, self).__init__()
        self.force_use_text_for_graph = False
        # ===== 特征投影层 =====
        self.protein_projection = nn.Linear(embed_dim, hidden_dim)
        self.text_projection = nn.Linear(nlp_dim, hidden_dim)
        
        # ===== 跨模态注意力机制 =====
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # ===== 蛋白质特征增强网络 =====
        self.protein_enhancer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ===== 融合网络 =====
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ===== 分类头 =====
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, label_num)
        )
        
        # ===== Layer Norm =====
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, protein_embed, text_embed=None):
        """
        Args:
            protein_embed: [batch_size, embed_dim]
            text_embed: [batch_size, nlp_dim]
        """
        # 特征投影
        protein_feat = self.protein_projection(protein_embed)  # [B, hidden_dim]
        protein_feat = self.layer_norm(protein_feat)
        
        if (self.force_use_text_for_graph or self.training) and text_embed is not None and not torch.all(text_embed == 0):
            # 训练模式：使用文本信息
            text_feat = self.text_projection(text_embed)  # [B, hidden_dim]
            text_feat = self.layer_norm(text_feat)
            
            # 添加序列维度用于注意力机制
            protein_feat_seq = protein_feat.unsqueeze(1)  # [B, 1, hidden_dim]
            text_feat_seq = text_feat.unsqueeze(1)  # [B, 1, hidden_dim]
            
            # 跨模态注意力：用文本增强蛋白质特征
            enhanced_protein, _ = self.cross_attention(
                query=protein_feat_seq,
                key=text_feat_seq,
                value=text_feat_seq
            )
            enhanced_protein = enhanced_protein.squeeze(1)  # [B, hidden_dim]
            
            # 残差连接
            protein_feat = protein_feat + enhanced_protein
            
            # 特征融合
            combined = torch.cat([protein_feat, text_feat], dim=1)  # [B, hidden_dim*2]
            fused_features = self.fusion_network(combined)  # [B, hidden_dim]
        else:
            # 测试模式：只使用蛋白质特征
            fused_features = self.protein_enhancer(protein_feat)
        
        # 分类
        logits = self.classifier(fused_features)
        
        return logits