import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class CustomModel(nn.Module):
    def __init__(self, embed_dim, nlp_dim, label_num, 
                 hidden_dim=512, fusion_dim=256, dropout=0.3):
        super().__init__()
        
        self.label_num = label_num
        
        # ========== 序列编码分支（主分支，测试时使用） ==========
        self.seq_encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ========== 文本编码分支（仅训练时使用） ==========
        self.text_encoder = nn.Sequential(
            nn.Linear(nlp_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ========== 跨模态融合模块 ==========
        # 使用注意力机制动态融合两个模态
        self.fusion_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, fusion_dim),
            nn.Tanh(),
            nn.Linear(fusion_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # ========== 分类头 ==========
        # 序列独立分类器（测试时使用）
        self.seq_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, label_num)
        )
        
        # 融合分类器（训练时使用，当文本可用时）
        self.fusion_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, label_num)
        )
        
        # ========== 针对长尾分布的类别权重嵌入 ==========
        self.class_weight_layer = nn.Parameter(torch.ones(label_num))
        
    def forward(self, seq_embed, text_embed, return_features=False):
        """
        Args:
            seq_embed: 序列向量 [batch, embed_dim]
            text_embed: 文本向量 [batch, nlp_dim]
            return_features: 是否返回中间特征（用于知识蒸馏）
        """
        # 编码序列特征
        seq_features = self.seq_encoder(seq_embed)
        
        # 检测文本向量是否为零向量（测试时或训练集中无标签样本）
        text_mask = (text_embed.abs().sum(dim=1) > 1e-6).float().unsqueeze(1)
        
        if self.training and text_mask.sum() > 0:
            # 训练模式且存在有效文本向量
            text_features = self.text_encoder(text_embed)
            
            # 跨模态注意力融合
            concat_features = torch.cat([seq_features, text_features], dim=1)
            attention_weights = self.fusion_attention(concat_features)  # [batch, 2]
            
            # 加权融合
            fused_features = (attention_weights[:, 0:1] * seq_features + 
                            attention_weights[:, 1:2] * text_features * text_mask)
            
            # 两个分类器的输出
            seq_logits = self.seq_classifier(seq_features)
            fusion_logits = self.fusion_classifier(fused_features)
            
            if return_features:
                return {
                    'seq_logits': seq_logits,
                    'fusion_logits': fusion_logits,
                    'seq_features': seq_features,
                    'text_features': text_features,
                    'fused_features': fused_features,
                    'text_mask': text_mask
                }
            else:
                return seq_logits, fusion_logits, text_mask
        else:
            # 测试模式或无有效文本
            seq_logits = self.seq_classifier(seq_features)
            
            if return_features:
                return {
                    'seq_logits': seq_logits,
                    'seq_features': seq_features
                }
            else:
                return seq_logits


class MultiLabelLoss(nn.Module):
    """针对长尾分布的多标签损失函数"""
    def __init__(self, label_num, class_freq=None, use_focal=True, 
                 focal_gamma=2.0, distill_alpha=0.5, distill_temp=4.0):
        super().__init__()
        self.label_num = label_num
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.distill_alpha = distill_alpha
        self.distill_temp = distill_temp
        
        # 根据类别频率计算权重（处理长尾分布）
        if class_freq is not None:
            # 使用有效样本数重加权
            effective_num = 1.0 - torch.pow(0.9999, class_freq)
            weights = (1.0 - 0.9999) / effective_num
            weights = weights / weights.sum() * label_num
            self.register_buffer('class_weights', weights).cuda()
        else:
            self.register_buffer('class_weights', torch.ones(label_num))
            if torch.cuda.is_available():
                self.class_weights = self.class_weights.cuda()
    
    def focal_loss(self, logits, targets, weights):
        """Focal Loss for multi-label classification"""
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        if self.use_focal:
            pt = torch.exp(-bce_loss)
            focal_weight = (1 - pt) ** self.focal_gamma
            loss = focal_weight * bce_loss
        else:
            loss = bce_loss
        
        # 应用类别权重
        weighted_loss = loss * weights.unsqueeze(0)
        return weighted_loss.mean()
    
    def distillation_loss(self, student_logits, teacher_logits, temperature):
        """知识蒸馏损失"""
        student_soft = torch.sigmoid(student_logits / temperature)
        teacher_soft = torch.sigmoid(teacher_logits / temperature)
        
        # KL散度
        kl_loss = F.kl_div(
            torch.log(student_soft + 1e-8),
            teacher_soft,
            reduction='batchmean'
        )
        return kl_loss * (temperature ** 2)
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: 模型输出，可以是：
                     - 单个tensor (测试时)
                     - tuple of (seq_logits, fusion_logits, text_mask) (训练时)
            targets: 真实标签 [batch, label_num]
        """
        if isinstance(outputs, tuple):
            seq_logits, fusion_logits, text_mask = outputs
            
            # 主任务损失：序列分类器的损失（所有样本）
            seq_loss = self.focal_loss(seq_logits, targets, self.class_weights)
            
            # 辅助损失：融合分类器的损失（有文本的样本）
            if text_mask.sum() > 0:
                fusion_loss = self.focal_loss(fusion_logits, targets, self.class_weights)
                
                # 知识蒸馏损失：让序列分类器学习融合分类器
                distill_loss = self.distillation_loss(
                    seq_logits, fusion_logits.detach(), self.distill_temp
                )
                
                # 综合损失
                total_loss = (seq_loss + 
                             self.distill_alpha * fusion_loss + 
                             self.distill_alpha * distill_loss)
            else:
                total_loss = seq_loss
            
            return total_loss
        else:
            # 仅计算序列分类器损失
            return self.focal_loss(outputs, targets, self.class_weights)