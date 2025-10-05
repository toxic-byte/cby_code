import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class ProteinFunctionPredictor(nn.Module):
    def __init__(self, embed_dim, nlp_dim, label_num, hidden_dim=1024):
        super().__init__()
        
        # 序列编码器
        self.seq_encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 文本编码器
        self.text_encoder = nn.Sequential(
            nn.Linear(nlp_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 投影头用于对比学习
        self.seq_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        self.text_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, label_num),
            nn.Sigmoid()
        )
        
        self.temperature = 0.07  # 对比学习温度参数
        
    def forward(self, seq_embed, text_embed=None,mode="train"):
        # 编码序列
        seq_features = self.seq_encoder(seq_embed)
        
        if mode=="train":
            # 训练模式：使用对比学习
            text_features = self.text_encoder(text_embed)
            
            # 投影到对比学习空间
            seq_proj = F.normalize(self.seq_projector(seq_features), dim=-1)
            text_proj = F.normalize(self.text_projector(text_features), dim=-1)
            
            # 分类预测
            logits = self.classifier(seq_features)
            
            return {
                'logits': logits,
                'seq_proj': seq_proj,
                'text_proj': text_proj,
                'seq_features': seq_features,
                'text_features': text_features
            }
        elif mode=="test":
            # 测试模式：仅使用序列
            logits = self.classifier(seq_features)
            return logits
    
    def contrastive_loss(self, seq_proj, text_proj):
        """
        计算InfoNCE对比损失
        seq_proj: [batch_size, proj_dim]
        text_proj: [batch_size, proj_dim]
        """
        batch_size = seq_proj.shape[0]
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(seq_proj, text_proj.T) / self.temperature  # [batch_size, batch_size]
        
        # 对角线是正样本对,其他是负样本对
        labels = torch.arange(batch_size).cuda()
        
        # 双向对比损失
        loss_seq2text = F.cross_entropy(sim_matrix, labels)
        loss_text2seq = F.cross_entropy(sim_matrix.T, labels)
        
        return (loss_seq2text + loss_text2seq) / 2


def create_model_and_optimizer(config, label_num):
    """创建模型、损失函数和优化器"""
    model = ProteinFunctionPredictor(
        config['embed_dim'], 
        config['nlp_dim'], 
        label_num,
        hidden_dim=config.get('hidden_dim', 1024)
    ).cuda()
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config['step_size'], 
        gamma=config['gamma']
    )

        # 检查哪些参数会被训练
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")

    return model, criterion, optimizer, scheduler


def train_one_epoch(model, train_dataloader, criterion, optimizer, epoch, epoch_num, key, 
                    contrastive_weight=0.5):
    """训练一个epoch"""
    model.train()
    loss_mean = 0
    cls_loss_mean = 0
    contrastive_loss_mean = 0
    
    for batch_idx, batch_data in tqdm(enumerate(train_dataloader), 
                                    desc=f"Epoch {epoch+1}/{epoch_num} Training",
                                    total=len(train_dataloader)):
        optimizer.zero_grad()
        
        batch_embeddings = batch_data['embedding'].cuda()
        batch_labels = batch_data['labels'].cuda()
        batch_nlp = batch_data['nlp_embedding'].cuda()
        
        # 前向传播
        outputs = model(batch_embeddings, batch_nlp,mode="train")
        
        # 分类损失
        cls_loss = criterion(outputs['logits'], batch_labels)
        
        # 对比学习损失
        contrastive_loss = model.contrastive_loss(
            outputs['seq_proj'], 
            outputs['text_proj']
        )
        
        # 总损失
        loss = cls_loss + contrastive_weight * contrastive_loss
        
        loss.backward()

        # 检查梯度是否正常
        # total_norm = 0
        # for p in model.parameters():
        #     if p.grad is not None:
        #         param_norm = p.grad.data.norm(2)
        #         total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** 0.5
        # print(f'Gradient norm: {total_norm}')
        
        optimizer.step()
        
        loss_mean += loss.item()
        cls_loss_mean += cls_loss.item()
        contrastive_loss_mean += contrastive_loss.item()
        
        if (batch_idx + 1) % 100 == 0:
            print('{}  Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} (Cls: {:.4f}, Contrastive: {:.4f})'.format(
                key, epoch + 1, epoch_num, batch_idx + 1,
                len(train_dataloader),
                loss_mean / (batch_idx + 1),
                cls_loss_mean / (batch_idx + 1),
                contrastive_loss_mean / (batch_idx + 1)))
    
    return loss_mean / len(train_dataloader),


