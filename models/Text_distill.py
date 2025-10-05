import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class ProteinFunctionPredictor(nn.Module):
    def __init__(self, embed_dim, nlp_dim, label_num, hidden_dim=1024):
        super().__init__()
        
        # 序列分支(主分支,测试时使用)
        self.sequence_branch = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 文本分支(辅助分支,仅训练时使用)
        self.text_branch = nn.Sequential(
            nn.Linear(nlp_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 融合层(训练时使用)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 分类头
        self.sequence_classifier = nn.Linear(hidden_dim, label_num)
        self.fusion_classifier = nn.Linear(hidden_dim, label_num)
        
    def forward(self, seq_embed, text_embed=None, mode='train'):
        # 序列特征
        seq_feat = self.sequence_branch(seq_embed)
        seq_output = self.sequence_classifier(seq_feat)
        
        if mode == 'test' or text_embed is None:
            return seq_output
        
        # 训练模式:使用文本信息
        text_feat = self.text_branch(text_embed)
        
        # 融合特征
        fused_feat = torch.cat([seq_feat, text_feat], dim=-1)
        fused_feat = self.fusion(fused_feat)
        fused_output = self.fusion_classifier(fused_feat)
        
        return seq_output, fused_output, seq_feat, text_feat


class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2, temperature=3.0):
        super().__init__()
        self.alpha = alpha  # 融合分类器权重
        self.beta = beta    # 序列分类器权重
        self.gamma = gamma  # 蒸馏损失权重
        self.temperature = temperature
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, seq_output, fused_output, seq_feat, text_feat, labels):
        # 1. 融合分类器的监督损失(教师)
        loss_fusion = self.bce(fused_output, labels)
        
        # 2. 序列分类器的监督损失(学生)
        loss_seq = self.bce(seq_output, labels)
        
        # 3. 知识蒸馏损失:序列分类器学习融合分类器的软标签
        soft_fused = torch.sigmoid(fused_output / self.temperature)
        soft_seq = F.logsigmoid(seq_output / self.temperature)
        loss_kd = -torch.mean(soft_fused * soft_seq)
        
        # 4. 特征对齐损失:让序列特征接近文本特征
        loss_feat = F.mse_loss(seq_feat, text_feat.detach())
        
        total_loss = (self.alpha * loss_fusion + 
                     self.beta * loss_seq + 
                     self.gamma * loss_kd +
                     0.1 * loss_feat)
        
        return total_loss, {
            'fusion': loss_fusion.item(),
            'seq': loss_seq.item(),
            'kd': loss_kd.item(),
            'feat': loss_feat.item()
        }

def create_model_and_optimizer(config, label_num):
    """创建模型、损失函数和优化器"""
    model = ProteinFunctionPredictor(config['embed_dim'], config['nlp_dim'], label_num).cuda()
    criterion = DistillationLoss(alpha=0.5, beta=0.3, gamma=0.2, temperature=3.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    
    return model, criterion, optimizer, scheduler


def train_one_epoch(model, train_dataloader, criterion, optimizer, epoch, epoch_num, key):
    """训练一个epoch"""
    model.train()
    loss_mean = 0
    
    for batch_idx, batch_data in tqdm(enumerate(train_dataloader), 
                                    desc=f"Epoch {epoch+1}/{epoch_num} Training",
                                    total=len(train_dataloader)):
        optimizer.zero_grad()
        
        batch_embeddings = batch_data['embedding'].cuda()
        batch_labels = batch_data['labels'].cuda()
        batch_nlp = batch_data['nlp_embedding'].cuda()
        
        seq_output, fused_output, seq_feat, text_feat = model(
            batch_embeddings, batch_nlp, mode='train')
        
        loss, loss_dict = criterion(
            seq_output, fused_output, seq_feat, text_feat, batch_labels)
        
        loss.backward()
        optimizer.step()
        
        loss_mean += loss.item()
        
        if (batch_idx + 1) % 100 == 0:
            print('{}  Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                key, epoch + 1, epoch_num, batch_idx + 1,
                len(train_dataloader),
                loss_mean / (batch_idx + 1)))
    
    return loss_mean / len(train_dataloader)


