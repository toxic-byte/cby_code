from time import process_time_ns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os 
import sys
# from models.Text_name import Text_Name
# from models.Text_distill import ProteinFunctionPredictor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.Text_modify import CustomModel

embed_dim=640
nlp_dim=768
label_num=255

# 创建示例输入
dummy_input1 = torch.randn(32, embed_dim)
dummy_input2 = torch.zeros(32, nlp_dim)

# 前向传播
# model_text_name=Text_Name(embed_dim,label_num)
model_text_name=CustomModel(embed_dim,nlp_dim,label_num)
model_text_name.train()
model_text_name.force_use_text_for_graph = True

# 创建 TensorBoard 写入器
writer = SummaryWriter('runs/model_visualization')

# 将模型和输入数据添加到 TensorBoard
writer.add_graph(model_text_name, (dummy_input1, dummy_input2))

# 关闭写入器
writer.close()

print("模型已添加到 TensorBoard，运行以下命令查看：")
print("tensorboard --logdir=runs")