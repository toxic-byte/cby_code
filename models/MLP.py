import torch
import torch.nn as nn
import math

#Classifier
class Esm_mlp_2(nn.Module):
    def __init__(self, input_size, output_size):
        super(Esm_mlp_2, self).__init__()
        self.fc1 = nn.Linear(input_size, 2*input_size)  
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2*input_size, output_size)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x
    

import torch
import torch.nn as nn
import math

# Classifier - 3层MLP
class Esm_mlp_3(nn.Module):
    def __init__(self, input_size, output_size, hidden_size1=None, hidden_size2=None, dropout_rate=0.2):
        super(Esm_mlp_3, self).__init__()
        
        # 设置隐藏层大小，如果没有指定则使用默认比例
        if hidden_size1 is None:
            hidden_size1 = 2 * input_size  # 第一隐藏层：2倍输入维度
        if hidden_size2 is None:
            hidden_size2 = input_size      # 第二隐藏层：与输入相同维度
        
        self.fc1 = nn.Linear(input_size, hidden_size1)  
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_size2, output_size)
        torch.nn.init.kaiming_normal_(self.fc3.weight)
        self.fc3.bias.data.fill_(0.01)
        
        # 打印模型结构信息
        print(f"3-Layer MLP Model: {input_size} -> {hidden_size1} -> {hidden_size2} -> {output_size}")
        print(f"Dropout rate: {dropout_rate}")

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x