import torch
from collections import OrderedDict

def analyze_model_generic(model_path):
    """通用模型分析函数"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print("=" * 50)
    print("通用模型分析")
    print("=" * 50)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        
        print(f"\n模型层信息:")
        print("-" * 30)
        
        # 按层类型分组
        layers = OrderedDict()
        for name, param in state_dict.items():
            layer_type = name.split('.')[0] if '.' in name else 'other'
            if layer_type not in layers:
                layers[layer_type] = []
            layers[layer_type].append((name, param.shape, param.numel()))
        
        # 打印每层信息
        total_params = 0
        for layer_type, params in layers.items():
            print(f"\n{layer_type}:")
            layer_params = 0
            for name, shape, numel in params:
                print(f"  {name}: {shape} ({numel} 参数)")
                layer_params += numel
                total_params += numel
            print(f"  该层总参数: {layer_params}")
        
        print(f"\n模型总参数: {total_params}")
        
        # 推断模型结构
        print(f"\n结构推断:")
        weight_layers = [p for p in state_dict.items() if 'weight' in p[0]]
        if weight_layers:
            print("权重层:")
            for name, param in weight_layers:
                print(f"  {name}: {param.shape}")
    
    return checkpoint

# 使用通用分析
model_path = "/e/cuiby/paper/cby_code/ckpt/cafa5/linear/20251004005720Mlp_esm2_t30_150M_UR50D_biological_process_best.pt"
analyze_model_generic(model_path)