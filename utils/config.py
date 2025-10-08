import os
import os
import torch
import random
import numpy as np

def setup_environment():
    """设置环境变量和随机种子"""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["WANDB_DISABLED"] = "true"
    
    seed = 7
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed

def get_config(run_mode="full", text_mode="def"):
    """获取配置参数"""
    config = {
        'run_mode': run_mode,
        'text_mode': text_mode,
        'nlp_path': '/e/cuiby/huggingface/hub/models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract/snapshots/d673b8835373c6fa116d6d8006b33d48734e305d',
        'nlp_dim': 768,
        'embed_dim': 640,
        'MAXLEN': 2048,
        'cache_dir': 'embeddings_cache',
        'output_path': 'eval/function_linear',
        'tax_path': 'data/Original/train_taxonomy.tsv',
        'obo_path': 'data/Original/go-basic.obo',
        'ia_path': 'data/Original/IA.txt',
        'embed_path': 'FunP/embed',
        'batch_size_train':32,
        'batch_size_test': 64,
        'learning_rate': 5e-4,
        'epoch_num': 200,
        'patience': 15,
        'step_size': 10,
        'gamma': 0.6,
        'projection_dim': 256,
        'dropout': 0.3,
        'alpha': 0.5,  # 对比学习权重
        'temperature': 0.07,
        'hidden_dim':512,
        'weight_decay':0.01,
        'contrastive_type': 'supcon'  # 或 'supcon'
    }
    
    if run_mode == "sample":
        config['train_path'] = "data/sampled/cafa5_train_10percent.txt"
        config['test_path'] = "data/sampled/cafa5_test_10percent.txt"
    elif run_mode == "full":
        config['train_path'] = "data/sampled/cafa5_train_100.txt"
        config['test_path'] = "data/sampled/cafa5_test_100.txt"
    
    os.makedirs(config['cache_dir'], exist_ok=True)
    return config